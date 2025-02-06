"""This is a prototype feature."""
import itertools
import logging
import math
import pickle
import random
import time
from os import path
from typing import Dict, List

import numpy as np
import torch
from torch import nn

from horizon_plugin_pytorch.fx.fx_helper import (
    convert_fx_node_name,
    get_fx_node_input_output,
    is_fx_node_name_match_module_name,
)
from horizon_plugin_pytorch.quantization.fx.graph_module import (
    GraphModuleWithAttr,
)
from horizon_plugin_pytorch.quantization.quantization_mappings import (
    get_qat_module_mappings,
)
from horizon_plugin_pytorch.quantization.quantize_fx import QuantizationTracer
from horizon_plugin_pytorch.utils.typeguard import typechecked
from .mask_generator import MaskGenerator, SemistructedMaskGenerator

logger = logging.getLogger(__name__)


class PermutationSearcher(object):
    @typechecked
    def __init__(self, mask_generator: MaskGenerator = None):
        """
        Permutation searcher.

        Args:
            mask_generator (MaskGenerator, optional): mask generator.
                Defaults to None.
        """
        self.mask_generator = (
            SemistructedMaskGenerator()
            if mask_generator is None
            else mask_generator
        )
        self.m = self.mask_generator.m
        self.n = self.mask_generator.n
        self.perturbations = 0
        self.perturbation_limit = 0
        self.stripe_set = None
        self.stripe_set_config = None
        self.master_unique_permutation_list = {}

    def _get_sum_after_unstructed_pruning(self, mat):
        """Get the sum of magnitudes after pruning."""
        mask = self.mask_generator(mat)
        return mat.abs().mul(mask).sum()

    def _get_sum_improvement_after_column_swapping(self, mat, col_a, col_b):
        """Get the sum improvement of magnitudes after column swapping."""
        group_a_start_idx = col_a // self.m * self.m
        group_a_end_idx = group_a_start_idx + self.m
        group_b_start_idx = col_b // self.m * self.m
        group_b_end_idx = group_b_start_idx + self.m

        sum_before = self._get_sum_after_unstructed_pruning(
            mat[..., group_a_start_idx:group_a_end_idx]
        ) + self._get_sum_after_unstructed_pruning(
            mat[..., group_b_start_idx:group_b_end_idx]
        )

        mat[..., [col_a, col_b]] = mat[..., [col_b, col_a]]

        sum_after = self._get_sum_after_unstructed_pruning(
            mat[..., group_a_start_idx:group_a_end_idx]
        ) + self._get_sum_after_unstructed_pruning(
            mat[..., group_b_start_idx:group_b_end_idx]
        )

        mat[..., [col_a, col_b]] = mat[..., [col_b, col_a]]

        return sum_after - sum_before

    def _is_canonical(self, perm, col):
        """Check if adding a column to a permutation would keep its form."""
        # new group
        if len(perm) % self.m == 0:
            for val in range(col):
                if val not in perm:
                    return False
            # keep first element sorted between groups
            return col > perm[-self.m]

        # not a new group, keep sorted in the group
        return col > perm[-1]

    def _generate_unique_combinations(
        self, built_permutation, remaining_columns, full_permutation_list
    ):
        """Build a unique permutation one column index at a time.

        order of columns within a group and order of groups don't matter,
        keep them sorted to avoid depulicated permutation.
        """
        if len(remaining_columns) == 0:
            full_permutation_list.append(np.copy(built_permutation))
        else:
            for c in range(len(remaining_columns)):
                col_to_add = remaining_columns[c]

                if self._is_canonical(built_permutation, col_to_add):
                    # add the column to the running permutation, remove it from
                    # remaining columns
                    built_permutation.append(col_to_add)
                    remaining_columns.pop(c)
                    self._generate_unique_combinations(
                        built_permutation,
                        remaining_columns,
                        full_permutation_list,
                    )
                    # remove the most recent column and put it back on the
                    # remaining column list where we found it (sorted)
                    remaining_columns.insert(c, built_permutation.pop(-1))

    def _generate_all_unique_combinations(self, c):
        """Generate all unique combinations.

        order of columns within a group and order of groups don't matter,
        keep them sorted to avoid depulicated permutation.
        """
        if len(self.master_unique_permutation_list) == 0 and path.exists(
            "master_list.pkl"
        ):
            with open("master_list.pkl", "rb") as cache:
                self.master_unique_permutation_list = pickle.load(cache)

        if (c, self.m) not in self.master_unique_permutation_list:
            permutation_list = []
            self._generate_unique_combinations(
                [0], list(range(1, c)), permutation_list
            )
            self.master_unique_permutation_list[(c, self.m)] = permutation_list

            with open("master_list.pkl", "wb") as cache:
                pickle.dump(self.master_unique_permutation_list, cache)

        unique_permutations = self.master_unique_permutation_list[(c, self.m)]

        return unique_permutations

    def _generate_stripe_groups(self, num_stripes, num_stripes_in_group):
        """Generate all possible stripe groups."""
        return set(
            itertools.combinations(
                list(range(num_stripes)), num_stripes_in_group
            )
        )

    def _collect_stripes(self, mat, stripes):
        """Gather stripes from a larger matrix into a single matrix."""
        subset = torch.zeros((mat.shape[0], len(stripes) * self.m))
        for s, stripe in enumerate(stripes):
            subset[..., s * self.m : s * self.m + self.m] = mat[
                ..., stripe * self.m : stripe * self.m + self.m
            ]
        return subset

    def _get_unique_combinations_num(self, c):
        assert c % self.m == 0
        g = c // self.m
        return math.factorial(c) // (
            int(math.pow(math.factorial(self.m), g)) * math.factorial(g)
        )

    def _search_matrix(self, mat):
        """Search the entire matrix exhaustively."""
        if self._get_unique_combinations_num(mat.shape[1]) > 100000:
            # too many permutations
            return mat, list(range(mat.shape[1])), 0.0
        permutation_list = self._generate_all_unique_combinations(mat.shape[1])
        mat_permutations = torch.cat([mat[:, p] for p in permutation_list])
        mat_permutations_mask = self.mask_generator(mat_permutations)
        sums = (
            mat_permutations.abs()
            .mul(mat_permutations_mask)
            .view(len(permutation_list), -1, mat.shape[1])
            .sum((1, 2))
        )
        max_index = torch.argmax(sums)
        imp = sums[max_index] - sums[0]
        best_permutation = permutation_list[max_index]
        return mat[:, best_permutation], best_permutation, imp

    def _build_stripe_map(
        self,
        mat,
        stripe_group_size,
        stripe_map,
        stripe_ids,
        perm_map,
        used_stripes,
    ):
        """Build the stripe map."""
        num_stripes_in_group = stripe_group_size // self.m

        if (
            self.stripe_set is None
            or self.stripe_set_config is None
            or self.stripe_set_config != (self.m, num_stripes_in_group)
        ):
            num_stripes = mat.shape[1] // self.m
            assert self.m * num_stripes == mat.shape[1]

            self.stripe_set = self._generate_stripe_groups(
                num_stripes, num_stripes_in_group
            )
            self.stripe_set_config = (self.m, num_stripes_in_group)

        for i, s in enumerate(self.stripe_set):
            sg = []
            need_update = i >= len(stripe_map)
            for stripe in s:
                sg.append(stripe)
                if stripe in used_stripes:
                    need_update = True

            if i >= len(stripe_map):
                stripe_ids.append(sg)
                stripe_map.append(0.0)
                perm_map.append(list(range(self.m * num_stripes_in_group)))

            if need_update:
                subset = self._collect_stripes(mat, sg)
                _, permutation, improvement = self._search_matrix(subset)
                stripe_map[i] = improvement
                perm_map[i] = permutation

        return stripe_map, stripe_ids, perm_map

    def _apply_stripe_group_permutation(
        self, stripe_group_permutation, stripe_group, permutation
    ):
        """Apply the stripe group permutation to the entire permutation."""
        new_permutation = permutation.copy()
        for subset_idx in range(len(stripe_group_permutation)):
            dst_stripe_idx = stripe_group[subset_idx // self.m]
            dst_col_idx = subset_idx % self.m

            subset_val = stripe_group_permutation[subset_idx]
            src_stripe_idx = stripe_group[subset_val // self.m]
            src_col_idx = subset_val % self.m

            new_permutation[
                dst_stripe_idx * self.m + dst_col_idx
            ] = permutation[src_stripe_idx * self.m + src_col_idx]

        return new_permutation

    def _use_stripe_map(
        self, mat, stripe_map, stripe_ids, perm_map, permutation
    ):
        """Use stripe map."""
        used_stripes = []
        # from high improvement to low improvement
        idx = np.flip(np.argsort(stripe_map))

        for i in range(len(idx)):
            stripe_group_id = idx[i]
            perm = perm_map[stripe_group_id].copy()

            if stripe_map[stripe_group_id] <= 0.0003:
                # perturbations
                if (
                    len(used_stripes) == 0
                    and self.perturbations < self.perturbation_limit
                ):
                    self.perturbations += 1
                    stripe_group_id = random.choice(idx)
                    perm = perm_map[stripe_group_id].copy()
                    # a little easier to escape from
                    src = np.random.randint(len(perm) // 2)
                    dst = len(perm) // 2 + np.random.randint(len(perm) // 2)
                    perm[src], perm[dst] = perm[dst], perm[src]
                else:
                    break

            stripe_group = stripe_ids[stripe_group_id]

            # don't work on stripes we've already touched
            if any(s in used_stripes for s in stripe_group):
                continue

            # apply the permutation we've already found to this stripe group
            subset = self._collect_stripes(mat, stripe_group)
            sub_result = subset[..., perm]
            permutation = self._apply_stripe_group_permutation(
                perm, stripe_group, permutation
            )

            for s, stripe in enumerate(stripe_group):
                g = perm[s * self.m : s * self.m + self.m]
                # entry 0 a multiple of 4, contiguous values
                if g[0] % self.m != 0 or any(
                    g[c] != g[c - 1] + 1 for c in range(1, self.m)
                ):
                    used_stripes.append(stripe)

                mat[
                    ..., stripe * self.m : stripe * self.m + self.m
                ] = sub_result[..., s * self.m : s * self.m + self.m]

        return mat, stripe_map, stripe_ids, used_stripes, permutation

    def _exhaustive_search(self, matrix, stripe_group_size, permutation):
        """Search best permutation exhaustively."""
        self.perturbations = 0
        self.stripe_set = None
        self.stripe_set_config = None
        mat = matrix.clone().detach()

        num_stripes = mat.shape[1] // self.m
        num_stripes_in_group = stripe_group_size // self.m
        # if the matrix is too large , subdivide
        if (
            num_stripes_in_group < num_stripes
            and math.comb(num_stripes, num_stripes_in_group) > 3000
        ):
            stripe_split = int(matrix.shape[1] / 2 // self.m)
            col_split = stripe_split * self.m
            (
                mat[:, :col_split],
                permutation[:col_split],
            ) = self._exhaustive_search(
                mat[:, :col_split],
                stripe_group_size,
                permutation[:col_split],
            )
            (
                mat[:, col_split:],
                permutation[col_split:],
            ) = self._exhaustive_search(
                mat[:, col_split:],
                stripe_group_size,
                permutation[col_split:],
            )
            # mat, permutation = self._exhaustive_search(
            #     mat, 2 * self.m, permutation
            # )
            return mat, permutation

        # small enough to optimize the entire matrix at once
        if stripe_group_size < matrix.shape[1]:
            stripe_map = []
            stripe_ids = []
            perm_map = []
            used_stripes = []

            self._generate_all_unique_combinations(stripe_group_size)

            while True:
                stripe_map, stripe_ids, perm_map = self._build_stripe_map(
                    mat,
                    stripe_group_size,
                    stripe_map,
                    stripe_ids,
                    perm_map,
                    used_stripes,
                )
                (
                    mat,
                    stripe_map,
                    stripe_ids,
                    used_stripes,
                    permutation,
                ) = self._use_stripe_map(
                    mat,
                    stripe_map,
                    stripe_ids,
                    perm_map,
                    permutation,
                )

                if len(used_stripes) == 0:
                    break
        else:
            # no stripe group, single iteration
            mat, permutation, _ = self._search_matrix(matrix)

        return mat, permutation

    @typechecked
    def search(self, matrix: torch.Tensor, options: Dict = None):
        """
        Search good permutation for unstructed pruning.

        Args:
            matrix (torch.Tensor): params to be pruned.
            options (Dict, optional): search options. Defaults to None.
        """
        if options is None:
            options = {}
        if "strategy" not in options:
            options["strategy"] = "exhaustive"

        if options["strategy"] == "exhaustive":
            if "stripe_group_size" not in options:
                options["stripe_group_size"] = 8
            if "escape_attempts" not in options:
                options["escape_attempts"] = 10
        elif options["strategy"] == "progressive channel swap":
            if "progressive_search_time_limit" not in options:
                options["progressive_search_time_limit"] = 60
            if "improvement_threshold" not in options:
                options["improvement_threshold"] = 1e-9

        assert matrix.dim() == 2
        mat = matrix.clone().detach()
        permutation = list(range(mat.shape[1]))

        if options["strategy"] == "progressive channel swap":
            start_time = time.perf_counter()
            while (
                time.perf_counter() - start_time
                < options["progressive_search_time_limit"]
            ):
                col_a = np.random.randint(mat.shape[1])
                col_b = np.random.randint(mat.shape[1])
                if col_a // self.m == col_b // self.m:
                    # two col in a group
                    continue
                improvement = self._get_sum_improvement_after_column_swapping(
                    mat, col_a, col_b
                )
                if improvement > options["improvement_threshold"]:
                    mat[..., [col_a, col_b]] = mat[..., [col_b, col_a]]
                    permutation[col_a], permutation[col_b] = (
                        permutation[col_b],
                        permutation[col_a],
                    )
        elif options["strategy"] == "exhaustive":
            self.perturbation_limit = options["escape_attempts"]
            _, permutation = self._exhaustive_search(
                mat,
                options["stripe_group_size"],
                list(range(mat.shape[1])),
            )

        return permutation


def _replicate_sequence(sequence, replications):
    """Replicate a permutation to a multiple of channel counts."""
    replicated_sequence = []

    for rep in range(replications):
        offset = len(sequence) * rep
        for c in sequence:
            replicated_sequence.append(c + offset)

    return replicated_sequence


class Permutation(object):
    # these module types may be the target of permutations,
    # have potentially sparse weights or are attributes with no parents
    permutation_target_mod_types = [
        "torch.nn.modules.conv.Conv1d",
        "torch.nn.modules.conv.Conv2d",
        "torch.nn.modules.linear.Linear",
        "torch.nn.modules.linear.LazyLinear",
        "torch.nn.modules.linear.NonDynamicallyQuantizableLinear",
        "torch.nn.modules.activation.MultiheadAttention",
        "get_attr",
    ]

    # these module types have parameters that must be permuted along K as
    # well as need to pass the permutation thru to parents' K
    permute_k_and_passthru_mod_types = [
        "torch.nn.modules.batchnorm.BatchNorm2d",
        "torch.nn.modules.normalization.LayerNorm",
        "torch.nn.modules.instancenorm.InstanceNorm2d",
        "torch.nn.modules.batchnorm.SyncBatchNorm",
    ]

    # these module types are not permuted, but must pass any permutation
    # seen by a child's C or passed-thru K to the parents' K
    simple_passthru_mod_types = [
        "torch.nn.modules.activation.ReLU6",
        "torch.nn.modules.activation.ReLU",
        "torch.nn.modules.dropout.Dropout",
        "torch.nn.modules.dropout.Dropout1d",
        "torch.nn.modules.dropout.Dropout2d",
        "torch.nn.modules.dropout.Dropout3d",
        "torch.nn.modules.dropout.AlphaDropout",
        "torch.nn.modules.dropout.FeatureAlphaDropout",
        "torch.nn.modules.pooling.MaxPool2d",
        "torch.nn.modules.pooling.AdaptiveAvgPool2d",
        "torch.nn.modules.pooling.AvgPool2d",
        "torch.nn.modules.activation.Hardsigmoid",
        "torch.nn.modules.activation.Hardswish",
        "torch.nn.modules.activation.GELU",
        "torch.nn.modules.normalization.LocalResponseNorm",
        "torch.nn.modules.activation.Softmin",
        "torch.nn.modules.activation.Softmax",
        "torch.nn.modules.activation.Softmax2d",
        "torch.nn.modules.activation.LogSoftmax",
        "torch.nn.modules.activation.AdaptiveLogSoftmaxWithLoss",
        "torch.nn.modules.activation.SiLU",
        "torch.nn.modules.activation.Sigmoid",
        "concat",
        "torch.nn.modules.flatten.Flatten",
    ]

    # these module types cannot be permuted safely (today),
    # and cause neighboring layers to have permutations disabled.
    disallow_permutations_mod_types = [
        # to handle: influence GCD of real children's sibling group
        "torch.nn.modules.normalization.GroupNorm",
        # permute one input along in1_features and the other along in2_features
        "torch.nn.modules.linear.Bilinear",
        # may work OOTB, but need to explicitly handle dimsionality change
        "torch.nn.modules.activation.GLU",
    ]

    @typechecked
    def __init__(
        self,
        model: nn.Module,
        prune_params: List,
        mask_generator: MaskGenerator,
        seed=1,
    ):
        """
        Class to generate param permutation for pruning.

        Args:
            model (nn.Module): model to be pruned.
            prune_params (List): model params to be pruned.
            mask_generator (MaskGenerator): mask generator.
            seed (int, optional): permutation random seed. Defaults to 1.
        """
        self.model = model
        self.prune_params = prune_params
        self.params = []
        self.permuted_c_dim_params = []
        self.permuted_k_dim_params = []
        self.unpermuted_dims = []
        self.group_data = {
            "next_sibling_group_id": 0,
            "next_coparent_group_id": 0,
            "sibling_groups": {},
            "sibling_group_permutations": {},
            "sibling_group_c_params": {},
            "skipped_sibling_groups": {},
            "coparent_groups": {},
            "skipped_coparent_groups": {},
        }
        self.tcpstore_port = 2341
        self.seed = seed
        self._reset_seed()
        self.searcher = PermutationSearcher(mask_generator)

        for mod_name, mod in self.model.named_modules():
            if isinstance(mod, nn.modules.container.Sequential):
                continue

            for param_name, param in mod.named_parameters():
                self.params.append((mod_name, mod, param_name, param))

            # mean and var of bn are not learnable params
            if isinstance(mod, nn.modules.batchnorm.BatchNorm2d):
                mean = self.model.state_dict().get(mod_name + ".running_mean")
                var = self.model.state_dict().get(mod_name + ".running_var")
                assert mean is not None and var is not None, (
                    f"cannot find running_mean or running_var of {mod_name} "
                    f"in model.state_dict()"
                )
                self.params.append((mod_name, mod, "running_mean", mean))
                self.params.append((mod_name, mod, "running_var", var))

    def _reset_seed(self):
        """Set seed to get same permutation among multi-GPU."""
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _build_fx_graph(self):
        """Build model graph with torch fx."""
        tracer = QuantizationTracer([], list(get_qat_module_mappings().keys()))
        graph = tracer.trace(self.model)
        self.model.node_name_to_scope = tracer.node_name_to_scope
        graph_module = GraphModuleWithAttr(self.model, graph)

        fx_graph = {}
        mod_name_type_dict = {}
        mod_name_group_conv_dict = {}
        mod_name_c_dict = {}
        mod_name_k_dict = {}
        for mod_name, mod in self.model.named_modules():
            mod_name_type_dict[mod_name] = str(type(mod)).split("'")[1]
            if isinstance(mod, nn.Conv2d):
                mod_name_c_dict[mod_name] = str(mod.in_channels)
                mod_name_k_dict[mod_name] = str(mod.out_channels)
                mod_name_group_conv_dict[mod_name] = str(mod.groups)
            elif isinstance(mod, nn.Linear):
                mod_name_c_dict[mod_name] = str(mod.in_features)
                mod_name_k_dict[mod_name] = str(mod.out_features)
                mod_name_group_conv_dict[mod_name] = "None"
            elif isinstance(mod, nn.MultiheadAttention):
                mod_name_c_dict[mod_name] = str(mod.embed_dim)
                mod_name_k_dict[mod_name] = str(mod.embed_dim)
                mod_name_group_conv_dict[mod_name] = "None"
            else:
                mod_name_c_dict[mod_name] = "None"
                mod_name_k_dict[mod_name] = "None"
                mod_name_group_conv_dict[mod_name] = "None"

        for node in graph_module.graph.nodes:
            if node.op in ("placeholder", "output"):
                continue
            inputs, outputs = get_fx_node_input_output(node)
            node_name = convert_fx_node_name(
                node.target if node.op == "get_attr" else node.name
            )

            if node.op == "get_attr":

                def _getattr(obj, attr):
                    try:
                        left, right = attr.split(".", 1)
                    except Exception:
                        return getattr(obj, attr)
                    return _getattr(getattr(obj, left), right)

                fx_graph[node_name] = {
                    "inputs": inputs,
                    "outputs": outputs,
                    "mod_type": "get_attr",
                    "groups_param": "None",
                    "attr": _getattr(graph_module, node.target),
                    "c_param": 1,
                    "k_param": -1,
                }

            elif node.op == "call_function":
                fx_graph[node_name] = {
                    "inputs": inputs,
                    "outputs": outputs,
                    "fx_op": "call_function",
                }

                # concat along K can be handled by reducing the size of
                # children's C. see fixup_concats, if no dim arg,
                # default is 0 (handled automatically)
                if (
                    node.target == torch.cat
                    and len(node.args) > 1
                    and node.args[1] == 1
                ):
                    fx_graph[node_name].update(
                        {
                            "fx_op": "call_module",
                            "mod_type": "concat",
                            "groups_param": "N/A",
                            "c_param": "N/A",
                            "k_param": "N/A",
                        }
                    )

            elif node.op == "call_method":
                fx_graph[node_name] = {
                    "inputs": inputs,
                    "outputs": outputs,
                    "fx_op": "call_method",
                }

            elif node.op == "call_module":
                fx_graph[node_name] = {
                    "inputs": inputs,
                    "outputs": outputs,
                    "fx_op": "call_module",
                    "mod_type": mod_name_type_dict[node.target],
                    "groups_param": mod_name_group_conv_dict[node.target],
                    "c_param": mod_name_c_dict[node.target],
                    "k_param": mod_name_k_dict[node.target],
                }

        self.fx_graph = fx_graph
        logger.info("FX graph is built.")

    def _init_grouped_conv_permutation_flags(self, node):
        """Handle grouped conv to make dimensions match."""

        node_c = int(node.get("c_param"))
        node_k = int(node.get("k_param"))
        node_groups = int(node.get("groups_param"))

        assert node_c % node_groups == 0
        node_c = int(node_c / node_groups)
        node["c_param"] = str(node_c)

        if node_c == 1:
            # group == channel
            if node_groups == node_k:
                # depthwise
                node["k_permutable"] = True
                node["k_permuted"] = False
                node["k_passthru"] = True
                node["is_real"] = False
            # else: G != K, in this condition,
            # permutation along K would be tricky and not likely useful
        elif node_c >= 4 and node_c % 4 == 0:
            # permutations only help if there's more than one 2:4 pruning group
            node["c_permutable"] = True
            node["c_permuted"] = False

    def _insert_mha_out_proj(self, node):
        """Insert out_proj node of MHA, as it is hidden."""
        out_proj_node_name = node + ".out_proj"
        self.fx_graph[out_proj_node_name] = {
            "inputs": [node],
            "outputs": self.fx_graph[node]["outputs"],
            "fx_op": "call_module",
            "mod_type": "torch.nn.modules.linear.Linear",
            "groups_param": "None",
            "c_param": self.fx_graph[node]["c_param"],
            "k_param": self.fx_graph[node]["k_param"],
            "c_permutable": False,
            "k_permutable": True,
            "c_permuted": False,
            "k_permuted": False,
            "k_passthru": False,  # pass a K permutation to its inputs
            "is_real": True,
            "sibling_group_id": None,
            "coparent_group_id": None,
        }
        self.fx_graph[node].update(
            {
                "outputs": [out_proj_node_name],
                "k_permutable": False,
                "c_permutable": True,
            }
        )

        for node_name, cur_node in self.fx_graph.items():
            inputs = cur_node["inputs"]
            if node_name != out_proj_node_name and node in inputs:
                inputs.remove(node)
                inputs.append(out_proj_node_name)
                cur_node["inputs"] = inputs

    def _init_permutation_flags(self):
        """Set the permutation flags based on node's module type and params."""
        multi_head_attention_nodes = []

        for node_name, node in self.fx_graph.items():
            node_mod_type = node.get("mod_type")

            if node_mod_type is None:
                continue

            node.update(
                {
                    "c_permutable": False,
                    "k_permutable": False,
                    "c_permuted": False,
                    "k_permuted": False,
                    "k_passthru": False,  # pass a K permutation to its inputs
                    "is_real": False,
                    "sibling_group_id": None,
                    "coparent_group_id": None,
                }
            )

            # update each node to be more permissive if supported
            if node_mod_type in Permutation.permutation_target_mod_types:
                node["is_real"] = True
                node_groups = node.get("groups_param")
                if node_groups in ["None", "1"]:
                    # get_attr and non-multi-group call_module
                    node["c_permutable"] = True
                    node["k_permutable"] = True
                else:
                    # grouped conv
                    self._init_grouped_conv_permutation_flags(node)

            elif node_mod_type in Permutation.permute_k_and_passthru_mod_types:
                node["k_permutable"] = True
                node["k_passthru"] = True

            elif node_mod_type in Permutation.simple_passthru_mod_types:
                node["k_passthru"] = True

            elif node_mod_type in Permutation.disallow_permutations_mod_types:
                node["is_real"] = True
                node["c_param"] = 1
                node["k_param"] = 1
                node["groups_param"] = 1

            elif "activation" in node_mod_type:
                # do not change the shape
                node["k_passthru"] = True

            else:
                node["is_real"] = True
                # dummy entries:
                node["c_param"] = 1
                node["k_param"] = 1
                node["groups_param"] = 1

            # MHA nodes only handle the in_proj, need to add out_proj nodes
            # explicitly. keep track to iterate and change fx_graph.
            if (
                node_mod_type
                == "torch.nn.modules.activation.MultiheadAttention"
            ):
                multi_head_attention_nodes.append(node_name)

        for node in multi_head_attention_nodes:
            self._insert_mha_out_proj(node)

    def _find_node_real_inputs(self, node, found_inputs):
        """Find the real inputs of a node."""

        if "real_inputs" in node.keys():
            return found_inputs.union(node["real_inputs"])

        for input in node["inputs"]:
            if input in self.fx_graph:  # not the input node
                if self.fx_graph[input].get("is_real", False):
                    found_inputs.add(input)
                else:
                    found_inputs = self._find_node_real_inputs(
                        self.fx_graph[input], found_inputs
                    )

        return found_inputs

    def _find_real_inputs(self):
        """Find the real inputs of all nodes in the graph."""
        for _, node in self.fx_graph.items():
            real_inputs = self._find_node_real_inputs(node, set())
            node["real_inputs"] = sorted(real_inputs)

    def _find_node_real_outputs(self, node, found_outputs):
        """Find the real outputs of a node."""

        if "real_outputs" in node.keys():
            return found_outputs.union(node["real_outputs"])

        for output in node["outputs"]:
            if output in self.fx_graph:  # not the output node
                if self.fx_graph[output].get("is_real", False):
                    found_outputs.add(output)
                else:
                    found_outputs = self._find_node_real_outputs(
                        self.fx_graph[output], found_outputs
                    )

        return found_outputs

    def _find_real_outputs(self):
        """Find the real outputs of all nodes in the graph."""
        for node_name in reversed(list(self.fx_graph.keys())):
            # find from back to front to use the already saved 'real_outputs'
            node = self.fx_graph[node_name]
            real_outputs = self._find_node_real_outputs(node, set())
            node["real_outputs"] = sorted(real_outputs)

    def _find_node_siblings(self, node, all_siblings):
        """Find all siblings of a node."""
        siblings = set()
        for input in node.get("real_inputs"):
            outputs = self.fx_graph.get(input).get("real_outputs")
            for output in outputs:
                siblings.add(output)

        new_siblings = siblings.difference(all_siblings)
        all_siblings.update(new_siblings)

        for new_sibling in new_siblings:
            all_siblings = self._find_node_siblings(
                self.fx_graph.get(new_sibling), all_siblings
            )

        return all_siblings

    def _find_node_coparents(self, node, all_coparents):
        """Find all coparents of a node."""
        coparents = set()
        for output in node.get("real_outputs"):
            inputs = self.fx_graph.get(output).get("real_inputs")
            for input in inputs:
                coparents.add(input)

                # coparents are used to restrict what nodes can be permuted
                # along C, so we need to track if the current inputs also
                # pass their K permutations up.
                if self.fx_graph[input]["k_passthru"]:
                    k_passthru_inputs = self.fx_graph[input]["real_inputs"]
                    for k_passthru_input in k_passthru_inputs:
                        coparents = coparents.union(
                            self._find_node_coparents(
                                self.fx_graph[k_passthru_input], coparents
                            )
                        )

        new_coparents = coparents.difference(all_coparents)
        all_coparents.update(new_coparents)

        for new_coparent in new_coparents:
            all_coparents = self._find_node_coparents(
                self.fx_graph[new_coparent], all_coparents
            )

        return all_coparents

    def _make_sibling_coparent_groups(self):
        """Make sibling and coparent groups of all real nodes."""
        for node_name, node in self.fx_graph.items():
            if not node.get("is_real", False):
                continue
            sibling_group_id = node["sibling_group_id"]
            if sibling_group_id is None:
                # need to make a new sibling group for this node
                all_siblings = self._find_node_siblings(node, {node_name})
                # deterministic order for DDP setups
                all_siblings = sorted(all_siblings)
                sibling_group_id = self.group_data["next_sibling_group_id"]
                self.group_data["sibling_groups"][
                    sibling_group_id
                ] = all_siblings
                self.group_data["next_sibling_group_id"] += 1

                sibling_group_c_params = []
                for sibling in all_siblings:
                    self.fx_graph[sibling][
                        "sibling_group_id"
                    ] = sibling_group_id
                    sibling_c_param = int(self.fx_graph[sibling]["c_param"])
                    sibling_group_c_params.append(sibling_c_param)

                # if grouped convolutions make the input channels different
                # among siblings different sizes, restrict the permutation atom
                # to the greatest common divisor so it can be tiled as needed
                # for each sibling.
                sibling_group_c_param = str(
                    np.gcd.reduce(sibling_group_c_params)
                )
                self.group_data["sibling_group_c_params"][
                    sibling_group_id
                ] = sibling_group_c_param
                self.group_data["skipped_sibling_groups"][
                    sibling_group_id
                ] = None

            coparent_group_id = node["coparent_group_id"]
            if coparent_group_id is None:
                all_coparents = self._find_node_coparents(node, {node_name})
                coparent_group_id = self.group_data["next_coparent_group_id"]
                self.group_data["coparent_groups"][
                    coparent_group_id
                ] = all_coparents
                self.group_data["next_coparent_group_id"] += 1
                self.group_data["skipped_coparent_groups"][
                    coparent_group_id
                ] = None

                for coparent in all_coparents:
                    self.fx_graph[coparent][
                        "coparent_group_id"
                    ] = coparent_group_id

    def _fixup_concats(self):
        """Fixup concat.

        Concat along K can be handled by reducing the size of children's C.
        """
        for _, node in self.fx_graph.items():
            if node.get("module_type") == "concat":
                real_inputs = node["real_inputs"]
                # some concats are at the front of networks
                if len(real_inputs) == 0:
                    continue

                inputs_k_params = []
                for input in real_inputs:
                    input_k_param = int(self.fx_graph[input]["k_param"])
                    inputs_k_params.append(input_k_param)
                    self.fx_graph[input]["allow_k_mismatch"] = "concat op"

                # if grouped convolutions make the input channels different
                # among siblings different sizes, restrict the permutation atom
                # to the greatest common divisor so it can be tiled as needed
                # for each sibling.
                output_c_param = str(np.gcd.reduce(inputs_k_params))

                sibling_group_id = -1
                for output in node["real_outputs"]:
                    sibling_group_id = self.fx_graph[output][
                        "sibling_group_id"
                    ]
                    self.fx_graph[output]["c_param"] = output_c_param

                old_output_c_param = self.group_data["sibling_group_c_params"][
                    sibling_group_id
                ]
                self.group_data["sibling_group_c_params"][
                    sibling_group_id
                ] = output_c_param

                # fixup this node's dimensions
                # use the functionality of grouped convolutions
                node["c_param"] = output_c_param
                node["k_param"] = old_output_c_param
                node["groups_param"] = str(
                    int(old_output_c_param) // int(output_c_param)
                )

    def _enforce_dimension_agreement(self):
        """Check nodes' channel against parents and children. e.g. flatten."""
        for node_name, node in self.fx_graph.items():
            if not node.get("is_real", False):
                continue

            node_c = int(node["c_param"])

            if node["groups_param"] not in ["1", "None"]:
                node_c = node_c * int(node["groups_param"])

            if len(node["real_inputs"]) == 0:
                node["c_permutable"] = False
            else:
                for real_input in node["real_inputs"]:
                    input_k = int(self.fx_graph[real_input]["k_param"])
                    allow_k_mismatch = self.fx_graph[real_input].get(
                        "allow_k_mismatch"
                    )

                    if (
                        allow_k_mismatch is None
                        and input_k >= 0
                        and node_c != input_k
                    ):
                        self.fx_graph[node_name]["c_permutable"] = False
                        self.fx_graph[real_input]["k_permutable"] = False

            if len(node["real_outputs"]) == 0:
                node["k_permutable"] = False

    def _propagate_sibling_group(self, all_siblings):
        """Propagate permutation flags in sibling group.

        Check a sibling group for ability to be permuted, disallow all siblings
        and coparents if there's an issue.
        """
        made_change = False
        c_permutable = True
        for sibling in all_siblings:
            c_permutable = (
                c_permutable and self.fx_graph[sibling]["c_permutable"]
            )
            if not c_permutable:
                break

        if not c_permutable:
            for sibling in all_siblings:
                if self.fx_graph[sibling]["c_permutable"]:
                    self.fx_graph[sibling]["c_permutable"] = False
                    made_change = True

                # if the node cannot passthru, disable inputs k permutation
                if self.fx_graph[sibling]["k_passthru"]:
                    continue

                for sibling_input in self.fx_graph[sibling]["real_inputs"]:
                    if (
                        self.fx_graph[sibling_input]["k_permutable"]
                        or self.fx_graph[sibling_input]["k_passthru"]
                    ):
                        self.fx_graph[sibling_input]["k_permutable"] = False
                        self.fx_graph[sibling_input]["k_passthru"] = False
                        made_change = True

        return made_change

    def _propagate_coparent_group(self, all_coparents):
        """Propagate permutation flags in coparent group.

        Check a coparent group for ability to be permuted, disallow all fellow
        coparents if there's an issue.
        """
        made_change = False
        k_permutable = True
        for coparent in all_coparents:
            k_permutable = k_permutable and (
                self.fx_graph[coparent]["k_permutable"]
                or self.fx_graph[coparent]["k_passthru"]
            )
            if not k_permutable:
                break

        if not k_permutable:
            for coparent in all_coparents:
                if (
                    self.fx_graph[coparent]["k_permutable"]
                    or self.fx_graph[coparent]["k_passthru"]
                ):
                    self.fx_graph[coparent]["k_permutable"] = False
                    self.fx_graph[coparent]["k_passthru"] = False
                    made_change = True

                # outputs of coparents can't be permuted along C
                for coparent_output in self.fx_graph[coparent]["real_outputs"]:
                    if self.fx_graph[coparent_output]["c_permutable"]:
                        self.fx_graph[coparent_output]["c_permutable"] = False
                        made_change = True

        return made_change

    def _propagate_permutation_flags(self):
        """Propagate permutation flags.

        1. Disallow sibling groups which have different c_permutable flags.
        2. Disallow coparent groups which have different k_permutable flags.
        """
        made_change = True
        while made_change:
            made_change = False

            for _, node in self.fx_graph.items():
                # input layers can't be permuted along c
                if node.get("inputs") is None or (
                    "x" in node.get("inputs")
                    and node.get("c_permutable", False)
                ):
                    made_change = True
                    node["c_permutable"] = False

                # output layers can't be permuted along k
                if node.get("outputs") is None or (
                    "output" in node.get("outputs")
                    and node.get("k_permutable", False)
                ):
                    made_change = True
                    node["k_permutable"] = False
                    node["k_passthru"] = False

                if node.get("is_real", False):
                    all_siblings = self.group_data["sibling_groups"][
                        node["sibling_group_id"]
                    ]

                    made_change = (
                        self._propagate_sibling_group(all_siblings)
                        or made_change
                    )

                    all_coparents = self.group_data["coparent_groups"][
                        node["coparent_group_id"]
                    ]
                    made_change = (
                        self._propagate_coparent_group(all_coparents)
                        or made_change
                    )
        logger.info("Preparations before finding permutation are done.")

    def _skip_sibling_group(self, sibling_group_id, reason):
        """Keep track of sibling groups that cannot be permuted."""
        # grab a input to get the coparent group id
        sibling_group = self.group_data["sibling_groups"][sibling_group_id]
        a_input = self.fx_graph[list(sibling_group)[0]]["real_inputs"][0]
        coparent_group_id = self.fx_graph[a_input]["coparent_group_id"]

        self.group_data["skipped_sibling_groups"][sibling_group_id] = reason
        self.group_data["skipped_coparent_groups"][coparent_group_id] = reason

    def _find_node_sparse_params(self, node_name):
        """Find prune params and reshape to 2D for a node."""
        sparse_params = None

        for mod_name, _, _, param, _, _ in self.prune_params:
            if is_fx_node_name_match_module_name(node_name, mod_name):
                sparse_params = torch.zeros_like(param)
                sparse_params.copy_(param)

        if sparse_params is not None:
            shape = sparse_params.shape
            # 1d-tensor
            if len(shape) == 1:
                sparse_params = sparse_params.view(1, shape[0])
            # linear
            elif len(shape) == 2:
                sparse_params = sparse_params.view(shape[0], shape[1])
            # conv1d
            elif len(shape) == 3:
                sparse_params = (
                    sparse_params.permute(0, 2, 1)
                    .contiguous()
                    .view(-1, shape[1])
                )
            # conv2d
            elif len(shape) == 4:
                sparse_params = (
                    sparse_params.permute(2, 3, 0, 1)
                    .contiguous()
                    .view(-1, shape[1])
                )

        return sparse_params

    def _find_sibling_group_sparse_params(
        self, sibling_group, sibling_group_c_param
    ):
        """Find all sparse params in a sibling group.

        Find all sparse weights in a sibling group to serve as input to the
        permutation search.
        """
        matrix_group = None
        for sibling in sibling_group:
            sparse_params = self._find_node_sparse_params(sibling)

            if sparse_params is None:
                continue

            assert sparse_params.shape[1] % sibling_group_c_param == 0, (
                f"sibling {sibling}'s weight channel={sparse_params.shape[1]} "
                f"must be multiple of group's c param {sibling_group_c_param}"
            )

            sparse_params = torch.reshape(
                sparse_params, (-1, sibling_group_c_param)
            )

            if matrix_group is None:
                matrix_group = sparse_params
            else:
                matrix_group = torch.cat((matrix_group, sparse_params), dim=0)

        return matrix_group

    def _find_matrix_group_permutation(self, matrix_group):
        """Find a good permutation for a matrix."""
        num_channels = matrix_group.shape[1]
        group_permutation = list(range(num_channels))

        original_magnitude = (torch.abs(matrix_group)).sum(dtype=torch.float64)
        pruned_magnitude = self.searcher._get_sum_after_unstructed_pruning(
            matrix_group
        )
        diff_ratio = (
            abs(original_magnitude - pruned_magnitude) / original_magnitude
        )

        # if magnitude is large enough, don't need to permute
        if diff_ratio <= 1e-3:
            return group_permutation, False

        # call the permutation search CUDA kernels.
        search_options = {}
        # No.1 Strategy: Exhaustive Search
        search_options["strategy"] = "exhaustive"
        search_options["stripe_group_size"] = 8
        search_options["escape_attempts"] = 100
        # No.2 Strategy: Progressive Channel Swap Search
        # search_options['strategy'] = 'progressive channel swap'
        # search_options['progressive_search_time_limit'] = 10
        # search_options['improvement_threshold'] = 1e-9

        # search time is too long for large channel num
        # change to Progressive Channel Swap Search.
        if num_channels > 1024:
            search_options = {}
            search_options["strategy"] = "progressive channel swap"
            search_options["progressive_search_time_limit"] = 120
            search_options["improvement_threshold"] = 1e-9

        group_permutation = self.searcher.search(matrix_group, search_options)

        return group_permutation, True

    def _find_sibling_group_permutation(self, sibling_group_id):
        """Find a good permutation for a sibling group."""
        self._reset_seed()

        sibling_group = self.group_data["sibling_groups"][sibling_group_id]
        sibling_group_c_param = int(
            self.group_data["sibling_group_c_params"][sibling_group_id]
        )

        if sibling_group_c_param % 4 != 0 or sibling_group_c_param < 8:
            self._skip_sibling_group(
                sibling_group_id,
                f"Useless C: {sibling_group_c_param}",
            )
            return

        matrix_group = self._find_sibling_group_sparse_params(
            sibling_group, sibling_group_c_param
        )

        if matrix_group is None:
            self._skip_sibling_group(sibling_group_id, "Dense")
            return

        logger.info(f"Finding permutation in sibling group {sibling_group_id}")
        group_permutation, found = self._find_matrix_group_permutation(
            matrix_group
        )

        if not found:
            self._skip_sibling_group(sibling_group_id, "Not needed")
            return

        self.group_data["sibling_group_permutations"][
            sibling_group_id
        ] = group_permutation

    def _find_permutations(self):
        """Search for permutations for all sibling groups."""
        sibling_group_num = len(self.group_data["sibling_groups"])
        logger.info(f"sibling group num: {sibling_group_num}")
        for sibling_group_id, sibling_group in self.group_data[
            "sibling_groups"
        ].items():
            search_this_group = True
            if torch.distributed.is_initialized():
                rank = torch.distributed.get_rank()
                world_size = torch.distributed.get_world_size()

                if sibling_group_id % world_size != rank:
                    search_this_group = False

            self.group_data["sibling_group_permutations"][
                sibling_group_id
            ] = None

            if search_this_group:
                if self.fx_graph[list(sibling_group)[0]]["c_permutable"]:
                    self._find_sibling_group_permutation(sibling_group_id)
        logger.info("Permutations are found.")

    def _sync_permutations(self):
        """Sync permutation when using multiple GPUs."""
        if not torch.distributed.is_initialized():
            return
        else:
            torch.distributed.barrier()

        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        dist_store = torch.distributed.TCPStore(
            "127.0.0.1", self.tcpstore_port, world_size, rank == 0
        )

        torch.distributed.barrier()

        for sibling_group_id in sorted(
            self.group_data["sibling_groups"].keys()
        ):
            src_rank = sibling_group_id % world_size

            if src_rank == rank:
                to_send = self.group_data["sibling_group_permutations"].get(
                    sibling_group_id, None
                )
                skip_reason = None
                if to_send is None:
                    skip_reason = self.group_data[
                        "skipped_sibling_groups"
                    ].get(sibling_group_id)
                    if skip_reason is None:
                        to_send = ""
                    else:
                        to_send = "skip"
                else:
                    to_send = ",".join(str(c) for c in to_send)

                dist_store.set(str(sibling_group_id), to_send)
                if skip_reason is not None:
                    dist_store.set(f"skip {sibling_group_id}", skip_reason)

        torch.distributed.barrier()

        for sibling_group_id in sorted(
            self.group_data["sibling_groups"].keys()
        ):
            permutation = dist_store.get(str(sibling_group_id)).decode()

            if permutation == "skip":
                permutation = None
                skip_reason = dist_store.get(
                    f"skip {sibling_group_id}"
                ).decode()
                self._skip_sibling_group(sibling_group_id, skip_reason)
            else:
                if len(permutation) == 0:
                    permutation = None
                else:
                    permutation = [int(c) for c in permutation.split(",")]

            self.group_data["sibling_group_permutations"][
                sibling_group_id
            ] = permutation

        torch.distributed.barrier()
        logger.info("Permutations are synced.")

    def _apply_permutation_in_c_dim(
        self, node_name, permutation_sequence, dryrun
    ):
        """Apply permutation for a node in C dim."""
        if len(permutation_sequence) == 0:
            return False

        is_node_in_sparse_parameters = False
        success_permutation = False
        for mod_name, _, param_name, param, _, _ in self.prune_params:
            if is_fx_node_name_match_module_name(node_name, mod_name):
                is_node_in_sparse_parameters = True
                permutation_to_apply = permutation_sequence
                if param.shape[1] != len(permutation_sequence):
                    # grouped convolutions or concatenated weights
                    if param.shape[1] % len(permutation_sequence) != 0:
                        return False

                    permutation_to_apply = _replicate_sequence(
                        permutation_sequence,
                        param.shape[1] // len(permutation_sequence),
                    )

                if not dryrun:
                    param.data.copy_(param[:, permutation_to_apply, ...])
                    self.permuted_c_dim_params.append(
                        node_name + "." + param_name
                    )

                success_permutation = True

        if not is_node_in_sparse_parameters:
            # if one of siblings is prune module, the node may need to apply
            # the permutation in C dim like its siblings in prune module.
            try:
                for mod_name, _, param_name, param in self.params:
                    if (
                        is_fx_node_name_match_module_name(node_name, mod_name)
                        and param_name == "weight"
                    ):
                        permutation_to_apply = permutation_sequence
                        if param.shape[1] != len(permutation_sequence):
                            # assumed to be grouped convolutions
                            if param.shpae[1] % len(permutation_sequence) != 0:
                                return False

                            permutation_to_apply = _replicate_sequence(
                                permutation_sequence,
                                param.shape[1] // len(permutation_sequence),
                            )

                        if not dryrun:
                            param.data.copy_(
                                param[:, permutation_to_apply, ...]
                            )
                            self.permuted_c_dim_params.append(
                                node_name + "." + param_name
                            )

                        success_permutation = True
            except Exception:
                success_permutation = False

        return success_permutation

    def _permute_attr(self, node_name, permutation_sequence, dryrun):
        """Permute a node's attributes."""
        attr = self.fx_graph[node_name]["attr"]

        found_perm = False
        for dim in range(len(attr.shape)):
            if attr.shape[dim] == len(permutation_sequence):
                if found_perm:
                    # already permuted, but there is another dim
                    # satisfied the permutation length
                    return False

                found_perm = True
                if not dryrun:
                    order = list(range(len(attr.shape)))
                    order[0] = dim
                    order[dim] = 0
                    prmt = tuple(order)

                    # permute the dimension of interest to the front, permute
                    # within that dimension, then reset it.
                    temp_weight = torch.clone(attr)
                    temp_weight = torch.permute(temp_weight, prmt)
                    temp_weight.copy_(temp_weight[permutation_sequence, ...])
                    temp_weight = torch.permute(temp_weight, prmt)
                    attr.data.copy_(temp_weight)

                    self.permuted_k_dim_params.append(
                        node_name + "_" + str(dim)
                    )

        return found_perm

    def _apply_permutation_in_k_dim(
        self, node_name, permutation_sequence, dryrun
    ):
        """Apply permutation for a node in K dim."""
        if len(permutation_sequence) == 0:
            return False

        # permute attribute nodes
        if "attr" in self.fx_graph[node_name]:
            return self._permute_attr(node_name, permutation_sequence, dryrun)

        is_node_in_all_parameters = False

        for mod_name, _, param_name, param in self.params:
            if is_fx_node_name_match_module_name(node_name, mod_name):
                is_node_in_all_parameters = True
                permutation_to_apply = permutation_sequence

                if param.shape[0] != len(permutation_sequence):
                    # grouped convolutions
                    if param.shape[0] % len(permutation_sequence) != 0:
                        return False

                    permutation_to_apply = _replicate_sequence(
                        permutation_sequence,
                        param.shape[0] // len(permutation_sequence),
                    )

                if not dryrun:
                    param.data.copy_(param[permutation_to_apply, ...])
                    self.permuted_k_dim_params.append(
                        node_name + "." + param_name
                    )

        return is_node_in_all_parameters

    def _apply_permutation_in_k_dim_to_children(
        self, node_name, permutation_sequence, dryrun
    ):
        """Apply permutation for a node's outputs in k dim."""
        success = True
        outputs = self.fx_graph[node_name]["outputs"]

        for output in outputs:
            if self.fx_graph[output].get("is_real", False):
                continue

            if self.fx_graph[output].get("mod_type", "None") == "None":
                success = (
                    success
                    and self._apply_permutation_in_k_dim_to_children(
                        output, permutation_sequence, dryrun
                    )
                )
            elif not self.fx_graph[output]["c_permutable"]:
                if (
                    self.fx_graph[output]["k_permutable"]
                    and not self.fx_graph[output]["k_permuted"]
                ):
                    child_permuted = self._apply_permutation_in_k_dim(
                        output, permutation_sequence, dryrun
                    )
                    success = success and child_permuted
                    if not dryrun:
                        self.fx_graph[output]["k_permuted"] = child_permuted
                    assert self.fx_graph[output]["k_passthru"]

                if self.fx_graph[output]["k_passthru"]:
                    success = (
                        success
                        and self._apply_permutation_in_k_dim_to_children(
                            output, permutation_sequence, dryrun
                        )
                    )

        return success

    def _permute_sibling_group(self, sibling_group_id, group_permutation):
        """Apply a permutation to a sibling group."""
        logger.info(f"Permuting sibling group: {sibling_group_id}.")
        sibling_group = self.group_data["sibling_groups"][sibling_group_id]
        success = True
        coparent_group_id = None

        # dryrun is just for test, if there is no problem, the permute will be
        # performed in the second run.
        for dryrun in [True, False]:
            for sibling in sibling_group:
                assert (
                    self.fx_graph[sibling]["c_permutable"]
                    and not self.fx_graph[sibling]["c_permuted"]
                )

                sibling_permuted = self._apply_permutation_in_c_dim(
                    sibling, group_permutation, dryrun
                )
                if dryrun:
                    success = success and sibling_permuted
                else:
                    assert (
                        sibling_permuted
                    ), "shouldn't fail permuting siblings after the dry run"
                    self.fx_graph[sibling]["c_permuted"] = sibling_permuted

                a_input = self.fx_graph[sibling]["real_inputs"][0]
                if coparent_group_id is None:
                    coparent_group_id = self.fx_graph[a_input][
                        "coparent_group_id"
                    ]

            coparents = self.group_data["coparent_groups"][coparent_group_id]
            for coparent in coparents:
                assert (
                    self.fx_graph[coparent]["k_permutable"]
                    and not self.fx_graph[coparent]["k_permuted"]
                )

                coparent_permuted = self._apply_permutation_in_k_dim(
                    coparent, group_permutation, dryrun
                )
                if dryrun:
                    success = success and coparent_permuted
                else:
                    assert (
                        coparent_permuted
                    ), "shouldn't fail permuting coparents after the dry run"
                    self.fx_graph[coparent]["k_permuted"] = coparent_permuted

                children_permuted = (
                    self._apply_permutation_in_k_dim_to_children(
                        coparent, group_permutation, dryrun
                    )
                )
                if dryrun:
                    success = success and children_permuted
                else:
                    assert children_permuted, (
                        "shouldn't fail permuting coparents' children after "
                        "the dry run"
                    )

            if not success:
                self._skip_sibling_group(sibling_group_id, "dryrun_failure")
                break

    def _apply_permutations(self):
        """Apply permutations got in previous step."""
        for sibling_group_id, permutation in self.group_data[
            "sibling_group_permutations"
        ].items():
            if permutation is not None:
                self._permute_sibling_group(sibling_group_id, permutation)
        logger.info("Permutations are applied.")

    def _check_graph_for_unpermuted_nodes(self):
        """Make sure that all permutable params are permuted."""
        for node_name, node in self.fx_graph.items():
            if node.get("c_permutable", False) and not node["c_permuted"]:
                if (
                    node["is_real"]
                    and self.group_data["skipped_sibling_groups"][
                        node["sibling_group_id"]
                    ]
                    is None
                ):
                    self.unpermuted_dims.append(node_name + "_c")

            if node.get("k_permutable", False) and not node["k_permuted"]:
                if (
                    node["is_real"]
                    and self.group_data["skipped_coparent_groups"][
                        node["coparent_group_id"]
                    ]
                    is None
                ):
                    self.unpermuted_dims.append(node_name + "_k")

        # make sure all GPUs agree
        if torch.distributed.is_initialized():
            self.unpermuted_dims = sorted(self.unpermuted_dims)
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
            dist_store = torch.distributed.TCPStore(
                "127.0.0.1", self.tcpstore_port, world_size, rank == 0
            )
            torch.distributed.barrier()

            dist_store.set(str(rank), ",".join(self.unpermuted_dims))
            torch.distributed.barrier()

            if rank == 0:
                my_list = dist_store.get("0").decode()

                for peer in range(1, world_size):
                    peer_list = dist_store.get(str(peer)).decode()
                    assert my_list == peer_list, (
                        f"peer {peer} disagreed with rank 0's list of "
                        f"unpermuted nodes: \n{my_list}\n{peer_list}"
                    )

    def permute(self):
        """Permute params to keep more important params."""
        self._build_fx_graph()
        self._init_permutation_flags()
        self._find_real_inputs()
        self._find_real_outputs()
        self._make_sibling_coparent_groups()
        self._fixup_concats()
        self._enforce_dimension_agreement()
        self._propagate_permutation_flags()
        self._find_permutations()
        self._sync_permutations()
        self._apply_permutations()
        self._check_graph_for_unpermuted_nodes()
        logger.info("Parameters are permuted.")
