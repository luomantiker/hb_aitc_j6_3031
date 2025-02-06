"""This is a prototype feature."""
import logging
import os
import types
from typing import Dict

import numpy as np
import torch
from jinja2 import Environment, PackageLoader
from torch.optim.optimizer import Optimizer

import horizon_plugin_pytorch as hpp
from horizon_plugin_pytorch.utils.typeguard import typechecked
from .mask_generator import (
    MaskGenerator,
    SemistructedMaskGenerator,
    UnstructedMaskGenerator,
)
from .permutation import Permutation

logger = logging.getLogger(__name__)


class SemistructedPruner(object):
    @typechecked
    def __init__(
        self,
        mask_generator: MaskGenerator = None,
        enable_permutation: bool = False,
    ):
        """
        Semi-structed pruner. This is a prototype feature.

        Args:
            mask_generator (MaskGenerator, optional): mask generator.
                Defaults to None.
            enable_permutation (bool, optional): Whether enable permutation.
                Defaults to False.
        """
        logger.warning(
            "Pruner is a prototype feature, Please use with caution!"
        )
        self.mask_generator = (
            SemistructedMaskGenerator()
            if mask_generator is None
            else mask_generator
        )
        self.layer_prune_ratio = self.mask_generator.n / self.mask_generator.m
        self.enable_permutation = enable_permutation
        self.prune_module_type_to_params = {
            # torch.nn.Linear: ["weight"], # sparse 1 * 1 conv is useless
            # torch.nn.Conv1d: ["weight"],
            torch.nn.Conv2d: ["weight"],
        }
        self.prune_module_type = tuple(self.prune_module_type_to_params.keys())

    def _is_prunable(self, mod):
        """Judge if a module is prunable."""
        if not isinstance(mod, self.prune_module_type):
            return False

        if isinstance(mod, torch.nn.Conv2d) and (
            mod.stride != (1, 1)
            or (mod.weight.shape[2] == 1 and mod.weight.shape[3] == 1)
        ):
            return False

        return True

    def _update_prune_modules(self):
        """Update prune_modules."""
        for name, mod in self.model.named_modules():
            if (
                self.allowed_module_names is not None
                and name not in self.allowed_module_names
            ):
                continue

            if not self._is_prunable(mod):
                continue

            if name not in self.disallowed_module_names:
                self.prune_modules.append((name, mod))

    def _add_mask_in_prune_modules(self):
        """Add mask buffer in prune_modules."""
        prune_param_num = 0
        prune_module_num = 0
        for mod_name, mod in self.prune_modules:
            prune_params = self.prune_module_type_to_params[type(mod)]
            is_prunable = False

            for param_name, param in mod.named_parameters():
                if param_name not in prune_params or not param.requires_grad:
                    continue
                if not self.mask_generator.is_maskable(param):
                    continue

                is_prunable = True
                logger.info(f"{param_name} of {mod_name} is prunable.")

                # register buffer for mask
                mask = torch.ones_like(param).bool()
                buffer_name = param_name.split(".")[-1]
                mod.register_buffer("__%s_prune_mask" % buffer_name, mask)

                # need to store in CPU memory?
                pruned_param = torch.zeros_like(param)
                mod.register_buffer(
                    "__%s_prune_param" % buffer_name, pruned_param
                )

                self.prune_params.append(
                    (mod_name, mod, param_name, param, mask, pruned_param)
                )
                prune_param_num += mask.numel() * self.layer_prune_ratio
            if is_prunable:
                prune_module_num += 1
        total_param_num = sum(p.numel() for p in self.model.parameters())
        logger.info(
            f"Total prune ratio: {prune_param_num / total_param_num: .2%}. "
            f"Total prune module num: {prune_module_num}."
        )

    def _update_prune_optimizer(self) -> Optimizer:
        """
        Set pruned params and their gradient in optimizer.

        Returns:
            Optimizer: optimizer with pruned params.
        """
        # keep raw step function
        self.optimizer._step = self.optimizer.step

        def new_step(optimizer, *args, **kwargs):
            # apply mask to gradient and weight
            with torch.no_grad():
                for _, _, _, param, mask, _ in self.prune_params:
                    if param.grad is not None:
                        param.grad.mul_(mask)

            ret = optimizer._step(*args, **kwargs)

            with torch.no_grad():
                for _, _, _, param, mask, _ in self.prune_params:
                    param.mul_(mask)
            return ret

        self.optimizer.step = types.MethodType(new_step, self.optimizer)

    def update_mask(self):
        """Update mask buffer. This is a prototype feature."""
        with torch.no_grad():
            # permute params to keep more important params.
            if self.enable_permutation:
                self.permutation.permute()

            # generate mask
            for (
                _,
                _,
                _,
                param,
                mask,
                pruned_param,
            ) in self.prune_params:
                if mask.sum() < mask.numel():
                    param.add_(pruned_param)

                mask.set_(self.mask_generator(param).bool())

                # update pruned param
                pruned_param.set_((param * (~mask)))
                param.mul_(mask)
        logger.info("Pruning mask is updated.")

    @typechecked
    def prune(
        self,
        model: torch.nn.Module,
        optimizer: Optimizer,
        pconfig_dict: Dict = None,
    ):
        """
        Prune model. This is a prototype feature.

        Args:
            model (torch.nn.Module): model for pruning.
            optimizer (Optimizer): optimizer for pruning.
            pconfig_dict (Dict, optional): Prune config dict.
                Defaults to {
                    "allowed_module_names": None,
                    "disallowed_module_names": None,
                }.
                allowed_module_names: only modules in this list
                    will be pruned.
                disallowed_module_names: modules in this list
                    won't be pruned.
        """
        self.model = model
        self.optimizer = optimizer
        self.pconfig_dict = {} if pconfig_dict is None else pconfig_dict
        self.allowed_module_names = self.pconfig_dict.get(
            "allowed_module_names", None
        )
        self.disallowed_module_names = self.pconfig_dict.get(
            "disallowed_module_names", None
        )
        if self.disallowed_module_names is None:
            self.disallowed_module_names = []
        assert (
            self.allowed_module_names is None
            or len(self.disallowed_module_names) == 0
        ), (
            "Don't set allowed_module_names and disallowed_module_names "
            "at same time."
        )
        self.prune_modules = []
        self.prune_params = []

        self._update_prune_modules()
        self._add_mask_in_prune_modules()

        if self.enable_permutation:
            self.permutation = Permutation(
                self.model, self.prune_params, self.mask_generator
            )

        self._update_prune_optimizer()
        self.update_mask()

    def _dump_sensitivity(self, sensitivity_dict, output_dir):
        """Dump sensitivity results."""
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        sorted_sensitivity = sorted(
            sensitivity_dict.items(), key=lambda x: x[1]
        )

        output_path = os.path.join(output_dir, "sensitivity.txt")
        with open(output_path, "w", encoding="utf-8") as f:
            for key, value in sorted_sensitivity:
                f.write(f"{key} {value: .2f}\n")

    @typechecked
    def sensitivity(
        self,
        model: torch.nn.Module,
        eval_func: types.FunctionType,
        output_dir: str = "sparse_sensitivity",
    ):
        """
        Calculate sparse sensitivity. This is a prototype feature.

        Args:
            model (torch.nn.Module): model for calculating sparse sensitivity.
            eval_func (types.FunctionType): function for evaluation.
            output_dir (str, optional): path to save results.
                Defaults to "sparse_sensitivity".
        """
        sensitivity_dict = {}
        for mod_name, mod in model.named_modules():
            if not self._is_prunable(mod):
                continue

            prune_params = self.prune_module_type_to_params[type(mod)]
            for param_name, param in mod.named_parameters():
                if param_name not in prune_params:
                    continue
                if not self.mask_generator.is_maskable(param):
                    continue
                with torch.no_grad():
                    mask = self.mask_generator(param).bool()
                    pruned_param = param * (~mask)
                    param.mul_(mask)

                    metric = eval_func()
                    key = f"{mod_name}.{param_name}"
                    sensitivity_dict[key] = float(metric)
                    logger.info(
                        f"{key} is sparsed, evaluation metric: {metric:.2f}"
                    )

                    # restore param
                    param.add_(pruned_param)

        self._dump_sensitivity(sensitivity_dict, output_dir)


class UnstructedPruner(object):
    def __init__(self):
        """Unstructed pruner. This is a prototype feature."""
        logger.warning(
            "Pruner is a prototype feature, Please use with caution!"
        )
        self.mask_generator = UnstructedMaskGenerator()
        self.prune_module_type_to_params = {
            torch.nn.Linear: ["weight"],
            torch.nn.Conv1d: ["weight"],
            torch.nn.Conv2d: ["weight"],
        }

        self.prune_module_type = tuple(self.prune_module_type_to_params.keys())

    def _update_prune_modules(self):
        """Update prune_modules."""
        for name, mod in self.model.named_modules():
            ratio = self.pconfig_dict.get(name, self.pconfig_dict.get("", 0.0))
            if isinstance(mod, self.prune_module_type) and ratio > 0:
                self.prune_modules.append((name, mod, ratio))

    def _add_mask_in_prune_modules(self):
        """Add mask buffer in prune_modules."""
        prune_param_num = 0
        prune_module_num = 0
        for mod_name, mod, ratio in self.prune_modules:
            prune_params = self.prune_module_type_to_params[type(mod)]
            is_prunable = False

            for param_name, param in mod.named_parameters():
                if param_name not in prune_params or not param.requires_grad:
                    continue

                is_prunable = True
                logger.info(
                    f"{param_name} of {mod_name} is prunable."
                    f"prune ratio: {ratio: .2%}"
                )

                # register buffer for mask
                mask = torch.ones_like(param).bool()
                buffer_name = param_name.split(".")[-1]
                mod.register_buffer("__%s_prune_mask" % buffer_name, mask)

                # need to store in CPU memory?
                pruned_param = torch.zeros_like(param)
                mod.register_buffer(
                    "__%s_prune_param" % buffer_name, pruned_param
                )

                self.prune_params.append(
                    (
                        mod_name,
                        mod,
                        param_name,
                        param,
                        mask,
                        pruned_param,
                        ratio,
                    )
                )
                prune_param_num += mask.numel() * ratio
            if is_prunable:
                prune_module_num += 1
        total_param_num = sum(p.numel() for p in self.model.parameters())
        logger.info(
            f"Total prune ratio: {prune_param_num / total_param_num: .2%}. "
            f"Total prune module num: {prune_module_num}."
        )

    def _update_prune_optimizer(self) -> Optimizer:
        """
        Set pruned params and their gradient in optimizer.

        Returns:
            Optimizer: optimizer with pruned params.
        """
        # keep raw step function
        self.optimizer._step = self.optimizer.step

        def new_step(optimizer, *args, **kwargs):
            self.step += 1
            step_ratio = 1 - ((1 - self.step / self.gmp_step) ** 3)

            # apply mask to gradient and weight
            with torch.no_grad():
                for _, _, _, param, mask, pruned_param, r in self.prune_params:
                    if (
                        self.step % self.gmp_update_step == 1
                        and step_ratio <= 1
                    ):
                        if mask.sum() < mask.numel():
                            param.add_(pruned_param)
                        mask.set_(
                            self.mask_generator(param, r * step_ratio).bool()
                        )
                        pruned_param.set_((param * (~mask)))
                        param.mul_(mask)

                    if param.grad is not None:
                        param.grad.mul_(mask)

            ret = optimizer._step(*args, **kwargs)

            with torch.no_grad():
                for _, _, _, param, mask, _, _ in self.prune_params:
                    param.mul_(mask)
            return ret

        self.optimizer.step = types.MethodType(new_step, self.optimizer)

    def update_mask(self):
        """Update mask buffer. This is a prototype feature."""
        with torch.no_grad():
            # generate mask
            for (
                _,
                _,
                _,
                param,
                mask,
                pruned_param,
                ratio,
            ) in self.prune_params:
                if mask.sum() < mask.numel():
                    param.add_(pruned_param)

                mask.set_(self.mask_generator(param, ratio).bool())

                # update pruned param
                pruned_param.set_((param * (~mask)))
                param.mul_(mask)
        logger.info("Pruning mask is updated.")

    @typechecked
    def prune(
        self,
        model: torch.nn.Module,
        optimizer: Optimizer,
        pconfig_dict: Dict = None,
    ):
        """
        Prune model. This is a prototype feature.

        Args:
            model (torch.nn.Module): model for pruning.
            optimizer (Optimizer): optimizer for pruning.
            pconfig_dict (Dict, optional): Prune config dict.
                Defaults to {"": 0.5}.
        """
        self.model = model
        self.optimizer = optimizer
        self.pconfig_dict = pconfig_dict if pconfig_dict else {"": 0.5}
        self.prune_modules = []
        self.prune_params = []
        self.step = 0
        self.gmp_step = self.pconfig_dict.get("gmp_step", 1)
        self.gmp_update_times = self.pconfig_dict.get("gmp_update_times", 1)
        self.gmp_update_step = self.gmp_step // self.gmp_update_times
        assert (
            self.gmp_update_step > 0
        ), "gmp_step should be larger than gmp_update_times."

        self._update_prune_modules()
        self._add_mask_in_prune_modules()
        self._update_prune_optimizer()
        self.update_mask()

    def _dump_sensitivity(self, sensitivity_dict, output_dir):
        """Dump sensitivity results."""
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        env = Environment(
            loader=PackageLoader(
                "horizon_plugin_pytorch", "prune/sensitivity_templates"
            )
        )
        template = env.get_template("sensitivity_template")

        # create a soft link of css and echarts.js
        src_dir = os.path.join(hpp.__path__[0], "prune/sensitivity_templates")
        css_src_dir = os.path.join(src_dir, "style.css")
        echarts_src_dir = os.path.join(src_dir, "echarts.min.js")
        css_dst_dir = os.path.join(output_dir, "style.css")
        echarts_dst_dir = os.path.join(output_dir, "echarts.js")
        for src, dst in zip(
            (css_src_dir, echarts_src_dir), (css_dst_dir, echarts_dst_dir)
        ):
            if not os.path.exists(dst):
                # if the origin file doesn't exists while the soft link exists,
                # delete the useless soft link
                if os.path.islink(dst):
                    os.remove(dst)
                # check the origin file must exists
                assert os.path.exists(src), f"Can not find file {src}!"
                os.symlink(src, dst)

        out = template.render(
            sensitivity_dict=sensitivity_dict,
        )

        output_path = os.path.join(output_dir, "sensitivity.html")
        with open(output_path, "w", encoding="utf-8") as f:
            f.writelines(out)

    @typechecked
    def sensitivity(
        self,
        model: torch.nn.Module,
        eval_func: types.FunctionType,
        output_dir: str = "sparse_sensitivity",
    ):
        """
        Calculate sparse sensitivity. This is a prototype feature.

        Args:
            model (torch.nn.Module): model for calculating sparse sensitivity.
            eval_func (types.FunctionType): function for evaluation.
            output_dir (str, optional): path to save results.
                Defaults to "sparse_sensitivity".
        """
        sensitivity_dict = {}
        for mod_name, mod in model.named_modules():
            if not isinstance(mod, self.prune_module_type):
                continue
            prune_params = self.prune_module_type_to_params[type(mod)]
            for param_name, param in mod.named_parameters():
                if param_name not in prune_params:
                    continue
                with torch.no_grad():
                    for ratio in np.arange(0.1, 1, 0.1):
                        mask = self.mask_generator(param, ratio).bool()
                        pruned_param = param * (~mask)
                        param.mul_(mask)

                        metric = eval_func()
                        key = f"{mod_name}.{param_name}"
                        if key not in sensitivity_dict:
                            sensitivity_dict[key] = []
                        sensitivity_dict[key].append(float(metric))
                        logger.info(
                            f"{key} in {ratio:.1f} sparse ratio, evaluation "
                            f"metric: {metric:.2f}"
                        )

                        # restore param
                        param.add_(pruned_param)

        self._dump_sensitivity(sensitivity_dict, output_dir)
