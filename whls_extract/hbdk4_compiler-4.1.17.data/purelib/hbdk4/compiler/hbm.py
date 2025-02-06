from asyncio.log import logger
import shlex
import os
import subprocess
import sys
import json
import tempfile
import tarfile
import shutil
import numpy as np
from enum import Enum
from typing import Any, List, Dict, Sequence
import functools

from sympy import false, true
from hbdk4.compiler.utils.process import run_program_redirect_realtime
from hbdk4.compiler.utils.pytree import TreeLikeFuncBase, pickle_object, unpickle_object
import logging
from typing import Optional, Union, Tuple
from hbdk4.compiler.march import March, MarchSeries
from hbdk4.compiler._mlir_libs import _hbrt4_py
from hbdk4.compiler.remote_bpu import RemoteBPU
from hbdk4.compiler.hbtl import is_torch_tensor
from hbdk4.compiler.hbm_tools import hbm_extract_desc, hbm_update_desc


class VariableInputSemantic(Enum):
    Normal = 1
    Pyramid = 2
    Resizer = 3
    ImageY = 4
    ImageUv = 5
    ImageRoi = 6

    @classmethod
    def _from_inner(cls, value: "_hbrt4_py.variable.VariableInputSemanticPy"):
        VariableInputSemanticPy = _hbrt4_py.variable.VariableInputSemanticPy
        if value == VariableInputSemanticPy.Normal:
            return cls.Normal
        if value == VariableInputSemanticPy.Pyramid:
            return cls.Pyramid
        if value == VariableInputSemanticPy.Resizer:
            return cls.Resizer
        if value == VariableInputSemanticPy.ImageY:
            return cls.ImageY
        if value == VariableInputSemanticPy.ImageUv:
            return cls.ImageUv
        if value == VariableInputSemanticPy.ImageRoi:
            return cls.ImageRoi
        raise ValueError(f"Bad value {value}")


class SpecialOperator(Enum):
    Normal = 1
    Filter = 2
    Rle = 3
    Dpp = 4
    Argmax = 5

    @classmethod
    def _from_inner(cls, value: "_hbrt4_py.variable.SpecialOperatorPy"):
        BindType = _hbrt4_py.variable.SpecialOperatorPy
        if value == BindType.Normal:
            return cls.Normal
        if value == BindType.Filter:
            return cls.Filter
        if value == BindType.Rle:
            return cls.Rle
        if value == BindType.Dpp:
            return cls.Dpp
        if value == BindType.Argmax:
            return cls.Argmax


class Version:
    def __init__(self, __handle: "_hbrt4_py.version.VersionPy"):
        self.__handle = __handle

    def _inner(self):
        return self.__handle

    @property
    def major(self) -> int:
        return self.__handle.major()

    @property
    def minor(self) -> int:
        return self.__handle.minor()

    @property
    def patch(self) -> int:
        return self.__handle.patch()

    @property
    def extra(self) -> str:
        return self.__handle.extra()

    @property
    def pre_release(self) -> str:
        return self.__handle.pre_release()

    @property
    def commit_hash(self) -> str:
        return self.__handle.commit_hash()

    def __str__(self) -> str:
        return self.__handle.str()

    def __repr__(self):
        return self.__str__()

    def __le__(self, other):
        return self.__handle.compare(other._inner()) <= 0

    def __lt__(self, other):
        return self.__handle.compare(other._inner()) < 0

    def __eq__(self, other):
        return self.__handle.compare(other._inner()) == 0


class Type:
    def __init__(self, __handle: "_hbrt4_py.types.Type"):
        self.__handle = __handle

    @property
    def dims(self) -> Optional[Tuple[Optional[int]]]:
        """None in dims is used for dynamic dims"""
        dims = self.__handle.dims()
        if dims is not None:
            return tuple(dims)

    @property
    def _strides(self) -> Optional[Tuple[Optional[int]]]:
        """None in strides is used for dynamic strides"""
        strides = self.__handle.strides()
        if strides is not None:
            return tuple(strides)

    @property
    def shape(self) -> Optional[Tuple[Optional[int]]]:
        """None in shape is used for dynamic shape"""
        return self.dims

    @property
    def element_type(self) -> Optional["Type"]:
        type = self.__handle.elem_type()
        if type is not None:
            return Type(type)

    @property
    def np_dtype(self) -> Optional["np.dtype"]:
        elem_type = self.element_type
        if elem_type is not None:
            dtype = elem_type.__handle.as_numpy_dtype()  # type: str
            if dtype is not None:
                return np.dtype(dtype)


class DescriptionCategory(Enum):
    String = 1
    Binary = 2

    @classmethod
    def _from_inner(cls, value: "_hbrt4_py.description.DescriptionPy"):
        BindType = _hbrt4_py.description.DescriptionCategoryPy
        if value == BindType.String:
            return cls.String
        if value == BindType.Binary:
            return cls.Binary
        raise ValueError(f"Bad value {value}")


class Description:
    def __init__(self, __handle: "_hbrt4_py.description.DescriptionPy"):
        self.__handle = __handle

    @property
    def category(self) -> DescriptionCategory:
        return DescriptionCategory._from_inner(self.__handle.category())

    @property
    def string_data(self) -> Optional[str]:
        return self.__handle.string_data()

    @property
    def binary_data(self) -> Optional[bytes]:
        inner_data = self.__handle.binary_data()
        if inner_data is None:
            return None
        return bytes(inner_data)

    @property
    def data(self) -> Optional[Union[str, bytes]]:
        if self.string_data is not None:
            return self.string_data
        if self.binary_data is not None:
            return self.binary_data


class MemspaceUsage(Enum):
    GraphInput = 1
    GraphOutput = 2
    Constant = 3
    Temporary = 4
    Intermediate = 5
    NodeCache = 6

    @classmethod
    def _from_inner(cls, value: "_hbrt4_py.memspace.MemspaceUsagePy"):
        BindType = _hbrt4_py.memspace.MemspaceUsagePy
        if value == BindType.GraphInput:
            return cls.GraphInput
        if value == BindType.GraphOutput:
            return cls.GraphOutput
        if value == BindType.Constant:
            return cls.Constant
        if value == BindType.Intermediate:
            return cls.Intermediate
        if value == BindType.Temporary:
            return cls.Temporary
        if value == BindType.NodeCache:
            return cls.NodeCache
        raise ValueError(f"Bad value {value}")


class Memspace:
    def __init__(self, __handle: "_hbrt4_py.memspace.MemspacePy"):
        self.__handle = __handle

    @property
    def name(self) -> str:
        return self.__handle.name()

    @property
    def usage(self) -> MemspaceUsage:
        return MemspaceUsage._from_inner(self.__handle.usage())

    @property
    def alignment(self) -> int:
        return self.__handle.alignment()

    @property
    def size(self) -> Optional[int]:
        return self.__handle.size()


class Variable:
    def __init__(self, __handle: "_hbrt4_py.variable.VariablePy", __hbm: "Hbm"):
        self.__handle = __handle
        self.__hbm_parent = __hbm
        self.unique_id = self.__handle.unique_id()

    def _inner(self):
        return self.__handle

    def __hash__(self):
        return self.unique_id

    def __eq__(self, other):
        if hasattr(other, "unique_id"):
            return self.unique_id == other.unique_id
        return False

    @property
    def name(self) -> str:
        return self.__handle.name()

    @property
    def staged_name(self) -> Union[None, str]:
        return self.__hbm_parent._get_staged_name_by_bind_obj(self)

    @staged_name.setter
    def staged_name(self, value: str):
        """
        This method does not really 'set' variable name. It does not in-place modify
        the underlying binary encoded in HBM or HBO. The value will be 'cached' and write
        to the HBM or HBO until 'save' method is called.
        """
        self.__hbm_parent._set_staged_name_by_bind_obj(self, value)

    @property
    def staged_desc(self) -> Union[None, str, bytes]:
        return self.__hbm_parent._get_staged_desc_by_bind_obj(self)

    @staged_desc.setter
    def staged_desc(self, value: Union[str, bytes]):
        """
        This method does not really 'set' variable desc. It does not in-place modify
        the underlying binary encoded in HBM or HBO. The value will be 'cached' and write
        to the HBM or HBO until 'save' method is called.
        """
        self.__hbm_parent._set_staged_desc_by_bind_obj(self, value)

    @property
    def desc(self) -> Optional[Description]:
        description = self.__handle.description()
        if description is not None:
            return Description(description)

    @property
    def type(self) -> Type:
        return Type(self.__handle.types())

    @property
    def children(self) -> List["Variable"]:
        return [Variable(x, self.__hbm_parent) for x in self.__handle.children()]

    @property
    def input_semantic(self) -> VariableInputSemantic:
        return VariableInputSemantic._from_inner(self.__handle.input_semantic())

    @property
    def defining_special_operator(self) -> Optional[SpecialOperator]:
        if self.__handle.defining_special_operator() is None:
            return None
        return SpecialOperator._from_inner(self.__handle.defining_special_operator())

    @property
    def memspace(self) -> Optional[Memspace]:
        m = self.__handle.memspace()
        if m is None:
            return None
        return Memspace(m)

    @property
    def offset_in_memspace(self) -> int:
        return self.__handle.offset_in_memspace()

    @property
    def is_constant(self) -> bool:
        return self.__handle.is_constant()


class DeviceCategory(Enum):
    Bpu = 1
    Cpu = 2

    @classmethod
    def _from_inner(cls, value: "_hbrt4_py.node.DeviceCategoryPy"):
        BindType = _hbrt4_py.node.DeviceCategoryPy
        if value == BindType.Bpu:
            return cls.Bpu
        if value == BindType.Cpu:
            return cls.Cpu
        raise ValueError(f"Bad value {value}")


class VariableNodeUsage(Enum):
    Input = 1
    Output = 2
    Constant = 3
    Temporary = 4
    NodeCache = 5

    @classmethod
    def _from_inner(cls, value: "_hbrt4_py.node.VariableNodeUsagePy"):
        BindType = _hbrt4_py.node.VariableNodeUsagePy
        if value == BindType.Input:
            return cls.Input
        if value == BindType.Output:
            return cls.Output
        if value == BindType.Constant:
            return cls.Constant
        if value == BindType.Temporary:
            return cls.Temporary
        if value == BindType.NodeCache:
            return cls.NodeCache
        raise ValueError(f"Bad value {value}")


class Node:
    def __init__(self, __handle: "_hbrt4_py.node.NodePy", __hbm: "Hbm"):
        self.__handle = __handle
        self.__hbm_parent = __hbm

    @property
    def name(self) -> str:
        return self.__handle.name()

    @property
    def device(self) -> DeviceCategory:
        return DeviceCategory._from_inner(self.__handle.device_category())

    @property
    def estimated_latency_micros(self) -> Optional[int]:
        return self.__handle.estimate_latency_micros()

    @property
    def inputs(self) -> List[Variable]:
        return [Variable(x, self.__hbm_parent) for x in self.__handle.inputs()]

    @property
    def outputs(self) -> List[Variable]:
        return [Variable(x, self.__hbm_parent) for x in self.__handle.outputs()]

    @property
    def memspaces(self) -> List[Memspace]:
        return [Memspace(x) for x in self.__handle.memspaces()]

    @property
    def ancestors(self) -> List["Node"]:
        return [Node(x, self.__hbm_parent) for x in self.__handle.ancestors()]

    @property
    def parameters(self) -> List[Variable]:
        return [Variable(x, self.__hbm_parent) for x in self.__handle.paramters()]

    @property
    def variables(self) -> List[Variable]:
        return [Variable(x, self.__hbm_parent) for x in self.__handle.variables()]

    @property
    def variable_usages(self) -> List[VariableNodeUsage]:
        return [
            VariableNodeUsage._from_inner(self.__handle.variables_usage(x._inner()))
            for x in self.variables
        ]


def _replace_special_characters(orig_str: str) -> str:
    return orig_str.replace("/", "_")


class Graph(TreeLikeFuncBase):
    def __init__(self, __handle: "_hbrt4_py.graph.GraphPy", __hbm: "Hbm"):
        self.__handle = __handle
        self.__hbm_parent = __hbm
        self.__flatten_inputs = None
        self.__flatten_outputs = None
        self.unique_id = self.__handle.unique_id()

    def __hash__(self):
        return self.unique_id

    def __eq__(self, other):
        if hasattr(other, "unique_id"):
            return self.unique_id == other.unique_id
        return False

    @property
    def nodes(self) -> List[Node]:
        """return all Nodes in graph

        Returns:
            List[Node]
        """
        return [Node(x, self.__hbm_parent) for x in self.__handle.nodes()]

    def __getitem__(self, index_or_name):
        if isinstance(index_or_name, int):
            if index_or_name < 0 or index_or_name >= len(self.nodes):
                raise IndexError(f"Index {index_or_name} is out of range")
            else:
                return self.nodes[index_or_name]
        elif isinstance(index_or_name, str):
            for node in self.nodes:
                if index_or_name == node.name:
                    return node
            raise ValueError(f'graph has no node "{index_or_name}"')
        else:
            raise TypeError(f"{index_or_name} has wrong type")

    @property
    def name(self) -> str:
        return self.__handle.name()

    @property
    def staged_name(self) -> Union[None, str]:
        return self.__hbm_parent._get_staged_name_by_bind_obj(self)

    @staged_name.setter
    def staged_name(self, value: str):
        """
        This method does not really 'set' graph name. It does not in-place modify
        the underlying binary encoded in HBM or HBO. The value will be 'cached' and write
        to the HBM or HBO until 'save' method is called.
        """
        self.__hbm_parent._set_staged_name_by_bind_obj(self, value)

    @property
    def staged_desc(self) -> Union[None, str, bytes]:
        return self.__hbm_parent._get_staged_desc_by_bind_obj(self)

    @staged_desc.setter
    def staged_desc(self, value: Union[str, bytes]):
        """
        This method does not really 'set' graph desc. It does not in-place modify
        the underlying binary encoded in HBM or HBO. The value will be 'cached' and write
        to the HBM or HBO until 'save' method is called.
        """
        self.__hbm_parent._set_staged_desc_by_bind_obj(self, value)

    @property
    def desc(self) -> Optional[Description]:
        description = self.__handle.description()
        if description is not None:
            return Description(description)

    @property
    def toolkit_version(self) -> Version:
        return Version(self.__handle.toolkit_version())

    @TreeLikeFuncBase._in_tree_spec.getter
    def _in_tree_spec(self):
        tree_dict = unpickle_object(self.__handle.tree_spec())
        if tree_dict is not None and "in" in tree_dict.keys():
            return unpickle_object(tree_dict["in"])
        else:
            return None

    @TreeLikeFuncBase._out_tree_spec.getter
    def _out_tree_spec(self):
        tree_dict = unpickle_object(self.__handle.tree_spec())
        if tree_dict is not None and "out" in tree_dict.keys():
            return unpickle_object(tree_dict["out"])
        else:
            return None

    @property
    def internal_desc(self) -> Optional[Description]:
        internal_desc = self.__handle.internal_description()
        if internal_desc is not None:
            return Description(internal_desc)

    @TreeLikeFuncBase.flatten_inputs.getter
    def flatten_inputs(self) -> List[Variable]:
        if self.__flatten_inputs is None:
            self.__flatten_inputs = [
                Variable(x, self.__hbm_parent) for x in self.__handle.inputs()
            ]
        return self.__flatten_inputs

    @TreeLikeFuncBase.flatten_outputs.getter
    def flatten_outputs(self) -> List[Variable]:
        if self.__flatten_outputs is None:
            self.__flatten_outputs = [
                Variable(x, self.__hbm_parent) for x in self.__handle.outputs()
            ]
        return self.__flatten_outputs

    # @property
    # def nodes(self) -> List[Node]:
    #     return [Node(x) for x in self.__handle.nodes()]

    def _get_inputs_outputs_from_snapshot(self, snapshot: str):
        res = {}
        res["inputs"] = {}
        input_bin_index = 1
        for input in self.flatten_inputs:
            if input.children:
                input_children = input.children
                pym_inputs = []
                for v in input_children:
                    file_name = os.path.join(
                        os.path.dirname(snapshot),
                        "input_" + str(input_bin_index) + ".bin",
                    )
                    array = np.fromfile(file_name, dtype=v.type.np_dtype)
                    strides = v.type._strides
                    shape = v.type.dims
                    stride_h = array.size // np.prod(shape[:-2])
                    yuv_strides = strides[:-3] + (stride_h,) + strides[-2:]
                    yuv_inputs = np.lib.stride_tricks.as_strided(
                        array, shape=shape, strides=yuv_strides
                    )
                    res["inputs"][v.name] = yuv_inputs
                    input_bin_index += 1
            else:
                file_name = os.path.join(
                    os.path.dirname(snapshot), "input_" + str(input_bin_index) + ".bin"
                )
                input_bin_index += 1
                res["inputs"][input.name] = np.lib.stride_tricks.as_strided(
                    np.fromfile(file_name, dtype=input.type.np_dtype),
                    shape=input.type.dims,
                    strides=input.type._strides,
                )

        with open(snapshot) as f:
            snapshot_dict = json.load(f)
            outputs_info = []
            for op in snapshot_dict["desc"]["opToRun"]["ord"]["ops"]:
                for cmp_key in op:
                    if cmp_key == "cmp":
                        if op[cmp_key]["rhs"]["addr"]["op"]["file"] == "output.bin":
                            info = {}
                            info["offset"] = int(
                                op[cmp_key]["rhs"]["addr"]["offset"]["i64"]
                            )
                            info["size_in_bytes"] = int(op[cmp_key]["size"]["i64"])
                            outputs_info.append(info)

        res["outputs"] = {}
        output_bin_name = os.path.join(os.path.dirname(snapshot), "output.bin")
        with open(output_bin_name, "rb") as f:
            raw_data = f.read()
        for index, output in enumerate(self.flatten_outputs):
            dtype = output.type.np_dtype
            assert dtype
            dims = output.type.dims
            strides = output.type._strides
            assert len(dims) == len(strides)  # HBDK-286: can be both empty for rank 0
            dsize = np.zeros(shape=1, dtype=dtype).dtype.itemsize
            count = outputs_info[index]["size_in_bytes"] // dsize
            assert count > 0
            offset = outputs_info[index]["offset"]
            array = np.frombuffer(raw_data, dtype=dtype, count=count, offset=offset)
            native_array = np.lib.stride_tricks.as_strided(
                array, shape=dims, strides=strides
            )
            res["outputs"][output.name] = native_array

        return res

    def _verify_result(
        self, snapshot: str, output_dir: str = None, debug_coord: Sequence[int] = None
    ):
        snapshot_inputs_outputs = self._get_inputs_outputs_from_snapshot(snapshot)
        hbm_outputs = self.feed(
            snapshot_inputs_outputs["inputs"], output_dir, debug_coord
        )
        error = ""
        for output in self.flatten_outputs:
            hbm_result = hbm_outputs[output.name]
            snapshot_result = snapshot_inputs_outputs["outputs"][output.name]
            if not np.array_equal(hbm_result, snapshot_result):
                error += f"{output.name} Check Failed!\n"
        if error:
            logging.error(error)
            return False
        else:
            return True

    def _gen_rt4_run_model_file(self, snapshot: str, output_dir: str = None):
        snapshot_inputs_outputs = self._get_inputs_outputs_from_snapshot(snapshot)
        tar_files = []
        config_dic = {}
        config_dic["inputs"] = []
        config_dic["outputs"] = []
        assert output_dir

        for input in self.flatten_inputs:
            input_info = {}
            if input.children:
                input_children = input.children
                input_dict = snapshot_inputs_outputs["inputs"]
                yuv_inputs = []
                max_w = max(v.type.shape[-2] for v in input_children)
                # b30g ddr 32 Bytes allign
                align_max_w = ((max_w + 32 - 1) // 32) * 32
                for v in input_children:
                    if v.name not in input_dict.keys():
                        raise ValueError("cannot find argument named {}".format(v.name))
                    data = input_dict[v.name]
                    if v.input_semantic == VariableInputSemantic.ImageUv:
                        uv_shape = data.shape[:-2] + (
                            data.shape[-2] * data.shape[-1],
                            1,
                        )
                        data = np.reshape(data, uv_shape)
                    yuv_inputs.append(data)
                combined_data = np.concatenate(yuv_inputs, axis=-3)
                combined_data.astype(input_children[0].type.np_dtype).tofile(
                    f"{output_dir}/{input.name}.bin"
                )
                # need relative path
                input_info["path"] = f"{input.name}.bin"
                input_info["input_type"] = "pyramid"
                input_info["stride"] = align_max_w
                config_dic["inputs"].append(input_info)
                tar_files.append(f"{output_dir}/{input.name}.bin")
            else:
                if input.name not in snapshot_inputs_outputs["inputs"].keys():
                    raise ValueError("cannot find argument named {}".format(input.name))
                data = snapshot_inputs_outputs["inputs"][input.name]
                data.astype(input.type.np_dtype).tofile(
                    f"{output_dir}/{input.name}.bin"
                )
                input_info["path"] = f"{input.name}.bin"
                input_info["input_type"] = "normal"
                config_dic["inputs"].append(input_info)
                tar_files.append(f"{output_dir}/{input.name}.bin")
        for output in self.flatten_outputs:
            output_info = {}
            if output.name not in snapshot_inputs_outputs["outputs"].keys():
                raise ValueError("cannot find output named {}".format(output.name))
            data = snapshot_inputs_outputs["outputs"][output.name]
            data.astype(output.type.np_dtype).tofile(f"{output_dir}/{output.name}.bin")
            output_info["path"] = f"{output.name}.bin"
            config_dic["outputs"].append(output_info)
            tar_files.append(f"{output_dir}/{output.name}.bin")
        shutil.copy(self.__hbm_parent.file_name, f"{output_dir}/{self.name}.hbm")
        tar_files.append(f"{output_dir}/{self.name}.hbm")
        run_model_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "runtime",
            "aarch64_unknown_linux_gnu",
            "nash",
            "bin",
            "hbrt4-run-model-nash",
        )
        libhbrt4_so_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "runtime",
            "aarch64_unknown_linux_gnu",
            "nash",
            "lib",
            "libhbrt4.so",
        )
        libhbtl_so_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "runtime",
            "aarch64_unknown_linux_gnu",
            "nash",
            "lib",
            "libhbtl.so",
        )
        for p in [run_model_path, libhbrt4_so_path, libhbtl_so_path]:
            # path = p
            if os.path.islink(p):
                p = os.path.realpath(p)
            tar_files.append(p)

        # gen shell scripts
        cmd_1 = "mkdir output\n"
        cmd_2 = ["./hbrt4-run-model-nash"]
        cmd_2 += ["-m"]
        if self.__hbm_parent.march == March.bayes:
            cmd_2 += ["Bayes2"]
        elif self.__hbm_parent.march == March.nash_b:
            cmd_2 += ["Bpu31"]
        elif (
            self.__hbm_parent.march == March.nash_e
            or self.__hbm_parent.march == March.nash_m
        ):
            cmd_2 += ["Bpu30g"]
        elif self.__hbm_parent.march == March.nash_p:
            cmd_2 += ["Bpu30p"]
        else:
            raise ValueError(f"Bad value {self.__hbm_parent.march}")
        cmd_2 += ["-f", f"{self.name}.hbm"]
        cmd_2 += ["-n", f"{self.name}"]

        for input in config_dic["inputs"]:
            cmd_2 += ["-i"]
            if input["input_type"] == "pyramid":
                cmd_2 += [input["path"] + ':{"stride":' + str(input["stride"]) + "}"]
            else:
                cmd_2 += [input["path"]]
        for output in config_dic["outputs"]:
            cmd_2 += ["-r", output["path"]]
        cmd_2 += ["-o", "output"]
        cmd_2 += ["--log-level", "Info"]
        cmd_2 = shlex.join(cmd_2)
        with open(f"{output_dir}/run.sh", "w") as file:
            file.write(cmd_1 + cmd_2)
        tar_files.append(f"{output_dir}/run.sh")
        with tarfile.open(f"{output_dir}/run_model.tar", "w") as tar:
            for file in tar_files:
                tar.add(file, arcname=os.path.basename(file))
        return True

    def _extract_variable_data_from_file(self, variable, file_name):
        dtype = variable.type.np_dtype
        assert dtype
        dims = variable.type.dims
        # Check whether existing dynamic shape
        is_dynamic_shape = any(item is None for item in dims)
        if is_dynamic_shape:
            file_size = variable.memspace.size
            dims = [-1 if item is None else item for item in dims]
        else:
            dsize = np.zeros(shape=1, dtype=dtype).dtype.itemsize
            file_size = functools.reduce(lambda x, y: x * y, dims, 1) * dsize
        assert file_size > 0
        return np.fromfile(file_name, dtype=dtype).reshape(dims)

    def _run_model_on_bpu_board(
        self,
        input_files,
        remote_ip: str = "",
        remote_port: int = 22,
        username: str = "root",
        password: str = "",
        local_work_path: str = "remote_bpu/",
        remote_work_root: str = "/tmp/",
    ):

        remote_obj = RemoteBPU(
            self.__hbm_parent.file_name,
            self.__hbm_parent.march,
            self.name,
            input_files,
            remote_ip,
            remote_port,
            local_work_path,
            remote_work_root,
            username,
            password,
        )

        ret_path = remote_obj.run_remote_bpu_board()
        ret_files = os.listdir(ret_path)
        res = []
        for output in self.flatten_outputs:
            tmp_file_name = output.name + ".bin"
            if tmp_file_name in ret_files:
                res.append(
                    self._extract_variable_data_from_file(
                        output, os.path.join(ret_path, tmp_file_name)
                    )
                )
        return res

    def __call__(
        self,
        *args,
        output_dir: str = None,
        debug_coord: Sequence[int] = None,
        remote_ip: str = None,
        remote_port: int = 22,
        username: str = "root",
        password: str = "",
        local_work_path: str = "remote_bpu/",
        remote_work_root: str = "/tmp/",
        **kwargs,
    ):
        if self.support_pytree:
            from torch.utils._pytree import tree_flatten, tree_unflatten

            input_list, input_spec = tree_flatten(args[0])
            if input_spec != self._in_tree_spec:
                raise ValueError(
                    "tree spec of function does not match given pytree input"
                )
            output_list = self._launch(
                *input_list,
                output_dir=output_dir,
                debug_coord=debug_coord,
                remote_ip=remote_ip,
                remote_port=remote_port,
                username=username,
                password=password,
                local_work_path=local_work_path,
                remote_work_root=remote_work_root,
                **kwargs,
            )
            return tree_unflatten(output_list, self._out_tree_spec)
        else:
            return self._launch(
                *args,
                output_dir=output_dir,
                debug_coord=debug_coord,
                remote_ip=remote_ip,
                remote_port=remote_port,
                username=username,
                password=password,
                local_work_path=local_work_path,
                remote_work_root=remote_work_root,
                **kwargs,
            )

    def feed(
        self,
        feed_dict: Dict[str, Any],
        output_dir: str = None,
        debug_coord: Sequence[int] = None,
        remote_ip: str = None,
        remote_port: int = 22,
        username: str = "root",
        password: str = "",
        local_work_path: str = "remote_bpu/",
        remote_work_root: str = "/tmp/",
    ) -> Dict[str, np.array]:
        feed_list = []
        for input in self.flatten_inputs:
            if input.children:  # compatible mode, pyramid and resizer input
                for v in input.children:
                    if v.name not in feed_dict.keys():
                        raise ValueError("cannot find argument named {}".format(v.name))
                    feed_list.append(feed_dict[v.name])
            else:
                if input.name not in feed_dict.keys():
                    raise ValueError("cannot find argument named {}".format(input.name))
                feed_list.append(feed_dict[input.name])
        output_list = self._launch(
            *feed_list,
            output_dir=output_dir,
            debug_coord=debug_coord,
            remote_ip=remote_ip,
            remote_port=remote_port,
            username=username,
            password=password,
            local_work_path=local_work_path,
            remote_work_root=remote_work_root,
        )
        return {v.name: array for v, array in zip(self.flatten_outputs, output_list)}

    def _launch(
        self,
        *args,
        output_dir: str = None,
        debug_coord: Sequence[int] = None,
        remote_ip: str = None,
        remote_port: int = 22,
        username: str = "root",
        password: str = "",
        local_work_path: str = "remote_bpu/",
        remote_work_root: str = "/tmp/",
        **kwargs,
    ) -> List[np.array]:
        if remote_ip is not None:
            with tempfile.TemporaryDirectory() as tmp_dir:
                input_files = []
                names = []
                for input in self.flatten_inputs:
                    if input.children:  # compatible mode, pyramid and resizer input
                        for v in input.children:
                            names.append(v.name)
                    else:
                        names.append(input.name)

                for name, data in zip(names, args):
                    if is_torch_tensor(data):
                        data = data.detach().numpy()
                    filename = os.path.join(tmp_dir, name + ".bin")
                    data.tofile(filename)
                    input_files.append(filename)
                return self._run_model_on_bpu_board(
                    input_files,
                    remote_ip,
                    remote_port,
                    username,
                    password,
                    local_work_path,
                    remote_work_root,
                )
        ddr_aligned_bytes = {MarchSeries.nash: 32, MarchSeries.bayes: 16}
        run_process = {
            MarchSeries.bayes: os.path.join(
                os.path.dirname(__file__), "_mlir_libs", "hbdk-sim"
            ),
            MarchSeries.nash: os.path.join(
                os.path.dirname(__file__), "_mlir_libs", "hbrt4-run-model-nash"
            ),
        }
        with tempfile.TemporaryDirectory() as dir:
            if output_dir:
                dir = output_dir
            inputs_file = []
            yuv_shape = []
            yuv_stride = []
            resizer_roi = []

            aligned_size = ddr_aligned_bytes[self.__hbm_parent.march.series]
            index = 0
            for input in self.flatten_inputs:
                if input.children:  # compatible mode, pyramid and resizer input
                    input_children = input.children
                    max_w = max(v.type.shape[-2] for v in input_children)
                    align_max_w = (
                        (max_w + aligned_size - 1) // aligned_size
                    ) * aligned_size
                    yuv_inputs = []
                    for v in input_children:
                        data = args[index]
                        if is_torch_tensor(data):
                            data = data.detach().numpy()
                        if v.input_semantic == VariableInputSemantic.ImageUv:
                            uv_shape = data.shape[:-2] + (
                                data.shape[-2] * data.shape[-1],
                                1,
                            )
                            data = np.reshape(data, uv_shape)
                        padding = align_max_w - data.shape[-2]
                        if (
                            padding > 0
                            and self.__hbm_parent.march.series == MarchSeries.bayes
                        ):
                            pad_width = [(0, 0)] * data.ndim
                            pad_width[-2] = (0, padding)
                            data = np.pad(
                                data, pad_width, mode="constant", constant_values=0
                            )
                        if v.input_semantic == VariableInputSemantic.ImageRoi:
                            roi = data[0].tolist()
                        else:
                            yuv_inputs.append(data)
                        index += 1

                    if input.input_semantic == VariableInputSemantic.Resizer:
                        shape = list(yuv_inputs[0].shape)
                        if len(yuv_inputs) == 2:  # nv12
                            shape[-1] = 3
                        align_stride = (
                            (shape[-2] + aligned_size - 1) // aligned_size
                        ) * aligned_size
                        yuv_stride.append(align_stride)
                        yuv_shape.append(shape)
                        resizer_roi.append(roi)
                    elif input.input_semantic == VariableInputSemantic.Pyramid:
                        yuv_stride.append(align_max_w)
                        yuv_shape.append(None)
                        resizer_roi.append(None)
                    # To combine y and uv inputs into the same input.bin
                    if len(yuv_inputs[0].shape) == 3:
                        axis = 0
                    else:
                        axis = -1
                    combined_data = np.concatenate(
                        list(
                            map(
                                lambda x: x.reshape((*x.shape[:-3], -1)),
                                yuv_inputs,
                            )
                        ),
                        axis,
                    )
                    combined_data.astype(input_children[0].type.np_dtype).tofile(
                        f"{dir}/{input.name}.bin"
                    )
                    inputs_file.append(f"{dir}/{input.name}.bin")
                else:
                    resizer_roi.append(None)

                    if (
                        input.input_semantic == VariableInputSemantic.ImageY
                        or input.input_semantic == VariableInputSemantic.ImageUv
                    ):
                        if (
                            input.type.shape[-2] is None
                            and input.type.shape[-3] is None
                        ):  # resizer input, provides shape by user
                            shape = args[index].shape
                            yuv_shape.append(list(shape))
                        else:  # pyramid input
                            shape = input.type.shape
                            yuv_shape.append(None)
                        stride = shape[-1] * shape[-2]
                        align_stride = (
                            (stride + aligned_size - 1) // aligned_size
                        ) * aligned_size
                        yuv_stride.append(align_stride)
                    else:
                        yuv_shape.append(None)
                        yuv_stride.append(None)

                    data = args[index]
                    if is_torch_tensor(data):
                        data = data.detach().numpy()
                    data.astype(input.type.np_dtype).tofile(f"{dir}/{input.name}.bin")
                    inputs_file.append(f"{dir}/{input.name}.bin")
                    index += 1

            output_dir = f"{dir}/output"
            os.makedirs(output_dir, exist_ok=True)

            # old file name may contain ",", however hbm input file need it as sep
            rename_inputs_file = []
            for old_file_name in inputs_file:
                new_file_name = old_file_name.replace(",", "_")
                os.rename(old_file_name, new_file_name)
                rename_inputs_file.append(new_file_name)
            cmd = [run_process[self.__hbm_parent.march.series]]
            cmd += ["-f", self.__hbm_parent.file_name]
            cmd += ["-n", self.name]
            cmd += ["-o", output_dir]
            env = {}
            if self.__hbm_parent.march.series == MarchSeries.bayes:
                cmd += ["-i", ",".join(rename_inputs_file)]
                if debug_coord:
                    assert len(debug_coord) == 4
                    cmd += ["--enable-logging"]
                    env["HBDK_DEBUG_COORD1"] = ",".join([str(d) for d in debug_coord])

                if any(element is not None for element in yuv_stride):
                    cmd += [
                        "--yuv-stride",
                        ",".join([str(d) if d else "" for d in yuv_stride]),
                    ]
            elif self.__hbm_parent.march.series == MarchSeries.nash:
                if self.__hbm_parent.march == March.nash_b:
                    cmd += ["-m", "Bpu31"]
                if (
                    self.__hbm_parent.march == March.nash_e
                    or self.__hbm_parent.march == March.nash_m
                ):
                    cmd += ["-m", "Bpu30g"]
                if self.__hbm_parent.march == March.nash_p:
                    cmd += ["-m", "Bpu30p"]
                for index, input_file in enumerate(rename_inputs_file):
                    if resizer_roi[index]:  # only for compatible mode
                        cmd += [
                            "-i",
                            input_file
                            + ':{"shape":'
                            + str(yuv_shape[index])
                            + ","
                            + '"stride":'
                            + str(yuv_stride[index])
                            + ","
                            + '"roi":'
                            + str(resizer_roi[index])
                            + "}",
                        ]
                    elif yuv_shape[index] and yuv_stride[index]:
                        cmd += [
                            "-i",
                            input_file
                            + ':{"shape":'
                            + str(yuv_shape[index])
                            + ","
                            + '"stride":'
                            + str(yuv_stride[index])
                            + "}",
                        ]
                    elif yuv_shape[index] is None and yuv_stride[index]:
                        cmd += [
                            "-i",
                            input_file + ':{"stride":' + str(yuv_stride[index]) + "}",
                        ]
                    else:
                        cmd += ["-i", input_file]
            else:
                ValueError(f"Bad value {self.__hbm_parent.march}")
            print(f"cmd{shlex.join(cmd)}")
            p = run_program_redirect_realtime(
                cmd, stdout=sys.stdout, stderr=sys.stderr, env=env
            )
            ret = p.returncode
            if ret != 0:
                raise RuntimeError("HBDK sim FAIL, please check with HBDK Group")
            res = []
            for output in self.flatten_outputs:
                file_name = (
                    f"{output_dir}/{_replace_special_characters(output.name)}.bin"
                )
                res.append(self._extract_variable_data_from_file(output, file_name))
            return res


class GraphGroupClassification(Enum):
    Batch = 1
    Single = 2

    @classmethod
    def _from_inner(cls, value: "_hbrt4_py.graph_group.GraphGroupClassificationPy"):
        BindType = _hbrt4_py.graph_group.GraphGroupClassificationPy
        if value == BindType.Batch:
            return cls.Batch
        if value == BindType.Single:
            return cls.Single
        raise ValueError(f"Bad value {value}")


class GraphGroup:
    def __init__(self, __handle: "_hbrt4_py.graph.GraphGroupPy", __hbm: "Hbm"):
        self.__handle = __handle
        self.__hbm_parent = __hbm

    @property
    def name(self) -> str:
        return self.__handle.name()

    @property
    def classification(self) -> GraphGroupClassification:
        return GraphGroupClassification._from_inner(self.__handle.classification())

    @property
    def graphs(self) -> List[Graph]:
        return [Graph(x, self.__hbm_parent) for x in self.__handle.graphs()]


class Hbo:
    def __init__(self, hbo_file_name: str):
        self.__hbo_file_name = hbo_file_name

    def get_name(self):
        return self.__hbo_file_name


def _get_desc_type(desc):
    if isinstance(desc, str):
        return "string"
    elif isinstance(desc, bytes):
        return "binary"
    raise TypeError("Unsupported desc type: {}".format(type(desc).__name__))


def _update_desc_by_staged_desc(desc, staged_desc):
    if staged_desc is None:
        return
    desc["desc"] = staged_desc
    desc["desc_type"] = _get_desc_type(staged_desc)


def _update_desc_by_staged_name(desc, staged_name):
    if staged_name is None:
        return
    desc["new_name"] = staged_name


def _init_if_absent(desc, key):
    if key not in desc:
        desc[key] = {}


class Hbm:
    def __init__(self, hbm_file_name: str):
        self.__handle = _hbrt4_py.hbm.HbmPy.create_by_filename(hbm_file_name)
        self.__hbrt4_disas = os.path.join(
            os.path.dirname(__file__), "_mlir_libs", "hbrt4-disas"
        )
        self.__object_to_staged_desc: dict[Any, Union[bytes, str]] = {}
        self.__object_to_staged_name: dict[Any, str] = {}
        self.file_name = hbm_file_name

    @property
    def desc(self) -> Optional[Description]:
        description = self.__handle.desc()
        if description is not None:
            return Description(description)

    def _get_staged_desc_by_bind_obj(self, object):
        if object not in self.__object_to_staged_desc:
            return None
        return self.__object_to_staged_desc[object]

    def _get_staged_name_by_bind_obj(self, object):
        if object not in self.__object_to_staged_name:
            return None
        return self.__object_to_staged_name[object]

    def _set_staged_desc_by_bind_obj(self, object, value):
        self.__object_to_staged_desc[object] = value

    def _set_staged_name_by_bind_obj(self, object, value):
        self.__object_to_staged_name[object] = value

    @property
    def staged_desc(self) -> Union[None, str, bytes]:
        return self._get_staged_desc_by_bind_obj(self)

    @staged_desc.setter
    def staged_desc(self, value: Union[str, bytes]):
        """
        This method does not really 'set' hbm desc. It does not in-place modify
        the underlying binary encoded in HBM. The value will be 'cached' and write
        to the HBM or HBO until 'save' method is called.
        """
        self._set_staged_desc_by_bind_obj(self, value)

    @property
    def graphs(self) -> List[Graph]:
        """return all graphs in module

        Returns:
            List[Graph]
        """
        return [Graph(x, self) for x in self.__handle.graphs()]

    @property
    def graph_groups(self) -> List[GraphGroup]:
        return [GraphGroup(x, self) for x in self.__handle.graph_groups()]

    @property
    def march(self) -> March:
        """return hbm march

        Returns:
            March
        """
        BpuMarchPy = _hbrt4_py.hbm.BpuMarchPy
        if self.__handle.march() == BpuMarchPy.Bayes2:
            return March.bayes
        if self.__handle.march() == BpuMarchPy.Bpu31:
            return March.nash_b
        if self.__handle.march() == BpuMarchPy.Bpu30g:
            return March.nash_e
        if self.__handle.march() == BpuMarchPy.Bpu30g2:
            return March.nash_m
        if self.__handle.march() == BpuMarchPy.Bpu30p:
            return March.nash_p
        ValueError(f"Bad value {self.__handle.march()}")

    @property
    def march_name(self) -> str:
        return self.__handle.march_name()

    @property
    def toolkit_version(self):
        return Version(self.__handle.toolkit_version())

    @property
    def functions(self) -> List[Graph]:
        return self.graphs

    def __getitem__(self, index_or_name):
        if isinstance(index_or_name, int):
            return self.graphs[index_or_name]
        elif isinstance(index_or_name, str):
            for graph in self.graphs:
                if index_or_name == graph.name:
                    return graph
            raise ValueError(f'hbm has no graph "{index_or_name}"')
        else:
            raise TypeError(f"{index_or_name} has wrong type")

    def visualize(self, host: Optional[str] = None, port: int = 0, *, _visual=True):
        """Visualize hbm using netron

        Args:
            host (str, optional): Hostname used to access the website. Defaults to be auto determined
            port (int, optional): Port used for visualization webpage. Defaults is to auto determine the port
            _visual (bool, optional): Used internally for hbdk4 testing
        """

        import socket
        import time
        import netron

        servers = []

        with tempfile.NamedTemporaryFile(suffix=".prototxt") as f:
            onnx_file = f.name
            subprocess.check_call(
                [self.__hbrt4_disas, self.file_name, "--netron", "-o", f.name]
            )
            print("\033[92mTemporary proto file saved to {}\033[0m".format(onnx_file))

            server = netron.server

            if host is None:
                host = socket.gethostname()
            host, port = server.serve(onnx_file, None, address=(host, port))

            print("\033[92mVisit http://{}:{}\033[0m".format(host, port))

            servers.append(server)

            print('\033[92mEnter "c" to shutdown all servers and continue...\033[0m')

            if _visual is False:
                for server in servers:
                    server.stop()
                print("\033[92mStopping server...\033[0m")
                return

            while True:
                time.sleep(2)
                key = input()
                if key.strip() == "c":
                    for server in servers:
                        server.stop()
                    print("\033[92mStopping server...\033[0m")
                    break

    def save_by_staged_info(self, filename: str):
        desc_dict = hbm_extract_desc(self.file_name)
        _update_desc_by_staged_desc(desc_dict, self.staged_desc)
        graph_name_set = set([g.name for g in self.graphs])
        for g in self.graphs:
            if g.name not in desc_dict["models"]:
                raise RuntimeError(
                    "model {} does not appear in desc dict".format(g.name)
                )
            _update_desc_by_staged_desc(desc_dict["models"][g.name], g.staged_desc)
            if g.staged_name != g.name and g.staged_name in graph_name_set:
                raise ValueError(
                    'Graph name "{}" already in the hbm. Duplicate name is not allowed.'.format(
                        g.staged_name
                    )
                )
            _update_desc_by_staged_name(desc_dict["models"][g.name], g.staged_name)
            input_names = set([v.name for v in g.inputs])
            for v in g.inputs:
                _init_if_absent(desc_dict["models"][g.name]["input_features"], v.name)
                _update_desc_by_staged_desc(
                    desc_dict["models"][g.name]["input_features"][v.name], v.staged_desc
                )
                if v.staged_name != v.name and v.staged_name in input_names:
                    raise ValueError(
                        'Input variable name "{}" already exists. Duplicate name is not allowed.'.format(
                            v.staged_name
                        )
                    )
                _update_desc_by_staged_name(
                    desc_dict["models"][g.name]["input_features"][v.name], v.staged_name
                )
            output_names = set([v.name for v in g.outputs])
            for v in g.outputs:
                _init_if_absent(desc_dict["models"][g.name]["output_features"], v.name)
                _update_desc_by_staged_desc(
                    desc_dict["models"][g.name]["output_features"][v.name],
                    v.staged_desc,
                )
                if v.staged_name != v.name and v.staged_name in output_names:
                    raise ValueError(
                        'Output variable name "{}" already exists. Duplicate name is not allowed.'.format(
                            v.staged_name
                        )
                    )
                _update_desc_by_staged_name(
                    desc_dict["models"][g.name]["output_features"][v.name],
                    v.staged_name,
                )
        shutil.copy(self.file_name, filename)
        hbm_update_desc(filename, desc_dict)
