import itertools
import re
from typing import (
    TYPE_CHECKING,
    Any,
    Container,
    Dict,
    Iterable,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Set,
    Tuple,
    Union,
)

import numpy as np

from hmct.ir import DataType

if TYPE_CHECKING:
    from enum import Enum

    from hmct.ir import OnnxNode, OnnxVariable


def _check_equal(
    actual: Union[float, str, Iterable[float], Iterable[str]],
    desired: Union[float, str, Iterable[float], Iterable[str]],
    rtol: float,
    atol: float,
) -> bool:
    """Check if the actual value matches the desired value.

    Args:
        actual: The actual value to compare.
        desired: The desired value to compare.
        rtol: The relative tolerance parameter for the match.
            Only used for float comparison.
        atol: The absolute tolerance parameter for the match.
            Only used for float comparison.

    Returns:
        True if the actual value matches the desired value, False otherwise.

    Examples:
        >>> _check_equal(1, 1.0, rtol=1e-07, atol=0.0)
        True
        >>> _check_equal(1.2, 1.0, rtol=1e-07, atol=0.0)
        False
        >>> _check_equal([1, 2, 3], [1.0, 2.0, 3.0], rtol=1e-07, atol=0.0)
        True
        >>> _check_equal([1, 1.0, 1.0], 1.0, rtol=1e-07, atol=0.0)
        False
        >>> _check_equal([1, 2, 3], [1, 2, 3])
        True
        >>> _check_equal("hello", "hello")
        True
    """
    # For string comparison, we directly compare the actual and desired values.
    if isinstance(actual, str) or isinstance(desired, str):
        return actual == desired

    if not isinstance(actual, Iterable):
        actual = [actual]
    if isinstance(desired, Iterable):
        assert _len_iterable(desired) == _len_iterable(
            actual
        ), "The length of actual and desired must match."
    else:
        desired = [desired] * _len_iterable(actual)

    # Compare the actual and desired values element-wise.
    for a, d in zip(actual, desired):
        # For float comparison, we use the relative and absolute tolerance parameters.
        if isinstance(a, float) or isinstance(d, float):
            if not np.isclose(a, d, rtol=rtol, atol=atol):
                return False
        # For integer or string comparison, we directly compare the
        # actual and desired values.
        else:
            if a != d:
                return False

    return True


def _len_iterable(iterable: Iterable) -> int:
    """Return the length of the iterable object.

    Args:
        iterable: The iterable object to obtain the length.

    Returns:
        The length of the iterable object.

    Examples:
        >>> _len_iterable([1, 2, 3])
        3
        >>> _len_iterable(range(10))
        10
        >>> _len_iterable("hello")
        5
    """
    if hasattr(iterable, "__len__"):
        return len(iterable)
    return sum(1 for _ in iterable)


def _split_expression(expression: str, delimiter: str) -> List[str]:
    """Split the expression into a list of sub-expressions.

    The expression is split by the delimiter, but only if the delimiter is not
    within any brackets. The brackets are defined as "<", ">", "[", "]", "(", ")".

    Args:
        expression: The input expression to split.
        delimiter: The delimiter to split the expression.

    Returns:
        A list of sub-expressions.

    Examples:
        >>> _split_expression("a, b, c", ",")
        ["a", " b", " c"]
        >>> _split_expression("a, b<x=y, i=j>, c", ",")
        ["a", " b<x=y, i=j>", " c"]
        >>> _split_expression("a<x=[1, 2, 3]>, b, c", ",")
        ["a<x=[1, 2, 3]>", " b", " c"]
    """
    brackets_record = {"<": 0, "[": 0, "(": 0}
    brackets_mapping = {">": "<", "]": "[", ")": "("}
    for idx, ch in enumerate(expression):
        if ch in brackets_record:
            brackets_record[ch] += 1
        if ch in brackets_mapping:
            brackets_record[brackets_mapping[ch]] -= 1
        # split the expression if the delimiter is found and all brackets are closed
        if ch == delimiter and all(cnt == 0 for cnt in brackets_record.values()):
            # split the expression recursively by the delimiter
            return [
                expression[:idx],
                *_split_expression(expression[idx + 1 :], delimiter=delimiter),
            ]
    # return the whole expression if no delimiter is found
    return [expression]


class _DestInfoTuple(NamedTuple):
    """A named tuple to store the destination information of a variable pattern.

    Attributes:
        dest_op: The destination node pattern of the variable pattern.
        dest_idx: The destination input index of the variable pattern.

    Examples:
        >>> _DestInfoTuple(dest_op=node_pattern, dest_idx=0)
    """

    # dest_op field
    dest_op: "_NodePattern"
    # dest_idx field
    dest_idx: int


class _VariablePattern:
    """A class to represent a variable pattern object.

    Attributes:
        onnx_var: The onnx variable matched with the variable pattern.
        owning_graph: The owning graph pattern object.
        src_op: The source node pattern of the variable pattern.
        name: The name of the variable pattern.
        attributes: The dictionary of variable attributes.
        dest_infos: The list of destination information of the variable pattern.
    """

    def __init__(
        self,
        owning_graph: "_GraphPattern",
        src_op: Optional["_NodePattern"],
        name: str,
        attributes: Optional[str],
    ) -> None:
        """Initialize a variable pattern object.

        Args:
            owning_graph: The owning graph pattern object.
            src_op: The source node pattern of the variable pattern.
            name: The name of the variable pattern.
            attributes: The attribute expression of the variable pattern.

        Examples:
            >>> owning_graph = _GraphPattern(
            ...     '''
            ...     y0 = Sigmoid(x<shape=[1, 3, 224, 224], dtype=FLOAT32>)
            ...     y1 = Mul<commutative=True>(x, y0)
            ...     '''
            ... )
            >>> _VariablePattern(
            ...     owning_graph=owning_graph,
            ...     src_op=None,
            ...     name="x",
            ...     attributes="shape=[1, 3, 224, 224], dtype=FLOAT32"
            ... )
        """
        # record the onnx variable matched with the variable pattern
        self.onnx_var: OnnxVariable

        self.owning_graph = owning_graph
        self.owning_graph.variables[name] = self
        self.src_op = src_op
        self.name = name
        self.attributes: Dict[str, Any] = {}
        self.dest_infos: List[_DestInfoTuple] = []

        # parse variable attributes expressions
        if attributes is not None:
            for attr_expr in _split_expression(attributes, delimiter=","):
                attr_name, value_expr = attr_expr.split("=")
                attr_name, value_expr = attr_name.strip(), value_expr.strip()
                assert attr_name in [
                    "shape",
                    "dtype",
                    "value",
                    "rank",
                    "numel",
                    "is_param",
                ], f"Unsupported variable attribute: {attr_name}"
                if attr_name == "dtype":
                    self.attributes[attr_name] = self._eval_dtype(value_expr)
                else:
                    self.attributes[attr_name] = eval(value_expr)

    def _eval_dtype(self, dtype_expr: str) -> Union[DataType, List[DataType]]:
        """Evaluate the dtype expression.

        Args:
            dtype_expr: The dtype expression to evaluate.

        Returns:
            The evaluated data type or a list of data types.
        """
        if dtype_expr.startswith("[") and dtype_expr.endswith("]"):
            return [
                getattr(DataType, dtype.strip())
                for dtype in dtype_expr[1:-1].split(",")
            ]
        return getattr(DataType, dtype_expr)

    def match(
        self,
        onnx_var: "OnnxVariable",
        rtol: float,
        atol: float,
        memo: Optional[Set[Union["_NodePattern", "_VariablePattern"]]] = None,
    ) -> bool:
        """Match the onnx variable with the variable pattern.

        Args:
            onnx_var: The onnx variable to match.
            rtol: The relative tolerance parameter for the match.
                Only used for float comparison.
            atol: The absolute tolerance parameter for the match.
                Only used for float comparison.
            memo: The set to store the matched patterns.

        Returns:
            True if the onnx variable matches the variable pattern, False otherwise.
        """
        # If the variable pattern is already matched with some onnx variable,
        # we only check if the given onnx variable is the same as the matched one.
        if hasattr(self, "onnx_var"):
            return self.onnx_var is onnx_var

        # check if the variable attributes match
        for attr_name, attr_val in self.attributes.items():
            if attr_name == "shape":  # noqa: SIM102
                if onnx_var.shape is None or list(attr_val) != list(onnx_var.shape):
                    return False
            if attr_name == "dtype":  # noqa: SIM102
                if (
                    isinstance(attr_val, Container) and onnx_var.dtype not in attr_val
                ) or (
                    not isinstance(attr_val, Container) and onnx_var.dtype != attr_val
                ):
                    return False
            if attr_name == "value":  # noqa: SIM102
                if onnx_var.value is None or not _check_equal(
                    onnx_var.value.flatten(), attr_val, rtol=rtol, atol=atol
                ):
                    return False
            if attr_name == "rank":  # noqa: SIM102
                if onnx_var.shape is None or attr_val != len(onnx_var.shape):
                    return False
            if attr_name == "numel":  # noqa: SIM102
                if onnx_var.is_shape_dynamic or attr_val != np.prod(onnx_var.shape):
                    return False
            if attr_name == "is_param":  # noqa: SIM102
                if attr_val != onnx_var.is_param:
                    return False

        # For non-leaf variables, we also check if the destination information matches.
        # The condition for non-leaf variables:
        #     1. The source node is not None.
        #     2. The destination information is not empty.
        if (
            self.src_op is not None
            and self.dest_infos
            and (
                len(self.dest_infos) != len(onnx_var.dest_infos)
                or onnx_var in onnx_var.owning_graph.outputs
            )
        ):
            return False

        # check recursively if the source node matches
        if self.src_op is not None:
            if onnx_var.src_op is None:
                return False
            if not self.src_op.match(onnx_var.src_op, rtol=rtol, atol=atol, memo=memo):
                return False

        # record the matched onnx variable
        self.onnx_var = onnx_var
        # update the memo set
        if memo is not None:
            memo.add(self)
        return True

    def reset_binding(self) -> None:
        """Reset the bound onnx variable."""
        if hasattr(self, "onnx_var"):
            del self.onnx_var


class _NodePattern:
    """A class to represent a node pattern object.

    Attributes:
        onnx_node: The onnx node matched with the node pattern.
        owning_graph: The owning graph pattern object.
        op_type: The op_type of the node pattern.
        inputs: The list of input variable patterns.
        outputs: The list of output variable patterns.
        attributes: The dictionary of node attributes.
    """

    def __init__(
        self,
        owning_graph: "_GraphPattern",
        op_type: str,
        inputs: str,
        outputs: str,
        attributes: Optional[str],
    ) -> None:
        """Initialize a node pattern object.

        Args:
            owning_graph: The owning graph pattern object.
            op_type: Th op_type of the node pattern.
            inputs: The input expression of the node pattern.
            outputs: The output expression of the node pattern.
            attributes: The attribute expression of the node pattern.

        Examples:
            >>> owning_graph = _GraphPattern(
            ...     '''
            ...     y0 = Sigmoid(x)
            ...     y1 = Mul<commutative=True>(x, y0)
            ...     '''
            ... )
            >>> _NodePattern(
            ...     owning_graph=owning_graph,
            ...     op_type="Mul",
            ...     inputs="x, y0",
            ...     outputs="y1",
            ...     attributes="commutative=True",
            ... )
        """
        # record the onnx node matched with the node pattern
        self.onnx_node: OnnxNode

        self.owning_graph = owning_graph
        self.owning_graph.nodes.append(self)
        self.op_type = op_type
        self.inputs: List[_VariablePattern] = []
        self.outputs: List[_VariablePattern] = []
        self.attributes: Dict[str, Any] = {}

        # parse node inputs expressions
        for input_expr in _split_expression(inputs, delimiter=","):
            # parse logic if the input is a constant
            if re.match(
                r"(^ *[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)? *$)"
                r"|(^ *[+-]?(\d+)/(\d+) *$)",
                input_expr,
            ):
                # obtain a unique name for the constant variable
                name_suffix = len(self.owning_graph.variables)
                while True:
                    unique_name = f"variable_{name_suffix}"
                    if unique_name not in self.owning_graph.variables:
                        break
                    name_suffix += 1
                var_name = unique_name
                var_attrs = f"is_param=True, value={input_expr}"
            # parse logic if the input is a variable
            else:
                match = re.match(
                    r"^ *(?P<var_name>[^< ]*)( *|(<(?P<var_attrs>.*?)>)? *)$",
                    input_expr,
                )
                var_name = match.group("var_name")
                var_attrs = match.group("var_attrs")
            # check if the variable is already defined
            if var_name in self.owning_graph.variables:
                # if the variable is already defined, use it
                input_var = self.owning_graph.variables[var_name]
            else:
                # if the variable is not defined, create a new one
                input_var = _VariablePattern(
                    owning_graph=self.owning_graph,
                    src_op=None,
                    name=var_name,
                    attributes=var_attrs,
                )
            # add the variable to the node inputs
            self.inputs.append(input_var)
            input_var.dest_infos.append(
                _DestInfoTuple(dest_op=self, dest_idx=len(self.inputs) - 1)
            )

        # parse node outputs expressions
        for output_expr in _split_expression(outputs, delimiter=","):
            match = re.match(
                r"^ *(?P<var_name>[^< ]*)( *|(<(?P<var_attrs>.*?)>)? *)$", output_expr
            )
            var_name = match.group("var_name")
            var_attrs = match.group("var_attrs")
            # create the new variable pattern and append it to the node outputs
            self.outputs.append(
                _VariablePattern(
                    owning_graph=self.owning_graph,
                    src_op=self,
                    name=var_name,
                    attributes=var_attrs,
                )
            )

        # parse node attributes expressions
        if attributes is not None:
            for attr_expr in _split_expression(attributes, delimiter=","):
                name_expr, value_expr = attr_expr.split("=")
                attr_name, attr_val = name_expr.strip(), eval(value_expr)
                self.attributes[attr_name] = attr_val

    def match(
        self,
        onnx_node: "OnnxNode",
        rtol: float,
        atol: float,
        memo: Optional[Set[Union["_NodePattern", "_VariablePattern"]]] = None,
    ) -> bool:
        """Match the onnx node with the node pattern.

        Args:
            onnx_node: The onnx node to match.
            rtol: The relative tolerance parameter for the match.
                Only used for float comparison.
            atol: The absolute tolerance parameter for the match.
                Only used for float comparison.
            memo: The set to store the matched patterns.

        Returns:
            True if the onnx node matches the node pattern, False otherwise.
        """
        # If the node pattern is already matched with some onnx node,
        # we only check if the given onnx node is the same as the matched one.
        if hasattr(self, "onnx_node"):
            return self.onnx_node is onnx_node

        # check if the node type and attributes match
        if self.op_type != onnx_node.op_type:
            return False
        for attr_name, attr_val in self.attributes.items():
            if attr_name != "commutative" and not _check_equal(
                onnx_node.attributes[attr_name], attr_val, rtol=rtol, atol=atol
            ):
                return False

        # check if the node inputs match recursively
        inputs_permutations: Iterable[Iterable[OnnxVariable]]
        if self.attributes.get("commutative", False):
            inputs_permutations = itertools.permutations(onnx_node.inputs)
        else:
            inputs_permutations = [onnx_node.inputs]
        # If onnx node is commutative, any permutation of node inputs matches
        # means the node matches.
        for inputs in inputs_permutations:
            inputs_matched = True
            incr_memo: Set[Union[_NodePattern, _VariablePattern]] = set()
            # If any input variable does not match the input pattern,
            # we break the loop and try the next permutation.
            for input_pattern, input_var in zip(self.inputs, inputs):
                if not input_pattern.match(
                    input_var, rtol=rtol, atol=atol, memo=incr_memo
                ):
                    inputs_matched = False
                    break
            # If all input variables match the input patterns, we break the loop.
            if inputs_matched:
                break
            # Reset the binding of matched patterns if the inputs do not match.
            for pattern in incr_memo:
                pattern.reset_binding()

        # If any inputs permutation matches the input patterns, return True.
        if inputs_matched:
            # record the matched onnx node
            self.onnx_node = onnx_node
            # update the memo set
            if memo is not None:
                memo.update(incr_memo)
                memo.add(self)
            return True
        return False

    def reset_binding(self) -> None:
        """Reset the bound onnx node."""
        if hasattr(self, "onnx_node"):
            del self.onnx_node


class _GraphPattern:
    """A class to represent a graph pattern object.

    Attributes:
        nodes: The list of node patterns.
        variables: The dictionary of variable patterns.
    """

    def __init__(self, pattern: str) -> None:
        """Initialize a graph pattern object.

        Args:
            pattern: The pattern string to match.

        Examples:
            >>> owning_graph = _GraphPattern(
            ...     '''
            ...     y0 = Sigmoid(x)
            ...     y1 = Mul<commutative=True>(x, y0)
            ...     '''
            ... )
        """
        self.nodes: List[_NodePattern] = []
        self.variables: Dict[str, _VariablePattern] = {}

        # parse the pattern into expressions line by line
        for expression in pattern.split("\n"):
            # skip empty lines
            if not expression.strip():
                continue
            # remove spaces in the expression
            expression.replace(" ", "")
            # split the expression into left and right parts via "="
            outputs, right_expr = _split_expression(expression, delimiter="=")
            # parse the right part of the expression
            match = re.match(
                r"^ *(?P<op_type>[^<\(]*)"
                r"(<(?P<attributes>.*?)>)?"
                r"(\((?P<inputs>.*?)\)) *$",
                right_expr,
            )
            # create the new node pattern object
            _NodePattern(
                owning_graph=self,
                op_type=match.group("op_type"),
                inputs=match.group("inputs"),
                outputs=outputs,
                attributes=match.group("attributes"),
            )

    def match(self, onnx_node: "OnnxNode", rtol: float, atol: float) -> bool:
        """Match the onnx node with the graph pattern.

        Args:
            onnx_node: The onnx node to match.
            rtol: The relative tolerance parameter for the match.
                Only used for float comparison.
            atol: The absolute tolerance parameter for the match.
                Only used for float comparison.

        Returns:
            True if the onnx node matches the graph pattern, False otherwise.
        """
        # Reset the binding of matched onnx_node/onnx_var for multiple matching.
        self.reset_binding()
        # We match the graph pattern recursively from the last node pattern
        # to the first node pattern.
        node_pattern = self.nodes[-1]
        # We suppose the given onnx node corresponds to the last node pattern.
        for output_pattern, output_var in zip(node_pattern.outputs, onnx_node.outputs):
            # If any output variable does not match the output pattern,
            # we return False.
            if not output_pattern.match(
                output_var,
                rtol=rtol,
                atol=atol,
            ):
                return False

        return True

    def reset_binding(self) -> None:
        """Reset the bound onnx nodes and variables."""
        for node_pattern in self.nodes:
            node_pattern.reset_binding()
        for var_pattern in self.variables.values():
            var_pattern.reset_binding()

    @property
    def matched_nodes(self) -> Tuple["OnnxNode", ...]:
        """Return the matched onnx nodes."""
        matched_nodes: List[OnnxNode] = [
            node_pattern.onnx_node for node_pattern in self.nodes
        ]
        return tuple(matched_nodes)


class PatternMatcher:
    """A class to represent a pattern matcher object.

    Attributes:
        matched_case: The matched case of the pattern matcher.
        matched_nodes: The matched onnx nodes of the pattern matcher.
        patterns: The dictionary of graph patterns to match.
    """

    def __init__(self, patterns: Union[str, Mapping[Optional["Enum"], str]]) -> None:
        """Initialize a pattern matcher object.

        Args:
            patterns: Single or dictionary of pattern strings to match.

        Examples:
            >>> pattern_matcher = PatternMatcher(
            ...    '''
            ...    y0 = Sigmoid(x)
            ...    y1 = Mul<commutative=True>(x, y0)
            ...    '''
            ... )
            >>> from enum import Enum
            >>> class MatchedCase(Enum):
            ...     SqrtReciprocal = 0
            ...     AddSqrtReciprocal = 1
            ...     SqrtAddReciprocal = 2
            >>> pattern_matcher = PatternMatcher(
            ...    {
            ...         MatchedCase.SqrtReciprocal: '''
            ...             y0 = Sqrt(x)
            ...             y1<dtype=FLOAT32> = Reciprocal(y0)
            ...         ''',
            ...         MatchedCase.AddSqrtReciprocal: '''
            ...             y0 = Add<commutative=True>(x, eps<numel=1, is_param=True>)
            ...             y1 = Sqrt(y0)
            ...             y2<dtype=FLOAT32> = Reciprocal(y1)
            ...         ''',
            ...         MatchedCase.SqrtAddReciprocal: '''
            ...             y0 = Sqrt(x)
            ...             y1 = Add<commutative=True>(y0, eps<numel=1, is_param=True>)
            ...             y2<dtype=FLOAT32> = Reciprocal(y1)
            ...         ''',
            ...    }
            ... )
        """
        self.matched_case: Optional["Enum"]
        self.matched_nodes: Tuple["OnnxNode", ...]
        self.patterns: Dict[Optional[Enum], _GraphPattern] = {}
        # convert a single pattern to a dictionary
        if isinstance(patterns, str):
            patterns = {None: patterns}
        # create _GraphPattern objects for each pattern string
        for case, pattern in patterns.items():
            self.patterns[case] = _GraphPattern(pattern)

    def match(
        self,
        onnx_node: "OnnxNode",
        rtol: float = 1e-07,
        atol: float = 0.0,
    ) -> bool:
        """Match the onnx node with the patterns.

        Args:
            onnx_node: The onnx node to match.
            rtol: The relative tolerance parameter for the match.
                Only used for float comparison.
            atol: The absolute tolerance parameter for the match.
                Only used for float comparison.

        Returns:
            True if the onnx node matches any of the patterns, False otherwise.

        Examples:
            >>> pattern_matcher = PatternMatcher(...)
            >>> if pattern_matcher.match(onnx_node):
            ...     print(pattern_matcher.matched_case)
            ...     for onnx_node in pattern_matcher.matched_nodes:
            ...         print(onnx_node)
        """
        for case, pattern in self.patterns.items():
            if pattern.match(onnx_node, rtol=rtol, atol=atol):
                self.matched_case = case
                self.matched_nodes = pattern.matched_nodes
                return True
        return False
