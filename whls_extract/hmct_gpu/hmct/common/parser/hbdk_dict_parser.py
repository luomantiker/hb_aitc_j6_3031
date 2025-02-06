from typing import Dict, List, Optional, Sequence, Tuple, Union


class HbdkDictParser:
    """Given hbdk_dict, return formatted hbdk-cc options.

    Configs in hbdk dict can be classified to two types:

        The first is config independent on model input, such as --O2, --fast,
        --core-num 2. Group these configs in a string, and pass it to hbdk dict
        with a specified key "hbdk_pass_through_params". For example hbdk_dict =
        {'hbdk_pass_through_params': '--O2 --fast --core-num 2'}

        The second is config related to specific input where the param is also a
        dict whose key is input name. As some model input may not appear in the
        config, you would better set the '_default_value' field for the config.
        For example,
        hbdk_dict = {'input-source': {'data': 'pyramid', '_default_value': 'ddr'}}.

    So, a complicated hbdk dict may look like
    {'hbdk_pass_through_params': '--O2 --fast --core-num 2',
    'input-source': {'data': 'pyramid', '_default_value': 'ddr'}}
    """

    def __init__(
        self, hbdk_dict: Optional[Dict[str, Union[str, Dict[str, str]]]]
    ) -> None:
        self.hbdk_dict = {} if hbdk_dict is None else hbdk_dict
        assert isinstance(self.hbdk_dict, dict), (
            f"type(hbdk_dict) should be either Dict or None, "
            f"but got {type(self.hbdk_dict)}."
        )

        self.input_independent_config, self.input_dependent_config = (
            self._parse_hbdk_dict()
        )

    def _parse_hbdk_dict(self) -> Tuple[List[str], Dict[str, Dict[str, str]]]:
        """Parse the hbdk configs into two classes of configs.

        Refer to the HbdkDictParser description above.

        Returns:
            A tuple with two classes of configs.
        """
        input_independent_config: List[str] = []
        input_dependent_config: Dict[str, Dict[str, str]] = {}
        for k, v in self.hbdk_dict.items():
            if k == "hbdk_pass_through_params":
                assert isinstance(v, str)
                input_independent_config = v.split()
            else:
                assert isinstance(v, dict)
                input_dependent_config.update({k: v})

        return input_independent_config, input_dependent_config

    def _generate_options_wrt_input(self, input_names: Sequence[str]) -> List[str]:
        """Convert hbdk configs related to specific input to hbdk-cc options.

        Args:
            input_names: specific input names to generate hbdk options

        Returns:
            List of formatted input-related options for hbdk-cc.
        """
        input_dependent_options = []
        for k, v in self.input_dependent_config.items():
            assert isinstance(v, dict)
            params = []
            default_value = v.get("_default_value", None)
            for input_name in input_names:
                if input_name in v:
                    params.append(v[input_name])
                elif default_value is not None:
                    params.append(default_value)
                else:
                    self.hbdk_dict[k].update({"_default_value": "xxx"})
                    raise ValueError(
                        f"For model input: {input_name} which does not appear "
                        f"in {k} config, a default value must be set. "
                        f"For example, hbdk_dict={self.hbdk_dict}.",
                    )
            if len(params):
                input_dependent_options.extend(["--" + k, ",".join(params)])
        return input_dependent_options

    def generate_options_for_cc(
        self,
        given_inputs: Sequence[str],
        **extra_options,
    ) -> List[str]:
        """Generate options for hbdk-cc.

        Args:
            given_inputs: input names to generate the options which are
                related to inputs.
            extra_options: extra options which are generated internally.

        Returns:
            List of options for hbdk-cc.
        """
        input_independent_options = self.input_independent_config.copy()
        # update the existing config
        for k, v in extra_options.items():
            if "--" + k in input_independent_options:
                pos = input_independent_options.index("--" + k)
                assert pos + 1 < len(input_independent_options)
                input_independent_options[pos + 1] = v
            else:
                input_independent_options.extend(["--" + k, v])
        input_dependent_options = self._generate_options_wrt_input(given_inputs)

        return input_independent_options + input_dependent_options

    def get_input_source(self) -> Union[Dict[str, str], None]:
        """Get the input source.

        Returns:
            A dictionary from input names to input source or None.
        """
        return self.input_dependent_config.get("input-source", None)
