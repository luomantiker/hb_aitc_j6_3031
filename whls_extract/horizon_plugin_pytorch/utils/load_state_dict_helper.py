import logging
from functools import wraps
from typing import Optional

from torch import nn

logger = logging.getLogger(__name__)


def load_state_dict_ignore_act(
    obj: nn.Module,
    state_dict: dict,
    prefix,
    local_metadata,
    strict,
    missing_keys,
    unexpected_keys,
    error_msgs,
):
    ignored_submod = "activation_post_process"
    ignored_prefix = prefix + ignored_submod
    ignored_buffers = []

    for k in state_dict:
        if k.startswith(ignored_prefix):
            ignored_buffers.append(k)

    for k in ignored_buffers:
        state_dict.pop(k)

    return nn.Module._load_from_state_dict(
        obj,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    )


def replace_mod_name(
    old_name: Optional[str], new_name: Optional[str], do_on_version=None
):
    """Decorate Module._load_from_state_dict to replace mod name in state_dict before loading it."""  # noqa: E501

    def decorator(_load_from_state_dict):
        @wraps(_load_from_state_dict)
        def load_with_replaced_mod_name(
            self,
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        ):
            if do_on_version is None or do_on_version(
                local_metadata.get("version", self._version)
            ):
                if old_name is None:
                    old_prefix = prefix
                else:
                    old_prefix = prefix + old_name + "."
                if new_name is None:
                    new_prefix = prefix
                else:
                    new_prefix = prefix + new_name + "."

                replace_key_mapping = {}

                for k in state_dict:
                    k: str
                    if k.startswith(old_prefix) and (
                        new_name is None or not k.startswith(new_prefix)
                    ):
                        suffix = k[len(old_prefix) :]
                        replace_key_mapping[k] = new_prefix + suffix

                if len(replace_key_mapping) > 0:
                    logger.warning(
                        "Following names in old version state_dict are "
                        "automatically modified to load them into the new "
                        "operator implementation, please update statd_dict by "
                        "saving it again to remove this warning"
                    )

                for k, v in replace_key_mapping.items():
                    logger.warning("{} -> {}".format(k, v))
                    state_dict[v] = state_dict.pop(k)

            return _load_from_state_dict(
                self,
                state_dict,
                prefix,
                local_metadata,
                strict,
                missing_keys,
                unexpected_keys,
                error_msgs,
            )

        return load_with_replaced_mod_name

    return decorator


def get_version(mod, prefix, local_metadata):
    version = local_metadata.get("version", None)
    if version is None:
        version = mod._version
        logger.warning(
            (
                "Do not find version in loaded metadata of module {}, "
                "use the default version {}.\nMost common reason is that "
                "the state_dict._metadata is missed when modifying "
                "state_dict".format(prefix, version)
            )
        )
    # Set mod._version to make sure saved state_dict is correct.
    mod._version = version

    return version
