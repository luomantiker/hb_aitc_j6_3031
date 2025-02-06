import contextlib
import json
import logging
import os

from hat.utils.checkpoint import load_checkpoint, load_state_dict
from hat.utils.global_var import set_value
from hat.utils.logger import MSGColor, format_msg

logger = logging.Logger(__name__)

IS_LOCAL = not os.path.exists("/running_package")


def use_elastic():
    return bool(int(os.getenv("USE_ELASTIC", "0")))


def elastic_need_resume():
    is_restart_training = int(os.getenv("ELASTIC_NEED_RESUME", "0")) > 0

    return is_restart_training and use_elastic()


class JsonStorage:
    def __init__(self, json_file) -> None:
        self.json_file = json_file

    def _load(self):
        try:
            with open(self.json_file, "r") as f:
                data = json.load(f)
                if not data:
                    return {}
                return data
        except FileNotFoundError:
            return {}

    def _save(self, data):
        with open(self.json_file, "w") as file:
            json.dump(data, file, indent=4)

    @classmethod
    def load_data(cls, json_file):
        return cls(json_file)._load()

    @classmethod
    def save_data(cls, json_file, data):
        return cls(json_file)._save(data)


class ElasticState:

    FINISHED = "FINISHED"
    RUNNING = "RUNNING"

    def __init__(self) -> None:
        if not IS_LOCAL:
            self._elastic_record = os.path.join(
                "/job_data/elastic/",
                "elastic_record.json",
            )
        else:
            self._elastic_record = os.path.join(
                ".elastic/",
                "elastic_record.json",
            )
        with contextlib.suppress(FileExistsError):
            os.makedirs(os.path.dirname(self._elastic_record), exist_ok=True)

    @classmethod
    def save_checkpoint(cls, checkpoint):
        """Save checkpoint path to json file.

        Args:
            checkpoint: path of checkpoint.
        """
        cur_command_info = cls().cur_command_info
        cur_command_info["checkpoint_path"] = checkpoint

        cls()._update_cur_command_info(cur_command_info)

    @classmethod
    def load_checkpoint(cls, model):
        """Resume model state from saved checkpoint."""
        checkpoint_path = cls().cur_command_info.get("checkpoint_path", None)
        if checkpoint_path and model:
            # Strangely, log will not be printed by using `logger.info()`
            logger.warning(
                format_msg(
                    f"Elastic will resume from checkpoint: {checkpoint_path}",
                    MSGColor.GREEN,
                )
            )
            model_checkpoint = load_checkpoint(
                path_or_dict=checkpoint_path,
                map_location="cpu",
                check_hash=False,
            )
            set_value("model_checkpoint", model_checkpoint)
            model = load_state_dict(
                model,
                path_or_dict=model_checkpoint,
                allow_miss=False,
                ignore_extra=False,
                verbose=True,
            )
            logger.warning(
                format_msg(
                    f"Elastic has resumed from checkpoint: {checkpoint_path}",
                    MSGColor.GREEN,
                )
            )

        return model

    @classmethod
    def set_cur_state(cls, state):
        """Update current command running state."""
        cur_command_info = cls().cur_command_info
        cur_command_info["state"] = state

        cls()._update_cur_command_info(cur_command_info)

    def _update_cur_command_info(self, cur_command_info):
        data = self.all_state_info
        data[self.cur_command] = cur_command_info
        JsonStorage.save_data(self._elastic_record, data)

    @property
    def cur_command_info(self):
        cur_command_info = self.all_state_info.get(self.cur_command, {})

        return cur_command_info

    @property
    def all_state_info(self):

        all_data = JsonStorage.load_data(self._elastic_record)
        return all_data

    @property
    def cur_command(self):
        return os.getenv("CURRENT_COMMAND", "")

    def cur_command_state(self):
        return self.cur_command_info.get("state", self.RUNNING)

    def get_cur_checkpoint(self):
        return self.cur_command_info.get("checkpoint_path", None)
