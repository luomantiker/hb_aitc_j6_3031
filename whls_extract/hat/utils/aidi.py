import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List, Optional

from hat.utils.config import crop_configs
from hat.utils.distributed import rank_zero_only
from hat.utils.logger import MSGColor, format_msg

logger = logging.getLogger(__name__)

try:
    from aidisdk import AIDIClient
    from aidisdk.experiment import Image, Table
    from aidisdk.experiment.artifact import Artifact
    from aidisdk.experiment.model_eval import ModelEvalDataset
    from aidisdk.utils import running_in_cluster
    from dataclasses_json import DataClassJsonMixin

    IS_LOCAL = not running_in_cluster()
except ImportError:
    logger.warning("`aidisdk` dependency is not available.")
    AIDIClient, Image, Table, Artifact, ModelEvalDataset = (
        None,
        None,
        None,
        None,
        None,
    )
    DataClassJsonMixin = object

    IS_LOCAL = True


__all__ = ["get_aidi_client", "get_aidi_token"]

CI_token = (
    "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJleHAiOjIyNjY5NzU4NDYsIlR"
    "va2VuVHlwZSI6ImxkYXAiLCJVc2VyTmFtZSI6ImhhdF90ZWFtIiwiT3JnYW5pemF"
    "0aW9uIjoicmVndWxhci1lbmdpbmVlciIsIk9yZ2FuaXphdGlvbklEIjoxfQ.Q1SC"
    "pQtWMFUcYIVmFDbh5z_mrBV8bnPzKILtvSZ-YlJCdSUxUjTXVXjwSK8LLWkOc45_"
    "VTdG1CPMqCUB-KzvUzcJP0qfPOC6YmaWKWTvY7CE-fhs3MkC3Z_SNQvG-hxcKlIY"
    "LspQixOW8h20IuO7pMA97unob_Rk3z2u5Cxz8eXnyj_H3VlNi4__g3Vv9dXxtxnM"
    "fgt6Z3LAlnquEsAurja9RbRGXujsGo4MzbD_nlWxThSrWZzKh3pvb4jQIUOQfBm5"
    "BF3cpWBjgB2_z0g4zLBbkidFytJoT321JpVvAyfmUjzW7Uga0aQwfhaZkR318ijw"
    "ZS87MnTkacgCiYPf_w"
)


def get_aidi_client():
    """Get user client.

    Returns:
      client: User client.
    """
    if AIDIClient is None:
        return None
    return AIDIClient()


def get_aidi_token():
    """Get user token.

    Returns:
      token: User token.
    """
    client = AIDIClient()
    token = client.session.token
    return token


def is_running_on_aidi():
    """Whether is running on aidi."""
    return (
        os.environ.get("WORKING_PATH", None) is not None
        and os.environ.get("PBS_JOBNAME", None) is not None
    )


@lru_cache()
def check_tracking_status(aidi_client=None):
    """Check whether to enabled aidi tracking."""
    if aidi_client is None:
        aidi_client = get_aidi_client()

    enable_tracking = bool(
        int(os.environ.get("HAT_ENABLE_MODEL_TRACKING", "0"))
    )

    try:
        enabled = aidi_client.experiment.enabled
    except Exception as e:
        logger.info(
            format_msg(
                f"AIDI Experiment: Get aidi experiment.enabled failed: {str(e)}, will set enable_tracking=False",  # noqa E501
                MSGColor.RED,
            )
        )
        enabled = False
    return enabled and enable_tracking


@rank_zero_only
def tracking_model_as_input(
    model_name: str,
    model_version: str,
    stage: str,
):
    """Tracking aidi model.

    Args:
        model_name: model name.
        model_version: model version.
        stage: model training stage.
    """
    client = get_aidi_client()

    if check_tracking_status(client):
        try:
            client.experiment.log_input_artifact(
                Artifact.from_model_stage(
                    client.model.finditem(
                        model_name=model_name,
                        model_version=model_version,
                        stage=stage,
                    )
                )
            )
            logger.info(
                format_msg(
                    f"AIDI Tracking:  tracking input checkpoint for model[{model_name}], "  # noqa E501
                    f"version[{model_version}], stage[{stage}] successfully.",  # noqa E501
                    MSGColor.GREEN,
                )
            )
        except Exception as e:
            logger.info(
                format_msg(
                    f"AIDI Tracking:  tracking input checkpoint for model[{model_name}], "  # noqa E501
                    f"version[{model_version}], stage[{stage}] failed: {e}.",  # noqa E501
                    MSGColor.RED,
                )
            )


@dataclass
class EvalResult(DataClassJsonMixin):
    """Dataclass for result of evaluation."""

    summary: dict = None
    tables: List[Table] = None
    plots: List[dict] = None
    images: List[Image] = None
    confusion_matrixes: List[dict] = None


class AIDIExperimentLogger:
    def __init__(self) -> None:
        self._client = None

    def init(
        self,
        experiment_name: str,
        project_id: str,
        experiment_path: str,
        enable_tracking: str,
        run_name: str,
        runtime: str = "local",
        config_file: str = None,
    ):
        if self.client.experiment.get_experiment(experiment_name) is None:
            self.client.experiment.create_experiment(
                name=experiment_name,
                project_id=project_id,
                experiment_path=experiment_path,
            )
        with self.client.experiment.init(
            experiment_name=experiment_name,
            run_name=run_name,
            enabled=enable_tracking,
        ) as run:
            run.log_runtime(runtime=runtime, config_file=config_file)

    def init_group(self, group):
        try:
            if self.enabled_tracking():
                self.client.experiment.init_group(group)
        except Exception as e:
            logger.warning(f"failed to init_group: {str(e)}")

    def log_config(self, config):
        try:
            if self.enabled_tracking():
                self.client.experiment.log_config(crop_configs(config))
        except Exception as e:
            logger.warning(f"failed to log_config: {str(e)}")

    def log_artifact(
        self,
        artifact_name: str,
        artifact_type: str,
        artifact_aliases: List[str],
        artifact_tags: List[str],
        files: Optional[List[str]] = None,
        overwrite: bool = False,
    ):
        if self.enabled_tracking():
            output_artifact = Artifact(
                artifact_name,
                type=artifact_type,
                aliases=artifact_aliases,
                tags=artifact_tags,
            )

            if files:
                for file in files:
                    if os.path.exists(file):
                        output_artifact.add_file(file, overwrite=overwrite)

            self.client.experiment.log_artifact(
                artifact=output_artifact,
                overwrite=overwrite,
            )

    @classmethod
    def enabled_tracking(cls, client=None):
        """Check whether to enabled aidi tracking."""
        return check_tracking_status(client)

    @classmethod
    def download_checkpoint_from_artifact(
        cls,
        artifact_path: str,
        enable_tracking: bool = False,
    ):
        (
            _,
            _,
            model_name,
            training_stage,
            entry_alias,
            ckpt_name,
        ) = artifact_path.split(os.sep)
        artifact_name = f"{model_name}-{training_stage}:{entry_alias}"

        input_artifact = cls().client.experiment.use_artifact(
            artifact_name, trace=enable_tracking
        )
        return input_artifact.get_file(ckpt_name)

    def log_exception(self, exception):
        if not IS_LOCAL:
            self.client.single_job.log_exception(exception)

    def log_metrics(
        self,
        metrics_list: Dict[str, Any],
        epoch_id: int,
        step_id: int,
        **kwargs,
    ):
        for k, v in metrics_list:
            if isinstance(v, EvalResult):
                if v.summary is not None:
                    self.client.experiment.log_summary(v.summary)
                if v.tables is not None:
                    for table in v.tables:
                        self.client.experiment.log_table(table)
                if v.plots is not None:
                    for plot in v.plots:
                        self.client.experiment.log_plot(
                            plot["Table"].name,
                            plot["Table"],
                            plot["Line"],
                        )
                if v.images is not None:
                    for image in v.images:
                        self.client.experiment.log_image(image)
                    self.client.experiment.flush()
                if v.confusion_matrixes is not None:
                    for confusion_matrix in v.confusion_matrixes:
                        self.client.experiment.log_confusion_matrix(
                            confusion_matrix["labels"],
                            confusion_matrix["matrix"],
                            confusion_matrix["name"],
                        )
            else:
                self.client.experiment.log_metrics(
                    key=k,
                    value=v,
                    step=step_id,
                    epoch_id=epoch_id,
                )

    @property
    def client(self):
        if self._client is None:
            self._client = get_aidi_client()
        return self._client
