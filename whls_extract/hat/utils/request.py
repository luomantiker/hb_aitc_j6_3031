import copy
import json
import logging
import os
import subprocess

logger = logging.getLogger(__name__)


def update_job_strategy(upload_strategy):
    """Update the strategy of training jobs.

    The pipeline of update is GET -> update -> POST.
    The format for uploading strategy is as follows:
      [
        {"name": 'sync_bn', "usage_detail": 'False'},
        {"name": 'checkpoint', "usage_detail": 'False'},
        ...
      ]

    """

    user_token = os.environ.get("USER_TOKEN", None)
    job_id = os.environ.get("JOB_ID", None)
    cluster = os.environ.get("CLUSTER", None)
    if None in (user_token, job_id, cluster):
        logger.warning(
            "Train strategy upload failed ! "
            + "Cannot get USER_TOKEN/JOB_ID/CLUSTER from env."
        )
        return

    try:
        output = subprocess.run(
            [
                "curl"
                + " --location"
                + " request"
                + " GET 'http://api.aidi.hobot.cc/infra/api/v1alpha/job_manager/job/get?job_id=%s&cluster=%s'"  # noqa
                % (job_id, cluster)
                + " --header"
                + " 'X-Forwarded-User:'%s''" % (user_token)
            ],
            shell=True,
            stdout=subprocess.PIPE,
        )
        record = eval(output.stdout.decode())

        strategy = []
        if "train_tricks" in record["data"]:
            strategy = copy.deepcopy(record["data"]["train_tricks"])

        stage = os.environ.get("HAT_TRAINING_STEP", None)
        strategy.append({"stage": stage, "tricks": upload_strategy})
        new_record = {"data": {"train_tricks": strategy}}
        new_record["data"]["job_id"] = job_id
        new_record["data"]["cluster"] = cluster

        subprocess.check_output(
            [
                "curl"
                + " --location"
                + " request"
                + " POST 'http://api.aidi.hobot.cc/infra/api/v1alpha/job_manager/job/update'"  # noqa
                + " --header"
                + " 'Content-Type: application/json'"
                + " --header"
                + " 'X-Forwarded-User:%s'" % (user_token)
                + " --data '%s'" % (json.dumps(new_record["data"]))
            ],
            shell=True,
        )
        logger.warning("Upload train strategy successfully.")
    except Exception as e:
        logger.warning(e)
        logger.warning("Train strategy upload failed !")
