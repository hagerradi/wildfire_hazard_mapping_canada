import os

from comet_ml import Experiment


class CometLogger:
    def __init__(
        self,
        project_name: str,
        workspace: str,
        experiment_name: str,
        experiment_tags: list[str] | None = None,
        api_key: str | None = None,
    ):
        # get the api key from os vars.
        self.api_key = api_key or os.getenv("COMET_API_KEY")
        if not self.api_key:
            raise ValueError("COMET_API_KEY env. variable not set or no API key provided.")

        # init. experiment
        self.experiment = Experiment(
            api_key=api_key,
            project_name=project_name,
            workspace=workspace,
            log_code=False,
            log_graph=False,
            auto_param_logging=False,
            auto_metric_logging=False,
            auto_histogram_tensorboard_logging=False,
            auto_histogram_weight_logging=False,
            auto_histogram_gradient_logging=False,
            auto_histogram_activation_logging=False,
            auto_output_logging="False",
            auto_log_co2=False,
            log_env_details=True,
            log_env_gpu=True,
            log_env_cpu=True,
            log_env_network=False,
            log_env_host=False,
            log_git_metadata=False,
            log_git_patch=False,
        )

        self.experiment.set_name(experiment_name)

        if experiment_tags:
            self.experiment.add_tags(experiment_tags)

    def log_metrics(self, metrics: dict, step: int | None = None, epoch: int | None = None):
        if epoch is not None:
            self.experiment.set_epoch(epoch)
        self.experiment.log_metrics(metrics, step=step, epoch=epoch)

    def log_params(self, params: dict):
        self.experiment.log_parameters(params)
