import datetime
import os
import signal
import sys
import warnings
from importlib.util import find_spec
from pathlib import Path
from typing import Callable, Dict, List

import hydra
import torch
from lightning.pytorch import Callback
from lightning.pytorch.loggers import Logger
from lightning.pytorch.utilities import rank_zero_only
from omegaconf import DictConfig

from gabbro.utils import pylogger, rich_utils

log = pylogger.get_pylogger(__name__)


def translate_bash_range(wildcard: str, verbose: bool = False):
    """Translate bash range to list of strings with the corresponding numbers.

    Parameters
    ----------
    wildcard : str
        Wildcard string with bash range (or not).
    verbose : bool, optional
        If True, print debug messages.

    Returns
    -------
    list
        List of strings with the corresponding numbers.
    """

    # raise value error if two ranges are found
    if wildcard.count("{") > 1:
        raise ValueError(
            f"Only one range is allowed in the wildcard. Provided the following wildcard: {wildcard}"
        )

    if "{" in wildcard and ".." in wildcard and "}" in wildcard:
        log.info("Bash range found in wildcard --> translating to list of remaining wildcards.")
        start = wildcard.find("{")
        end = wildcard.find("}")
        prefix = wildcard[:start]
        suffix = wildcard[end + 1 :]
        wildcard_range = wildcard[start + 1 : end]
        start_number = int(wildcard_range.split("..")[0])
        end_number = int(wildcard_range.split("..")[1])
        if verbose:
            log.info(
                f"Prefix: {prefix}, Suffix: {suffix}, Start: {start_number}, End: {end_number}"
            )
        return [f"{prefix}{i}{suffix}" for i in range(start_number, end_number + 1)]
    else:
        # print("No range found in wildcard")
        return [wildcard]


def task_wrapper(task_func: Callable) -> Callable:
    """Optional decorator that wraps the task function in extra utilities.

    Makes multirun more resistant to failure.

    Utilities:
    - Calling the `utils.extras()` before the task is started
    - Calling the `utils.close_loggers()` after the task is finished or failed
    - Logging the exception if occurs
    - Logging the output dir
    """

    def wrap(cfg: DictConfig):
        # execute the task
        try:
            # apply extra utilities
            extras(cfg)

            func_return = task_func(cfg=cfg)

        # things to do if exception occurs
        except Exception as ex:
            # save exception to `.log` file
            log.exception("")

            # when using hydra plugins like Optuna, you might want to disable raising exception
            # to avoid multirun failure
            raise ex

        # things to always do after either success or exception
        finally:
            # display output dir path in terminal
            log.info(f"Output dir: {cfg.paths.output_dir}")

            # close loggers (even if exception occurs so multirun won't fail)
            close_loggers()

        return func_return

    return wrap


def extras(cfg: DictConfig) -> None:
    """Applies optional utilities before the task is started.

    Utilities:
    - Ignoring python warnings
    - Setting tags from command line
    - Rich config printing
    """

    # return if no `extras` config
    if not cfg.get("extras"):
        log.warning("Extras config not found! <cfg.extras=null>")
        return

    # disable python warnings
    if cfg.extras.get("ignore_warnings"):
        log.info("Disabling python warnings! <cfg.extras.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # prompt user to input tags from command line if none are provided in the config
    if cfg.extras.get("enforce_tags"):
        log.info("Enforcing tags! <cfg.extras.enforce_tags=True>")
        rich_utils.enforce_tags(cfg, save_to_file=True)

    # pretty print config tree using Rich library
    if cfg.extras.get("print_config"):
        log.info("Printing config tree with Rich! <cfg.extras.print_config=True>")
        rich_utils.print_config_tree(cfg, resolve=True, save_to_file=True)


def instantiate_callbacks(callbacks_cfg: DictConfig, ckpt_path: str = None) -> Dict[str, Callback]:
    """Instantiates callbacks from config."""
    callbacks: Dict[str, Callback] = {}

    if not callbacks_cfg:
        log.warning("No callback configs found! Skipping..")
        return callbacks

    if not isinstance(callbacks_cfg, DictConfig):
        raise TypeError("Callbacks config must be a DictConfig!")

    for cb_name, cb_conf in callbacks_cfg.items():
        if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
            log.info(f"Instantiating callback <{cb_conf._target_}>")
            callbacks[cb_name] = hydra.utils.instantiate(cb_conf)

    return callbacks


def instantiate_loggers(logger_cfg: DictConfig) -> List[Logger]:
    """Instantiates loggers from config."""
    logger: List[Logger] = []

    if not logger_cfg:
        log.warning("No logger configs found! Skipping...")
        return logger

    if not isinstance(logger_cfg, DictConfig):
        raise TypeError("Logger config must be a DictConfig!")

    for _, lg_conf in logger_cfg.items():
        if isinstance(lg_conf, DictConfig) and "_target_" in lg_conf:
            log.info(f"Instantiating logger <{lg_conf._target_}>")
            logger.append(hydra.utils.instantiate(lg_conf))

    return logger


@rank_zero_only
def log_hyperparameters(object_dict: dict) -> None:
    """Controls which config parts are saved by lightning loggers.

    Additionally saves:
    - Number of model parameters
    """

    hparams = {}

    cfg = object_dict["cfg"]
    model = object_dict["model"]
    trainer = object_dict["trainer"]

    if not trainer.logger:
        log.warning("Logger not found! Skipping hyperparameter logging...")
        return

    hparams["model"] = cfg["model"]

    # save number of model parameters
    hparams["model/params/total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params/trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams["model/params/non_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    hparams["data"] = cfg["data"]
    hparams["trainer"] = cfg["trainer"]

    hparams["callbacks"] = cfg.get("callbacks")
    hparams["extras"] = cfg.get("extras")

    hparams["task_name"] = cfg.get("task_name")
    hparams["tags"] = cfg.get("tags")
    hparams["ckpt_path"] = cfg.get("ckpt_path")
    hparams["seed"] = cfg.get("seed")
    hparams["run_note"] = cfg.get("run_note")
    hparams["git"] = object_dict.get("git")
    hparams["slurm"] = object_dict.get("slurm")
    hparams["load_weights_from"] = object_dict.get("load_weights_from")
    hparams["gpu_properties"] = object_dict.get("gpu_properties")

    # send hparams to all loggers
    for logger in trainer.loggers:
        logger.log_hyperparams(hparams)


def get_metric_value(metric_dict: dict, metric_name: str) -> float:
    """Safely retrieves value of the metric logged in LightningModule."""

    if not metric_name:
        log.info("Metric name is None! Skipping metric value retrieval...")
        return None

    if metric_name not in metric_dict:
        raise Exception(
            f"Metric value not found! <metric_name={metric_name}>\n"
            "Make sure metric name logged in LightningModule is correct!\n"
            "Make sure `optimized_metric` name in `hparams_search` config is correct!"
        )

    metric_value = metric_dict[metric_name].item()
    log.info(f"Retrieved metric value! <{metric_name}={metric_value}>")

    return metric_value


def close_loggers() -> None:
    """Makes sure all loggers closed properly (prevents logging failure during multirun)."""

    log.info("Closing loggers...")

    if find_spec("wandb"):  # if wandb is installed
        import wandb

        if wandb.run:
            log.info("Closing wandb!")
            wandb.finish()


@rank_zero_only
def save_file(path: str, content: str) -> None:
    """Save file in rank zero mode (only on one process in multi-GPU setup)."""
    with open(path, "w+") as file:
        file.write(content)


def get_gpu_properties(verbose=True):
    """Returns a dictionary with the GPU properties of the available GPUs."""
    gpu_properties = {}
    for i in range(torch.cuda.device_count()):
        gpu_properties[f"rank{i}"] = torch.cuda.get_device_properties(i).name
    if verbose:
        print(gpu_properties)
    return gpu_properties


def update_existing_dict_values(d: dict, u: dict):
    """Update existing values in a dictionary with values from another dictionary."""
    for k, v in u.items():
        if k in d:
            d[k] = v
    return d


@rank_zero_only
def remove_empty_hydra_run_dir(dir_path):
    """Helper function to remove empty (by hydra created) output dir.

    Parameters
    ----------
    dir_path : str or Path
        Path to the output dir.
    """
    if isinstance(dir_path, str):
        dir_path = Path(dir_path)

    log.info(f"Removing empty (by hydra created) output dir {dir_path}")
    for file in ["config.yaml", "hydra.yaml", "overrides.yaml"]:
        filepath = dir_path / ".hydra" / file
        if filepath.exists():
            log.info(f"- Removing {filepath}")
            filepath.unlink()

    log.info(f"- Removing {dir_path / '.hydra'}")
    (dir_path / ".hydra").rmdir()
    log.info(f"- Removing {dir_path}")
    dir_path.rmdir()


def print_field_structure_of_ak_array(ak_array):
    """Prints the field structure of an awkward array."""
    for field in ak_array.fields:
        log.info(f" | {field}")
        if len(ak_array[field].fields) > 0:
            for inner_field in ak_array[field].fields:
                log.info(f" |   | {inner_field}")
                if len(ak_array[field][inner_field].fields) > 0:
                    for inner_inner_field in ak_array[field][inner_field].fields:
                        log.info(f" |   |   | {inner_inner_field}")


def translate_transformer_cfg_to_old_syntax(transformer_cfg, old_module_name):
    """Translate the config which would be passed to
    `gabbro.models.transformer.Transformer` to the config which would be passed to
    `gabbro.models.gpt_model.GPTDecoderStack`.

    Parameters
    ----------
    transformer_cfg : dict
        The config which would be passed to `gabbro.models.transformer.Transformer`.
    old_module_name : str
        The name of the module to which the config should be translated.
        Either `GPTDecoderStack` or `NormformerStack`.

    Returns
    -------
    dict
        The config which would be passed to `gabbro.models.gpt_model.GPTDecoderStack`.
    """

    if old_module_name == "NormformerStack":
        mapping = {
            "dim": "hidden_dim",
            "n_blocks": "num_blocks",
            "apply_causal_mask": "apply_causal_mask",
            "attn_cfg.num_heads": "num_heads",
            "attn_cfg.dropout_rate": "dropout_rate",
            "mlp_cfg.expansion_factor": "mlp_expansion_factor",
        }
    elif old_module_name == "GPTDecoderStack":
        mapping = {
            "dim": "embedding_dim",
            "apply_causal_mask": "apply_causal_mask",
            "n_blocks": "n_GPT_blocks",
            "norm_after_blocks": None,
            "attn_cfg.dropout_rate": "attention_dropout",
            "attn_cfg.num_heads": "n_heads",
            "attn_cfg.norm_before": "pre_norm",
            "residual_cfg.gate_type": None,
            "residual_cfg.init_value": None,
            "mlp_cfg.dropout_rate": "mlp_dropout",
            "mlp_cfg.norm_before": None,
            "mlp_cfg.expansion_factor": "mlp_factor",
            "mlp_cfg.activation": None,
            "apply_padding_mask_fix": "apply_padding_mask_fix",
        }
    else:
        raise ValueError(
            f"Module name {old_module_name} not supported! "
            "Only GPTDecoderStack and NormformerStack are supported."
        )

    translated_cfg = {}

    log.info("Translating transformer config to GPTDecoder syntax!")
    log.info("Transformer config:")
    for key, value in transformer_cfg.items():
        log.info(f" - {key}: {value}")

    for old_key, new_key in mapping.items():
        if new_key is None:
            log.warning(f"Key {old_key} not found in mapping! Skipping...")
            continue
        keys = old_key.split(".")
        value = transformer_cfg
        for key in keys:
            if key in value:
                value = value[key]
            else:
                value = None
                break
        if value is not None:
            translated_cfg[new_key] = value
    if old_module_name == "NormformerStack":
        if "residual_cfg" in transformer_cfg:
            if "init_value" in transformer_cfg["residual_cfg"]:
                translated_cfg["init_identity"] = (
                    True if transformer_cfg["residual_cfg"]["init_value"] == 0.0 else False
                )
    log.info("Translated config:")
    for key, value in translated_cfg.items():
        log.info(f" - {key}: {value}")

    return translated_cfg


def compare_two_pp_dicts(
    pp_dict_1: dict,
    pp_dict_2: dict,
    ignore_features_not_present_in_second_dict: bool = False,
):
    """Compare two pp_dicts.

    Parameters
    ----------
    pp_dict_1 : dict
        The first preprocessing dictionary to compare.
    pp_dict_2 : dict
        The second preprocessing dictionary to compare.
    ignore_features_not_present_in_second_dict : bool, optional
        If True, features that are present in the first dictionary but not in the second
        dictionary will be ignored. This is useful if the second dictionary is a subset
        of the first dictionary, by default False.
    """

    # check that for each feature that is present in both the backbone input and
    # the vq-vae input, the preprocessing parameters are the same
    for feature in pp_dict_1:
        if feature not in pp_dict_2 and not ignore_features_not_present_in_second_dict:
            raise ValueError(
                f"Feature {feature} is present in the first preprocessing dictionary "
                "but not in the second preprocessing dictionary. "
                "Set `ignore_features_not_present_in_second_dict=True` to ignore this feature."
            )
        if feature not in pp_dict_2 and ignore_features_not_present_in_second_dict:
            log.warning(
                f"Feature {feature} is present in the second preprocessing dictionary "
                "but not in the first preprocessing dictionary. "
                "This feature will be ignored."
            )
            continue
        if pp_dict_1[feature] != pp_dict_2[feature]:
            raise ValueError(
                f"Preprocessing parameters for feature {feature} do not match "
                "between the backbone and the VQ-VAE model. "
                f"pp_dict_1: {pp_dict_1[feature]} - "
                f"pp_dict_2: {pp_dict_2[feature]}"
            )
