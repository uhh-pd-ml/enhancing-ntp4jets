import argparse
import glob
import logging
import os
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from gabbro.data.data_tokenization import tokenize_jetclass_file
from gabbro.utils.binning import BinningTokenizer
from gabbro.utils.pylogger import get_pylogger

log = get_pylogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--jet_types",
    type=str,
    required=False,
    help="Comma-separated list of jet types. Has to match the jet-type names "
    "(with the trailing underscore) in the JetClass dataset.",
    default=",".join(
        [
            "ZJetsToNuNu_",
            "HToBB_",
            "HToCC_",
            "HToGG_",
            "HToWW4Q_",
            "HToWW2Q1L_",
            "ZToQQ_",
            "WToQQ_",
            "TTBar_",
            "TTBarLep_",
        ]
    ),
)
parser.add_argument(
    "--ckpt_path",
    type=str,
    default=None,
    help="Path to the VQ-VAE checkpoint to use for tokenization.",
)
parser.add_argument(
    "--binning_cfg_path",
    type=str,
    default=None,
    help="Path to the binning config to use for tokenization.",
)
parser.add_argument(
    "--n_files_train",
    type=int,
    required=False,
    default=100,
    help="Number of files to tokenize from the `train_100M` folder.",
)
parser.add_argument(
    "--n_files_val",
    type=int,
    required=False,
    default=5,
    help="Number of files to tokenize from the `val_5M` folder.",
)
parser.add_argument(
    "--n_files_test",
    type=int,
    required=False,
    default=20,
    help="Number of files to tokenize from the `test_20M` folder.",
)
parser.add_argument(
    "--output_suffix", type=str, default="", help="Suffix to add to the output folder name."
)
parser.add_argument(
    "--dry_run",
    action="store_true",
    default=False,
    help="Print the arguments and exit without running the tokenization.",
)
parser.add_argument(
    "--save_token_id_only",
    action="store_true",
    default=False,
    help="If this flag is used, only save the token IDs and not the features. "
    "This is mostly there for backwards compatibility with the old tokenization script.",
)
parser.add_argument(
    "--output_base_dir",
    required=True,
    type=str,
    help="Base directory to save the tokenized files.",
)
parser.add_argument(
    "--input_base_dir",
    required=True,
    type=str,
    help="Base directory to read the JetClass files from.",
)
parser.add_argument(
    "--dataset_type",
    type=str,
    default="jetclass",
    help="Type of dataset to tokenize. Options: [jetclass, landscape]",
)
parser.add_argument(
    "--shuffle_particles",
    action="store_true",
    default=False,
    help="Shuffle the particles before tokenization",
)


def main(
    jet_types,
    n_files_train,
    n_files_val,
    n_files_test,
    output_suffix="",
    output_base_dir: str = None,
    input_base_dir: str = None,
    ckpt_path: str = None,
    binning_cfg_path: str = None,
    save_token_id_only: bool = False,
    dataset_type: str = "jetclass",
    shuffle_particles: bool = False,
):
    """Tokenize the JetClass files and save them to the tokenized dir.

    Parameters
    ----------
    jet_types : list of str
        List of jet types to use for tokenization.
    n_files_train : int
        Number of files to use for training.
    n_files_val : int
        Number of files to use for validation.
    n_files_test : int
        Number of files to use for testing.
    output_suffix : str
        Suffix to add to the output folder name.
    output_base_dir : str
        Base directory to save the tokenized files.
    input_base_dir : str
        Base directory to read the JetClass files from.
    ckpt_path : str
        Path to the checkpoint to use for tokenization.
    binning_cfg_path : str
        Path to the binning config to use for tokenization.
    save_token_id_only : bool
        If True, only save the token IDs and not the features (mostly there for
        backwards compatibility with the old tokenization script).
    """

    log.info("Starting tokenization of JetClass files")

    allowed_dataset_types = ["jetclass", "landscape"]
    if dataset_type not in allowed_dataset_types:
        raise ValueError(
            f"dataset_type must be one of {allowed_dataset_types}, but got {dataset_type}"
        )

    if ckpt_path == "None" or ckpt_path == "null":
        ckpt_path = None
    if binning_cfg_path == "None" or binning_cfg_path == "null":
        binning_cfg_path = None

    if ckpt_path is None and binning_cfg_path is None:
        raise ValueError("Either ckpt_path or binning_cfg_path must be provided")
    elif ckpt_path is not None and binning_cfg_path is not None:
        raise ValueError("Only one of ckpt_path or binning_cfg_path can be provided")
    elif binning_cfg_path is not None:
        log.info(f"Using binning config: {binning_cfg_path}")
    elif ckpt_path is not None:
        log.info(f"Using checkpoint: {ckpt_path}")

    JETCLASS_DIR = input_base_dir
    JETCLASS_DIR_TRAIN = Path(JETCLASS_DIR) / "train_100M"
    JETCLASS_DIR_VAL = Path(JETCLASS_DIR) / "val_5M"
    JETCLASS_DIR_TEST = Path(JETCLASS_DIR) / "test_20M"
    log.info("Will look for train/val/test files in the following folders:")
    log.info(f"Train: {JETCLASS_DIR_TRAIN}")
    log.info(f"Val: {JETCLASS_DIR_VAL}")
    log.info(f"Test: {JETCLASS_DIR_TEST}")

    JETCLASS_DIR_TOKENIZED = output_base_dir
    run_id = Path(ckpt_path).parent.parent.name if ckpt_path is not None else "binning"
    JETCLASS_DIR_TOKENIZED = JETCLASS_DIR_TOKENIZED / (str(run_id) + output_suffix)
    JETCLASS_DIR_TOKENIZED_TRAIN = JETCLASS_DIR_TOKENIZED / "train_100M"
    JETCLASS_DIR_TOKENIZED_VAL = JETCLASS_DIR_TOKENIZED / "val_5M"
    JETCLASS_DIR_TOKENIZED_TEST = JETCLASS_DIR_TOKENIZED / "test_20M"

    # raise error if tokenized dir already exists
    if JETCLASS_DIR_TOKENIZED.exists():
        raise FileExistsError(
            f"Output folder already exists"
            " - please delete it before running this script."
            f"\n\nrm -rf {JETCLASS_DIR_TOKENIZED}\n"
        )

    # create tokenized dirs
    JETCLASS_DIR_TOKENIZED_TRAIN.mkdir(parents=True, exist_ok=True)
    JETCLASS_DIR_TOKENIZED_VAL.mkdir(parents=True, exist_ok=True)
    JETCLASS_DIR_TOKENIZED_TEST.mkdir(parents=True, exist_ok=True)

    files_train = []
    files_val = []
    files_test = []

    file_type = "root"

    for jt in jet_types:
        wildcard_train = f"{JETCLASS_DIR_TRAIN}/{jt}*.{file_type}"
        wildcard_val = f"{JETCLASS_DIR_VAL}/{jt}*.{file_type}"
        wildcard_test = f"{JETCLASS_DIR_TEST}/{jt}*.{file_type}"

        files_train.extend(sorted(list(glob.glob(wildcard_train)))[:n_files_train])
        files_val.extend(sorted(list(glob.glob(wildcard_val)))[:n_files_val])
        files_test.extend(sorted(list(glob.glob(wildcard_test)))[:n_files_test])

    log.info(f"Found {len(files_train)} train files:")
    for f in files_train:
        log.info(f)
    log.info(f"Found {len(files_val)} val files:")
    for f in files_val:
        log.info(f)
    log.info(f"Found {len(files_test)} test files:")
    for f in files_test:
        log.info(f)

    files_dict = {
        "train": files_train,
        "val": files_val,
        "test": files_test,
    }
    out_dirs = {
        "train": JETCLASS_DIR_TOKENIZED_TRAIN,
        "val": JETCLASS_DIR_TOKENIZED_VAL,
        "test": JETCLASS_DIR_TOKENIZED_TEST,
    }

    if binning_cfg_path is not None:
        # initialize the tokenizer and save it there
        os.system(f"cp {binning_cfg_path} {JETCLASS_DIR_TOKENIZED}/config.yaml")  # nosec
        # load the config
        cfg = OmegaConf.load(binning_cfg_path)
        binning_dict = {
            key: np.linspace(*cfg.particle_features[key]["binning"])
            for key in cfg.particle_features
        }
        tokenizer = BinningTokenizer(bin_edges_dict=binning_dict)
        # save the tokenizer
        model_ckpt_path = JETCLASS_DIR_TOKENIZED / "model_ckpt.ckpt"
        torch.save(tokenizer, model_ckpt_path)  # nosec
    else:
        # copy the checkpoint to the tokenized dir
        os.system(f"cp {ckpt_path} {JETCLASS_DIR_TOKENIZED}")  # nosec
        os.system(f"cp {ckpt_path} {JETCLASS_DIR_TOKENIZED}/model_ckpt.ckpt")  # nosec
        cfg_path = Path(ckpt_path).parent.parent / "config.yaml"
        os.system(f"cp {cfg_path} {JETCLASS_DIR_TOKENIZED}")  # nosec

    for stage, files in files_dict.items():
        for i, filename_in in enumerate(files):
            log.info(f"{stage} file {i + 1}/{len(files)}")
            filename_out = Path(out_dirs[stage]) / Path(filename_in).name.replace(
                f".{file_type}", "_tokenized.parquet"
            )
            log.info("Input file: %s", filename_in)
            log.info("Output file: %s", filename_out)
            log.info("---")
            tokenize_jetclass_file(
                filename_in=filename_in,
                filename_out=filename_out,
                add_start_end_tokens=True,
                model_ckpt_path=ckpt_path,
                binning_cfg=binning_cfg_path,
                save_token_id_only=save_token_id_only,
                dataset_type=dataset_type,
                shuffle_particles=shuffle_particles,
            )

    log.info("Finished tokenization of JetClass files")
    log.info(f"Tokenized files saved to {JETCLASS_DIR_TOKENIZED}")


if __name__ == "__main__":
    args = parser.parse_args()
    ckpt_path = args.ckpt_path
    jet_types = args.jet_types.split(",")
    n_files_train = args.n_files_train
    n_files_val = args.n_files_val
    n_files_test = args.n_files_test
    output_suffix = args.output_suffix
    save_token_id_only = args.save_token_id_only
    output_base_dir = args.output_base_dir
    input_base_dir = args.input_base_dir
    if args.dry_run:
        log.info("Dry run - not actually running tokenization")
        log.info(f"Using checkpoint: {ckpt_path}")
        log.info(f"Input base dir: {input_base_dir}")
        log.info(f"Output base dir: {output_base_dir}")
        log.info(f"Jet types: {jet_types}")
        log.info(f"Output suffix: {output_suffix}")
        log.info(f"n_files_train: {n_files_train}")
        log.info(f"n_files_val: {n_files_val}")
        log.info(f"n_files_test: {n_files_test}")
        exit(0)
    main(
        jet_types,
        n_files_train,
        n_files_val,
        n_files_test,
        output_suffix,
        ckpt_path=ckpt_path,
        binning_cfg_path=args.binning_cfg_path,
        save_token_id_only=save_token_id_only,
        output_base_dir=Path(output_base_dir),
        input_base_dir=Path(input_base_dir),
        dataset_type=args.dataset_type,
        shuffle_particles=args.shuffle_particles,
    )
