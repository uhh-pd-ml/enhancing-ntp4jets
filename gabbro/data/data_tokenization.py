from pathlib import Path

import awkward as ak
import numpy as np
import torch
import vector
from omegaconf import OmegaConf

from gabbro.data.loading import (
    read_cms_open_data_file,
    read_jetclass_file,
)
from gabbro.models.vqvae import VQVAELightning
from gabbro.utils.arrays import ak_select_and_preprocess
from gabbro.utils.jet_types import get_jet_type_from_file_prefix, jet_types_dict
from gabbro.utils.pylogger import get_pylogger

vector.register_awkward()

logger = get_pylogger(__name__)


def tokenize_jetclass_file(
    filename_in: str,
    filename_out: str = None,
    add_start_end_tokens: bool = False,
    print_model: bool = False,
    model_ckpt_path: str = None,
    save_token_id_only: bool = False,
    dataset_type: str = "jetclass",
    shuffle_particles: bool = False,
):
    """Tokenize a single file using a trained VQ-VAE model.

    Parameters
    ----------
    filename_in : str
        Path to the file to be tokenized.
    filename_out : str, optional
        Path to the output file.
    add_start_end_tokens : bool, optional
        Whether to add start and end tokens to the tokenized sequence.
    print_model : bool, optional
        Whether to print the model architecture.
    model_ckpt_path : str
        Path to the model checkpoint if using a VQ-VAE.
    save_token_id_only : bool, optional
        Whether to save only the token ids.
    dataset_type : str, optional
        The type of dataset. Options are "jetclass", "landscape", "cms_open_data". Default is "jetclass".
    shuffle_particles : bool, optional
        Whether to shuffle the particles. Default is False.
    """

    # input verification
    if model_ckpt_path is not None:
        logger.info(f"Using model checkpoint: {model_ckpt_path}")
    else:
        raise ValueError("Currently only model-based tokenization is supported.")

    # Validate dataset_type
    valid_dataset_types = ["jetclass", "landscape", "cms_open_data"]
    if dataset_type not in valid_dataset_types:
        raise ValueError(f"dataset_type must be one of {valid_dataset_types}, got {dataset_type}")

    if shuffle_particles and dataset_type == "cms_open_data":
        raise ValueError("Shuffling particles is only supported for JetClass files.")

    # TODO: currently all jet and particle features are hardcoded here. This should be
    #       changed to be more flexible and to be able to handle different datasets
    pp_dict_spectators_part = {
        "part_px": None,
        "part_py": None,
        "part_pz": None,
        "part_energy": None,
        "part_pt": None,
        "part_eta": None,
        "part_phi": None,
        "part_mass": None,
        "part_ptrel": None,
        "part_erel": None,
        "part_etarel": None,
        "part_phirel": None,
        "part_deltaR": None,
        "part_deta": None,
        "part_dphi": None,
    }
    if dataset_type == "jetclass":
        # only available in jetclass
        pp_dict_spectators_part.update(
            {
                "part_d0val": None,
                "part_d0err": None,
                "part_dzval": None,
                "part_dzerr": None,
                "part_charge": None,
                "part_isChargedHadron": None,
                "part_isNeutralHadron": None,
                "part_isPhoton": None,
                "part_isElectron": None,
                "part_isMuon": None,
            }
        )
    pp_dict_spectators_jet = {
        "jet_pt": None,
        "jet_eta": None,
        "jet_phi": None,
        "jet_energy": None,
        "jet_nparticles": None,
        "jet_mass_from_p4s": None,
        "jet_pt_from_p4s": None,
        "jet_eta_from_p4s": None,
        "jet_phi_from_p4s": None,
    }
    if dataset_type == "jetclass":
        # only available in jetclass
        pp_dict_spectators_jet.update(
            {
                "jet_sdmass": None,
                "jet_tau1": None,
                "jet_tau2": None,
                "jet_tau3": None,
                "jet_tau4": None,
            }
        )
    # --------------------------------
    # --- Model and config loading ---
    ckpt_path = Path(model_ckpt_path)
    config_path = ckpt_path.parent.parent / "config.yaml"
    cfg = OmegaConf.load(config_path)
    logger.info(f"Loaded config from {config_path}")
    tokenizer = VQVAELightning.load_from_checkpoint(ckpt_path)
    if print_model:
        print(tokenizer)
    pp_dict = OmegaConf.to_container(cfg.data.dataset_kwargs_common["feature_dict"])
    pp_dict_jet = (
        cfg.data.dataset_kwargs_common["feature_dict_jet"]
        if "feature_dict_jet" in cfg.data.dataset_kwargs_common
        else {}
    )
    pp_dict_jet = pp_dict_jet if pp_dict_jet is not None else {}
    if pp_dict_jet is not None and not isinstance(pp_dict_jet, dict):
        OmegaConf.to_container(cfg.data.dataset_kwargs_common["feature_dict_jet"])

    tokenizer = tokenizer.to("cuda")
    tokenizer.eval()

    # update the preprocessing dictionary with the spectators. if a variable
    # is both in the spectators and the particle/jet features, the one in the
    # particle/jet features will be used
    pp_dict_with_spectators = {**pp_dict_spectators_part, **pp_dict}
    pp_dict_with_spectators_jet = {**pp_dict_spectators_jet, **pp_dict_jet}

    # only leave preprocessing parameters that lead to selections, the others
    # we don't want to include to keep the exact same values as in the original file
    # (e.g. transforming pt to log(pt) and then back to pt can lead to numerical
    # inaccuracies)
    for key in pp_dict_with_spectators:
        if pp_dict_with_spectators[key] is None:
            continue
        pp_dict_with_spectators[key] = {
            "smaller_than": pp_dict_with_spectators[key].get("smaller_than", None),
            "larger_than": pp_dict_with_spectators[key].get("larger_than", None),
        }
    for key in pp_dict_with_spectators_jet:
        if pp_dict_with_spectators_jet[key] is None:
            continue
        pp_dict_with_spectators_jet[key] = {
            "smaller_than": pp_dict_with_spectators_jet[key].get("smaller_than", None),
            "larger_than": pp_dict_with_spectators_jet[key].get("larger_than", None),
        }

    logger.info("Preprocessing dictionary:")
    logger.info("Particle features:")
    for key, value in pp_dict_with_spectators.items():
        logger.info(f" | {key}: {value}")
    logger.info("Jet features:")
    if pp_dict_jet is not None:
        for key, value in pp_dict_with_spectators_jet.items():
            logger.info(f" | {key}: {value}")

    if dataset_type == "cms_open_data":
        # check if the file is empty
        pp_dict_with_spectators = {
            "part_pt": None,
            "part_etarel": None,
            "part_phirel": None,
        }
        pp_dict_jet = None
        x_ak, _, _ = read_cms_open_data_file(
            filename_in,
            particle_features=list(pp_dict_with_spectators.keys()),
            jet_features=None,
        )
    else:
        x_ak, x_ak_jet, _ = read_jetclass_file(
            filename_in,
            particle_features=pp_dict_with_spectators.keys(),
            jet_features=pp_dict_with_spectators_jet.keys()
            if pp_dict_with_spectators_jet is not None
            else None,
            return_p4=False,
            shuffle_particles=shuffle_particles,
        )

    # apply preprocessing and invert it, such that cuts are applied correctly
    # also to the original features
    x_ak = ak_select_and_preprocess(ak_array=x_ak, pp_dict=pp_dict_with_spectators)
    x_ak = ak_select_and_preprocess(ak_array=x_ak, pp_dict=pp_dict_with_spectators, inverse=True)

    # pad the sequence to the desired length
    if model_ckpt_path is not None:
        pad_length = cfg.data.dataset_kwargs_common["pad_length"]
    else:
        pad_length = cfg.general.pad_length

    x_ak = x_ak[:, :pad_length]

    tokenization_output = tokenizer.tokenize_ak_array(
        ak_arr=x_ak,
        pp_dict=pp_dict,
        ak_arr_jet=x_ak_jet if pp_dict_jet is not None else None,
        pp_dict_jet=pp_dict_jet,
        pad_length=pad_length,
        hide_pbar=False,
    )
    # print the fields and the nested fields
    logger.info("The tokenization output has the following fields:")
    for field in tokenization_output.fields:
        if len(tokenization_output[field].fields) > 0:
            logger.info(f" | {field}")
            for inner_field in tokenization_output[field].fields:
                logger.info(f" |   | {inner_field}")
    # new tokenization output has the following fields:
    #  - part_token_id
    #    | part_token_id
    #  - z_q
    #    | z_q_0
    #    | z_q_1
    #    | ...
    # old tokenization output was an awkward array without fields, just the token ids
    token_ids = (
        tokenization_output
        if len(tokenization_output.fields) == 0
        else tokenization_output["part_token_id"]
    )
    # reconstruct the features
    features_tokenized = tokenizer.reconstruct_ak_tokens(
        token_ids,
        pp_dict,
        jets_ak=x_ak_jet if pp_dict_jet is not None else None,
        pp_dict_jet=pp_dict_jet,
        hide_pbar=False,
    )

    if add_start_end_tokens:
        n_tokens = tokenizer.model.vqlayer.num_codes
        token_ids_plus_one = (
            token_ids + 1
            if len(token_ids.fields) == 0
            else (ak.Array({key: token_ids[key] + 1 for key in token_ids.fields}))
        )
        start_token = (
            ak.zeros_like(token_ids[:, :1])  # start token is 1
            if len(token_ids.fields) == 0
            else ak.Array({key: ak.zeros_like(token_ids[key][:, :1]) for key in token_ids.fields})
        )
        stop_token = (
            ak.ones_like(token_ids[:, :1]) + n_tokens  # end token is n_tokens + 2
            if len(token_ids.fields) == 0
            else ak.Array(
                {key: ak.ones_like(token_ids[key][:, :1]) + n_tokens for key in token_ids.fields}
            )
        )
        token_ids = ak.concatenate(
            [
                start_token,
                token_ids_plus_one,
                stop_token,
            ],
            axis=1,
        )

    tokens_int = ak.values_astype(token_ids, int)

    if filename_out is not None:
        logger.info(f"Saving tokenized file to {filename_out}")
        if save_token_id_only:
            ak.to_parquet(tokens_int, filename_out)
        else:
            logger.info("Saving additional features (continuous jet/particle features)")
            ak_arr_to_save = ak.Array(
                {
                    "part_token_id": tokens_int,
                    "particle_features_tokenized": features_tokenized,
                    "particle_features": x_ak,
                }
                | ({"jet_features": x_ak_jet} if x_ak_jet is not None else {})
            )
            logger.info("The following fields are saved:")
            for outer_key in ak_arr_to_save.fields:
                logger.info(f" | {outer_key}: ")
                for inner_key in ak_arr_to_save[outer_key].fields:
                    logger.info(
                        f" |   | {inner_key}: {ak_arr_to_save[outer_key][inner_key].fields}"
                    )
            ak.to_parquet(ak_arr_to_save, filename_out)
            logger.info("File saved.")


def reconstruct_jetclass_file(
    filename_in: str,
    model_ckpt_path: str,
    config_path: str,
    filename_out: str = None,
    start_token_included: bool = False,
    end_token_included: bool = False,
    shift_tokens_by_minus_one: bool = False,
    print_model: bool = False,
    device: str = "cuda",
    return_labels: bool = False,
    tokenizer_type: str = "vqvae",
    pad_length: int = 128,
):
    """Reconstruct a single file using a trained model and the tokenized file.

    Parameters
    ----------
    filename_in : str
        Path to the file to be tokenized.
    model_ckpt_path : str
        Path to the model checkpoint.
    config_path : str
        Path to the config file.
    filename_out : str, optional
        Path to the output file.
    start_token_included : bool, optional
        Whether the start token is included in the tokenized sequence.
    end_token_included : bool, optional
        Whether the end token is included in the tokenized sequence.
    shift_tokens_by_minus_one : bool, optional
        Whether to shift the tokens by -1.
    print_model : bool, optional
        Whether to print the model architecture.
    device : str, optional
        Device to use for the model.
    return_labels : bool, optional
        Whether to return the labels of the jet type. By default, the labels are not returned.
    tokenizer_type : str, optional
        Type of the tokenizer used. Options: ["vqvae"]

    Returns
    -------
    p4s_reco : ak.Array
        Momentum4D array of the reconstructed particles.
    x_reco_ak : ak.Array
        Array of the reconstructed particles.
    labels_onehot : np.ndarray
        One-hot encoded labels of the jet type. Only returned if return_labels is True.
    """

    # --- Model and config loading ---
    if tokenizer_type == "vqvae":
        ckpt_path = Path(model_ckpt_path)
        cfg = OmegaConf.load(config_path)
        logger.info(f"Loaded config from {config_path}")
        model = VQVAELightning.load_from_checkpoint(ckpt_path)
        if print_model:
            print(model)
        pp_dict = OmegaConf.to_container(cfg.data.dataset_kwargs_common["feature_dict"])
        pp_dict_jet = (
            cfg.data.dataset_kwargs_common["feature_dict_jet"]
            if "feature_dict_jet" in cfg.data.dataset_kwargs_common
            else {}
        )
        if pp_dict_jet is not None and not isinstance(pp_dict_jet, dict):
            OmegaConf.to_container(cfg.data.dataset_kwargs_common["feature_dict_jet"])
        # logger.info("Preprocessing dictionary:")
        # for key, value in pp_dict.items():
        #     logger.info(f" | {key}: {value}")
    else:
        supported_tokenizer_types = ["vqvae"]
        raise ValueError(
            f"Invalid tokenizer type. Supported types are: {supported_tokenizer_types}"
        )

    model = model.to(device)
    model.eval()
    # --------------------------------

    logger.info(f"Reading file {filename_in}")
    tokenized_file_ak_arr = ak.from_parquet(filename_in)
    # log the fields and the nested fields
    #  - part_token_id
    #    | part_token_id
    logger.info("The file has the following fields:")
    for field in tokenized_file_ak_arr.fields:
        logger.info(f" | {field}")
        if hasattr(tokenized_file_ak_arr[field], "fields"):
            for inner_field in tokenized_file_ak_arr[field].fields:
                logger.info(f" |   | {inner_field}")

    if len(tokenized_file_ak_arr.fields) > 0:
        logger.info("Loading tokens from the field 'part_token_id'")
        tokens = tokenized_file_ak_arr["part_token_id"]
        logger.info(f"Fields in tokens: {tokens.fields}")
        if len(tokens.fields) > 1:
            raise ValueError(
                "There should be only one field in 'part_token_id' for reconstruction."
            )
        else:
            if len(tokens.fields) == 1:
                tokens = tokens["part_token_id"]

    else:
        tokens = tokenized_file_ak_arr
        logger.warning(
            f"File {filename_in} is in the old format. At the moment this is still supported, "
            "but might lead to problems in the future."
        )

    jet_features = (
        tokenized_file_ak_arr["jet_features"]
        if "jet_features" in tokenized_file_ak_arr.fields
        else None
    )

    if end_token_included:
        logger.info("Removing end token")
        tokens = tokens[:, :-1]
    if start_token_included:
        logger.info("Removing start token")
        tokens = tokens[:, 1:]
    if shift_tokens_by_minus_one:
        logger.info("Shifting tokens by -1")
        tokens = tokens - 1

    logger.info(f"Smallest token in file: {ak.min(tokens)}")
    logger.info(f"Largest token in file:  {ak.max(tokens)}")

    x_reco_ak = model.reconstruct_ak_tokens(
        tokens,
        pp_dict,
        jets_ak=jet_features,
        pp_dict_jet=pp_dict_jet,
        hide_pbar=True,
        pad_length=pad_length,
    )

    # check if particle mass is available
    if "part_mass" in x_reco_ak.fields:
        particle_mass_field_name = "part_mass"
    elif "mass" in x_reco_ak.fields:
        particle_mass_field_name = "mass"
    else:
        particle_mass_field_name = None

    p4s_reco = ak.zip(
        {
            "pt": x_reco_ak.pt if "pt" in x_reco_ak.fields else x_reco_ak.part_pt,
            "eta": x_reco_ak.etarel if "etarel" in x_reco_ak.fields else x_reco_ak.part_etarel,
            "phi": x_reco_ak.phirel if "phirel" in x_reco_ak.fields else x_reco_ak.part_phirel,
            "mass": getattr(x_reco_ak, particle_mass_field_name)
            if particle_mass_field_name is not None
            else ak.zeros_like(x_reco_ak.pt if "pt" in x_reco_ak.fields else x_reco_ak.part_pt),
        },
        with_name="Momentum4D",
    )
    if return_labels:
        # extract jet type from filename and create the corresponding labels
        jet_type_prefix = filename_in.split("/")[-1].split("_")[0] + "_"
        jet_type_name = get_jet_type_from_file_prefix(jet_type_prefix)

        # one-hot encode the jet type
        labels_onehot = ak.Array(
            {
                f"label_{jet_type}": np.ones(len(x_reco_ak)) * (jet_type_name == jet_type)
                for jet_type in jet_types_dict
            }
        )

        return p4s_reco, x_reco_ak, labels_onehot

    return p4s_reco, x_reco_ak
