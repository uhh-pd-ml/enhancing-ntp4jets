from pathlib import Path

import awkward as ak
import fastjet as fj
import h5py
import numpy as np
import uproot
import vector
from tqdm import tqdm

from gabbro.utils.arrays import (
    ak_add_zero_padded_features,
    ak_select_and_preprocess,
    calc_additional_kinematic_features,
    combine_ak_arrays,
    np_to_ak,
    shuffle_ak_arr_along_axis1,
)
from gabbro.utils.jet_types import get_jet_type_from_file_prefix, jet_types_dict
from gabbro.utils.pylogger import get_pylogger
from gabbro.utils.utils import print_field_structure_of_ak_array

logger = get_pylogger(__name__)

vector.register_awkward()


def read_cms_open_data_file(
    filepath: str,
    particle_features: list = None,
    jet_features: list = None,
    return_p4: bool = False,
    labels=[
        "label_QCD",
        "label_Hbb",
        "label_Hcc",
        "label_Hgg",
        "label_H4q",
        "label_Hqql",
        "label_Zqq",
        "label_Wqq",
        "label_Tbqq",
        "label_Tbl",
    ],
):
    """Reads a single file from the CMS Open Data dataset.

    Parameters:
    -----------
    filepath : str
        Path to the h5 file.
    particle_features : list of str, optional
        List of particle-level features to load.
    jet_features : list of str, optional
        List of jet-level features to load.
    return_p4 : bool, optional
        Whether to return the 4-momentum of the particles.
    labels : list of str, optional
        List of truth labels to load. This is just there for compatibility with the
        JetClass dataset. We treat the CMS Open Data dataset as QCD jets.

    Returns:
    --------
    ak.Array
        An awkward array of the particle-level features or jet features if only one
        of the two is requested.
        If both are requested, a tuple of two awkward arrays is returned, the first
        one containing the particle-level features and the second one the jet-level.
    """
    # Selected all AK8 jets with pt > 300, abs(eta) < 2.5, passing standard jet ID criteria
    # Events are stored in h5 format with 4 keys
    # 'event_info' : [Run Number, LumiBlock, Event Number]
    # 'jet_kinematics' : [ pt, eta, phi, softdrop mass] of the AK8 jet
    # 'PFCands' : Zero padded list of up to 150 PFcandidates inside the AK8 jet. For each entry, shape is [150, 10].
    # Info for each candidate is [px, py, pz, E, d0, d0Err, dz, dzErr, charge, pdgId, PUPPI weight]
    # 'jet_tagging': Tagging info/scores for the AK8 jet, 13 entries
    # [nConstituents, tau1, tau2, tau3, tau4, PNet H4q vs QCD, PNet Hbb vs QCD, PNet Hcc vs QCD, PNet QCD score, PNet T vs QCD, PNet W vs QCD, PNet Z vs QCD, PNet regressed mass]
    if particle_features is None and jet_features is None:
        raise ValueError("Either particle_features or jet_features must be provided.")

    with h5py.File(filepath, "r") as f1:
        # Convert to numpy
        PFCands = f1["PFCands"][:]
        # event_info = f1["event_info"][:]
        jet_kinematics = f1["jet_kinematics"][:]
        # jet_tagging = f1["jet_tagging"][:]

    p4s_ak = np_to_ak(
        x=PFCands[:, :, :4],
        names=["px", "py", "pz", "E"],
        mask=PFCands[:, :, 3] != 0,
    )

    # Create an awkward vector from the PFCands
    p4 = ak.zip(
        {"px": p4s_ak.px, "py": p4s_ak.py, "pz": p4s_ak.pz, "E": p4s_ak.E},
        with_name="Momentum4D",
    )

    p4_jet = ak.sum(p4, axis=1)

    x_ak_particles = ak.Array(
        {
            "part_pt": p4.pt,
            "part_eta": p4.eta,
            "part_phi": p4.phi,
            "part_etarel": p4.deltaeta(p4_jet),
            "part_phirel": p4.deltaphi(p4_jet),
        }
    )

    x_ak_jets = ak.Array(
        {
            "jet_pt": jet_kinematics[:, 0],
            "jet_eta": jet_kinematics[:, 1],
            "jet_phi": jet_kinematics[:, 2],
            "jet_sdmass": jet_kinematics[:, 3],
            "jet_mass": p4_jet.mass,
        }
    )

    if particle_features is None:
        x_ak_particles = None
    else:
        x_ak_particles = x_ak_particles[particle_features]
    if jet_features is None:
        x_ak_jets = None
    else:
        x_ak_jets = x_ak_jets[jet_features]

    len_p4_jet = len(p4_jet)

    ak_labels = ak.Array(
        {
            "label_QCD": np.ones(len_p4_jet),
            "label_Hbb": np.zeros(len_p4_jet),
            "label_Hcc": np.zeros(len_p4_jet),
            "label_Hgg": np.zeros(len_p4_jet),
            "label_H4q": np.zeros(len_p4_jet),
            "label_Hqql": np.zeros(len_p4_jet),
            "label_Zqq": np.zeros(len_p4_jet),
            "label_Wqq": np.zeros(len_p4_jet),
            "label_Tbqq": np.zeros(len_p4_jet),
            "label_Tbl": np.zeros(len_p4_jet),
        }
    )

    if return_p4:
        return x_ak_particles, x_ak_jets, p4, ak_labels[labels]

    return x_ak_particles, x_ak_jets, ak_labels[labels]


def read_tokenized_jetclass_file(
    filepath,
    load_token_ids: bool = True,
    particle_features: list = None,
    particle_features_tokenized: list = None,
    labels: list = None,
    remove_start_token: bool = False,
    remove_end_token: bool = False,
    shift_tokens_minus_one: bool = False,
    add_padded_particle_features_start: bool = False,
    add_padded_particle_features_end: bool = False,
    n_load: int = None,
    random_seed: int = None,
    jet_features: list = None,
    verbose: bool = False,
):
    """Reads a file that contains the tokenized JetClass jets.

    Parameters
    ----------
    filepath : str
        Path to the file.
    load_token_ids : bool, optional
        Whether to load the token-ids.
    particle_features : List[str], optional
        A list of particle-level features to be loaded. These are the full-resolution
        ones.
    particle_features_tokenized : List[str], optional
        A list of particle-level features to be loaded. These are the tokenized ones.
    labels : List[str], optional
        A list of truth labels to be loaded. If None, the default JetClass labels are
        used.
    remove_start_token : bool, optional
        Whether to remove the start token from the tokenized sequence.
    remove_end_token : bool, optional
        Whether to remove the end token from the tokenized sequence.
    shift_tokens_minus_one : bool, optional
        Whether to shift the token values by -1.
    add_padded_particle_features_start : bool, optional
        Whether to add zero-padded padded particle features at the start of the sequence.
        Default is False.
    add_padded_particle_features_end : bool, optional
        Whether to add zero-padded padded particle features at the end of the sequence.
        Default is False.
    n_load : int, optional
        Number of events to load. If None, all events are loaded.
    random_seed : int, optional
        Random seed for shuffling the data. If None, no shuffling is performed.
    jet_features : List[str], optional
        A list of jet-level features to be loaded.
        Possible options are:
        - jet_pt
        - jet_eta
        - jet_phi
        - jet_energy
        - jet_nparticles
        - jet_sdmass
        - jet_tau1
        - jet_tau2
        - jet_tau3
        - jet_tau4

    Returns
    -------
    x_ak
        An awkward array of the particle-level features. These are the merged
        arrays of the full-resolution and tokenized features (and particle tokens).
    y : ak.Array
        An awkward array of the truth labels (one-hot encoded).
    x_jets : ak.Array
        An awkward array of the jet-level features.
    """

    if labels is None:
        labels = [
            "label_QCD",
            "label_Hbb",
            "label_Hcc",
            "label_Hgg",
            "label_H4q",
            "label_Hqql",
            "label_Zqq",
            "label_Wqq",
            "label_Tbqq",
            "label_Tbl",
        ]

    # check if it's the old or new file format:
    # old: tokenized files are a awkward.highlevel.Array objects that has the token-ids
    #      without any keys or so --> array.fields is empty
    # new: tokenized files are a awkward.highlevel.Array objects that has the token-ids
    #      as a key "part_token_id"

    # if type of `particle_features` is dict, convert it to a list with the keys
    if particle_features is not None and particle_features is not isinstance(
        particle_features, list
    ):
        particle_features = [feat for feat in particle_features]
    if particle_features_tokenized is not None and not isinstance(
        particle_features_tokenized, list
    ):
        particle_features_tokenized = [feat for feat in particle_features_tokenized]

    tokenized_file = ak.from_parquet(filepath)
    if jet_features is not None:
        try:
            ak_jets = tokenized_file["jet_features"]
        except AttributeError:
            print("No jet features found in the file!")
            print(
                "Please make sure that the file contains jet features or run without setting feature_dict_jet."
            )
    x_jets = ak_jets[jet_features] if jet_features is not None else None
    # check if the file is in the new format, and if so, extract the token-ids and
    # overwrite ak_tokens ak array
    if verbose:
        logger.info("The loaded file has the following structure:")
        print_field_structure_of_ak_array(tokenized_file)
    if len(tokenized_file.fields) > 0:
        ak_tokens = tokenized_file["part_token_id"]
    else:
        ak_tokens = tokenized_file
        logger.warning(
            f"File {filepath} is in the old format. At the moment this is still supported, "
            "but might lead to problems in the future."
        )

    if n_load is not None:
        ak_tokens = ak_tokens[:n_load]
        x_jets = x_jets[:n_load] if jet_features is not None else None

    # extract jet type from filename and create the corresponding labels
    jet_type_prefix = filepath.split("/")[-1].split("_")[0] + "_"
    jet_type_name = get_jet_type_from_file_prefix(jet_type_prefix)

    # one-hot encode the jet type
    labels_onehot = ak.Array(
        {
            f"label_{jet_type}": np.ones(len(ak_tokens)) * (jet_type_name == jet_type)
            for jet_type in jet_types_dict
        }
    )

    if len(ak_tokens.fields) != 0:
        # group tokenization (new setup)
        stop_token_value_dict = {  # noqa: F841
            field: ak_tokens[field][0, -1] for field in ak_tokens.fields
        }
    if remove_start_token:
        ak_tokens = ak_tokens[:, 1:]
    if remove_end_token:
        ak_tokens = ak_tokens[:, :-1]
    if shift_tokens_minus_one:
        # if the tokens are in a nested structure, we need to shift all of them
        # by -1 separately, otherwise we can just shift the whole array
        ak_tokens = (
            ak_tokens - 1
            if len(ak_tokens.fields) == 0
            else ak.Array({field: ak_tokens[field] - 1 for field in ak_tokens.fields})
        )

    if load_token_ids:
        if len(ak_tokens.fields) == 0:
            # non-group tokenization (initial setup)
            x_ak_tokens = ak.Array(
                {
                    "part_token_id": ak_tokens,
                    "part_token_id_duplicated": ak_tokens,
                    "part_token_id_without_last": ak_tokens[:, :-1],
                    "part_token_id_without_first": ak_tokens[:, 1:],
                }
            )
        else:
            # group tokenization (new setup)
            dict_for_x_ak_tokens = {}
            for field in ak_tokens.fields:
                ak_tokens_last_with_stop_overwritten = ak.concatenate(
                    [
                        ak_tokens[field][:, :-1],
                        ak.ones_like(ak_tokens[field][:, -1:]) * stop_token_value_dict[field],
                    ],
                    axis=1,
                )
                ak_tokens_with_last_two_particles_with_stop_overwritten = ak.concatenate(
                    [
                        ak_tokens[field][:, :-2],
                        ak.ones_like(ak_tokens[field][:, -2:]) * stop_token_value_dict[field],
                    ],
                    axis=1,
                )
                dict_for_x_ak_tokens[field] = ak_tokens[field]
                dict_for_x_ak_tokens[f"{field}_last_with_stop_overwritten"] = (
                    ak_tokens_last_with_stop_overwritten
                )
                dict_for_x_ak_tokens[f"{field}_last_two_with_stop_overwritten"] = (
                    ak_tokens_with_last_two_particles_with_stop_overwritten
                )
                dict_for_x_ak_tokens[f"{field}_without_last"] = ak_tokens[field][:, :-1]
                dict_for_x_ak_tokens[f"{field}_without_first"] = ak_tokens[field][:, 1:]
                # duplicate of default token ids
            dict_for_x_ak_tokens["part_token_id_duplicated"] = ak_tokens["part_token_id"]
            x_ak_tokens = ak.Array(dict_for_x_ak_tokens)
        logger.info(f"Available fields in x_ak_tokens: {x_ak_tokens.fields}")
    else:
        x_ak_tokens = None

    # check if any other features than the token-id are requested
    if particle_features is not None:
        logger.info(
            "Loading the following features from the `particle_features` section of the "
            f"file: {particle_features}"
        )
        # we support both "particle_features" and "features" as key
        key_tmp = (
            "particle_features" if "particle_features" in tokenized_file.fields else "features"
        )
        x_ak_features = tokenized_file[key_tmp][:n_load]
        logger.info("Calculating additional kinematic features.")
        x_ak_features = combine_ak_arrays(
            calc_additional_kinematic_features(
                x_ak_features,
                exclude_fields=x_ak_features.fields,
            ),
            x_ak_features,
        )
        x_ak_features = safe_load_features_from_ak_array(
            ak_array=x_ak_features,
            features=particle_features,
            load_zeros_if_not_present=True,
        )
        if add_padded_particle_features_start or add_padded_particle_features_end:
            x_ak_features = ak_add_zero_padded_features(
                x_ak_features,
                add_start=add_padded_particle_features_start,
                add_end=add_padded_particle_features_end,
            )
    else:
        x_ak_features = None

    if particle_features_tokenized is not None:
        logger.info(
            "Loading the following features from the `particle_features_tokenized` section of "
            f"the file: {particle_features_tokenized}"
        )
        # we support both "particle_features_tokenized" and "features_tokenized" as key
        key_tmp = (
            "particle_features_tokenized"
            if "particle_features_tokenized" in tokenized_file.fields
            else "features_tokenized"
        )
        x_ak_features_tokenized = tokenized_file[key_tmp][:n_load]
        # calculate additional features
        x_ak_features_tokenized = combine_ak_arrays(
            calc_additional_kinematic_features(
                x_ak_features_tokenized,
                exclude_fields=x_ak_features_tokenized.fields,
            ),
            x_ak_features_tokenized,
        )
        x_ak_features_tokenized = safe_load_features_from_ak_array(
            ak_array=x_ak_features_tokenized,
            features=particle_features_tokenized,
            load_zeros_if_not_present=True,
        )
        if add_padded_particle_features_start or add_padded_particle_features_end:
            x_ak_features_tokenized = ak_add_zero_padded_features(
                x_ak_features_tokenized,
                add_start=add_padded_particle_features_start,
                add_end=add_padded_particle_features_end,
            )
    else:
        x_ak_features_tokenized = None

    if jet_features is not None:
        pass
    else:
        x_jets = None

    # apply shuffling
    if random_seed is not None:
        logger.info(f"Shuffling data within one (tokenized) file with random seed {random_seed}.")
        rng = np.random.default_rng(random_seed)
        permutation = rng.permutation(len(x_ak_tokens))
        x_ak_tokens = x_ak_tokens[permutation]
        x_ak_features = x_ak_features[permutation] if particle_features is not None else None
        x_ak_features_tokenized = (
            x_ak_features_tokenized[permutation]
            if particle_features_tokenized is not None
            else None
        )
        x_jets = x_jets[permutation] if jet_features is not None else None
        labels_onehot = labels_onehot[permutation]

    return (
        combine_ak_arrays(x_ak_tokens, x_ak_features, x_ak_features_tokenized),
        labels_onehot[labels],
        x_jets,
    )


def read_jetclass_file(
    filepath,
    particle_features=["part_pt", "part_eta", "part_phi", "part_energy"],
    jet_features=["jet_pt", "jet_eta", "jet_phi", "jet_energy"],
    labels=[
        "label_QCD",
        "label_Hbb",
        "label_Hcc",
        "label_Hgg",
        "label_H4q",
        "label_Hqql",
        "label_Zqq",
        "label_Wqq",
        "label_Tbqq",
        "label_Tbl",
    ],
    return_p4=False,
    n_load=None,
    shuffle_particles=False,
    random_seed: int = None,
):
    """Loads a single file from the JetClass dataset.

    Parameters
    ----------
    filepath : str
        Path to the ROOT data file.
    particle_features : List[str], optional
        A list of particle-level features to be loaded.
        Possible options are:
        - part_px
        - part_py
        - part_pz
        - part_energy
        - part_deta
        - part_dphi
        - part_d0val
        - part_d0err
        - part_dzval
        - part_dzerr
        - part_charge
        - part_isChargedHadron
        - part_isNeutralHadron
        - part_isPhoton
        - part_isElectron
        - part_isMuon

    jet_features : List[str], optional
        A list of jet-level features to be loaded.
        Possible options are:
        - jet_pt
        - jet_eta
        - jet_phi
        - jet_energy
        - jet_nparticles
        - jet_sdmass
        - jet_tau1
        - jet_tau2
        - jet_tau3
        - jet_tau4
        - aux_genpart_eta
        - aux_genpart_phi
        - aux_genpart_pid
        - aux_genpart_pt
        - aux_truth_match

    labels : List[str], optional
        A list of truth labels to be loaded.
        - label_QCD
        - label_Hbb
        - label_Hcc
        - label_Hgg
        - label_H4q
        - label_Hqql
        - label_Zqq
        - label_Wqq
        - label_Tbqq
        - label_Tbl

    return_p4 : bool, optional
        Whether to return the 4-momentum of the particles.
    n_load : int, optional
        Number of jets to load. If None, all jets are loaded.
    shuffle_particles : bool, optional
        If True, the particles are shuffled. Default is False.
    random_seed : int, optional
        Random seed for shuffling the jets. If None, no shuffling is performed.

    Returns
    -------
    x_particles : ak.Array
        An awkward array of the particle-level features.
    x_jets : ak.Array
        An awkward array of the jet-level features.
    y : ak.Array
        An awkward array of the truth labels (one-hot encoded).
    p4 : ak.Array, optional
        An awkward array of the 4-momenta of the particles. Only returned if
        `return_p4` is set to True.
    """

    if n_load is not None:
        table = uproot.open(filepath)["tree"].arrays()[:n_load]
    else:
        table = uproot.open(filepath)["tree"].arrays()

    p4 = vector.zip(
        {
            "px": table["part_px"],
            "py": table["part_py"],
            "pz": table["part_pz"],
            "energy": table["part_energy"],
            # massless particles -> this changes the result slightly,
            # i.e. for example top jets then have a mass of 171.2 instead of 172
            # "mass": ak.zeros_like(table["part_px"]),
        }
    )
    p4_jet = ak.sum(p4, axis=1)

    table["part_pt"] = p4.pt
    table["part_eta"] = p4.eta
    table["part_phi"] = p4.phi
    table["part_mass"] = p4.mass
    table["part_ptrel"] = table["part_pt"] / p4_jet.pt
    table["part_erel"] = table["part_energy"] / p4_jet.energy
    table["part_etarel"] = p4.deltaeta(p4_jet)
    table["part_phirel"] = p4.deltaphi(p4_jet)
    table["part_deltaR"] = p4.deltaR(p4_jet)
    table["jet_mass_from_p4s"] = p4_jet.mass
    table["jet_pt_from_p4s"] = p4_jet.pt
    table["jet_eta_from_p4s"] = p4_jet.eta
    table["jet_phi_from_p4s"] = p4_jet.phi
    table["part_p"] = p4.p
    # Add the energy a second time:
    # workaround to select this feature twice, which is e.g. the case in ParT, where
    # the particle energy is used as log(energy) as standard particle feature
    # and as raw energy in the lorentz vector features
    table["part_energy_raw"] = table["part_energy"]

    p4_centered = ak.zip(
        {
            "pt": p4.pt,
            "eta": p4.deltaeta(p4_jet),
            "phi": p4.deltaphi(p4_jet),
            "mass": p4.mass,
        },
        with_name="Momentum4D",
    )

    table["part_px_after_centering"] = p4_centered.px
    table["part_py_after_centering"] = p4_centered.py
    table["part_pz_after_centering"] = p4_centered.pz
    table["part_energy_after_centering_raw"] = p4_centered.energy

    # check if any of the requested features contains "massless"
    if any("massless" in feature for feature in particle_features):
        p4_massless = ak.zip(
            {
                "pt": p4.pt,
                "eta": p4.eta,
                "phi": p4.phi,
                "mass": ak.zeros_like(table["part_px"]),
            },
            with_name="Momentum4D",
        )
        p4_jet_massless = ak.sum(p4_massless, axis=1)
        # features corresponding to
        table["part_pt_massless"] = p4_massless.pt
        table["part_px_massless"] = p4_massless.px
        table["part_py_massless"] = p4_massless.py
        table["part_pz_massless"] = p4_massless.pz
        table["part_ptrel_massless"] = p4_massless.pt / p4_jet_massless.pt
        table["part_energy_massless"] = p4_massless.energy
        table["part_erel_massless"] = p4_massless.energy / p4_jet_massless.energy
        table["part_etarel_massless"] = p4_massless.deltaeta(p4_jet_massless)
        table["part_phirel_massless"] = p4_massless.deltaphi(p4_jet_massless)
        table["part_deltaR_massless"] = p4_massless.deltaR(p4_jet_massless)
        table["part_energy_massless_raw"] = p4_massless.energy

    # check if any of the requested features contains "mlc" (massless + centered)
    if any(("mlc" in feature or "massless_centered" in feature) for feature in particle_features):
        # refer to this below as "mlc"
        p4_massless_centered = ak.zip(
            {
                "pt": p4_massless.pt,
                "eta": p4_massless.deltaeta(p4_jet_massless),
                "phi": p4_massless.deltaphi(p4_jet_massless),
                "mass": ak.zeros_like(p4_massless.mass),
            },
            with_name="Momentum4D",
        )
        p4_jet_massless_centered = ak.sum(p4_massless_centered, axis=1)
        # features corresponding to using massless + centered particles ("mlc")
        table["part_energy_mlc"] = p4_massless_centered.energy
        table["part_px_mlc"] = p4_massless_centered.px
        table["part_py_mlc"] = p4_massless_centered.py
        table["part_pz_mlc"] = p4_massless_centered.pz
        table["part_energy_mlc"] = p4_massless_centered.energy
        table["part_energy_mlc_raw"] = p4_massless_centered.energy
        table["part_erel_mlc"] = p4_massless_centered.energy / p4_jet_massless_centered.energy
        table["part_ptrel_mlc"] = p4_massless.pt / p4_jet_massless_centered.pt
        table["part_px_massless_centered"] = p4_massless_centered.px
        table["part_py_massless_centered"] = p4_massless_centered.py
        table["part_pz_massless_centered"] = p4_massless_centered.pz
        table["part_energy_massless_centered"] = p4_massless_centered.energy

    x_particles = table[particle_features] if particle_features is not None else None
    if shuffle_particles:
        logger.info("Shuffling particles within each jet.")
        x_particles = shuffle_ak_arr_along_axis1(x_particles, seed=None)
    x_jets = table[jet_features] if jet_features is not None else None
    y = ak.values_astype(table[labels], "int32") if labels is not None else None

    if random_seed is not None:
        logger.info(f"Shuffling data (within one file) with random seed {random_seed}.")
        rng = np.random.default_rng(random_seed)
        permutation = rng.permutation(len(x_particles))
        x_particles = x_particles[permutation] if x_particles is not None else None
        x_jets = x_jets[permutation] if x_jets is not None else None
        y = y[permutation] if y is not None else None
        p4 = p4[permutation] if return_p4 else None

    if return_p4:
        if shuffle_particles:
            raise ValueError("Cannot shuffle particles and return 4-momenta at the same time.")
        return x_particles, x_jets, y, p4

    return x_particles, x_jets, y


def load_landscape_file_and_split_into_qcd_and_top(
    filename: str,
    destination_folder: str,
    number: int,
    convert_to_jetclass_style_root: bool = False,
):
    """Will load the specified file, split it into qcd and tbqq and save them
    in the destination folder.
    The destination folder will get two files: ZJetsToNuNu.parquet and TTbar.parquet.
    This is useful to create the files for the landscape dataset in a JetClass-like format.

    Parameters
    ----------
    filename : str
        The path to the file to load.
    destination_folder : str
        The path to the folder where the files should be saved.
    number : int
        The number to append to the filename.
    convert_to_jetclass_style_root : bool
        If the file should be converted to the JetClass style (i.e. root file with the
        same feature naming conventions).
    """
    ak_arr = ak.from_parquet(filename)
    is_qcd = ak_arr["label"] == 0
    # save the files
    destination_folder = Path(destination_folder)
    destination_folder.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving qcd and top to {destination_folder}")
    ak.to_parquet(ak_arr[is_qcd], destination_folder / f"ZJetsToNuNu_{number}.parquet")
    ak.to_parquet(ak_arr[~is_qcd], destination_folder / f"TTBar_{number}.parquet")

    if convert_to_jetclass_style_root:
        logger.info("Converting to JetClass style root files.")
        landscape_file_to_jetclass_style(
            destination_folder / f"ZJetsToNuNu_{number}.parquet",
            destination_folder / f"ZJetsToNuNu_{number}.root",
        )
        landscape_file_to_jetclass_style(
            destination_folder / f"TTBar_{number}.parquet",
            destination_folder / f"TTBar_{number}.root",
        )
        logger.info("Deleting parquet files.")
        (destination_folder / f"ZJetsToNuNu_{number}.parquet").unlink()
        (destination_folder / f"TTBar_{number}.parquet").unlink()


def create_mini_root_file(input_file, output_file, max_entries=1000):
    """
    Create a new root file with a subset of the entries from the input file.

    Parameters
    ----------
    input_file : str
        Path to the input root file.
    output_file : str
        Path to the output root file.
    max_entries : int (optional)
        Maximum number of entries to include in the output root file.
        Default is 1000.
    """
    # load the initial root file and create a new root file with a subset of the entries
    with uproot.open(input_file) as file:
        tree = file["tree"]
        ak_arrays = tree.arrays(library="ak", how="zip")[:max_entries]
        dict_for_writing = {key: ak_arrays[key] for key in ak_arrays.fields}
    with uproot.recreate(output_file) as file:
        file["tree"] = dict_for_writing
        print(f"Created {output_file} with {max_entries} entries")


def landscape_file_to_jetclass_style(filename, save_path, n_jets=None):
    """Convert a landscape file to the JetClass style.

    Parameters
    ----------
    filename : str
        The path to the file to load.
    save_path : str
        The path to the file to save.
    """

    ak_arr = ak.from_parquet(filename)[:n_jets]
    labels_dict = {
        "label_Hbb": np.zeros(len(ak_arr)),
        "label_Hcc": np.zeros(len(ak_arr)),
        "label_Hgg": np.zeros(len(ak_arr)),
        "label_H4q": np.zeros(len(ak_arr)),
        "label_Hqql": np.zeros(len(ak_arr)),
        "label_Zqq": np.zeros(len(ak_arr)),
        "label_Wqq": np.zeros(len(ak_arr)),
        "label_Tbl": np.zeros(len(ak_arr)),
    }
    # check if it's a top jet file with two steps
    if "TTBar" in str(filename):
        # confirm that the file is a top jet file
        assert np.all(ak_arr["label"] == 1)
        labels_dict["label_Tbqq"] = np.ones(len(ak_arr))
        labels_dict["label_QCD"] = np.zeros(len(ak_arr))
    elif "ZJetsToNuNu" in str(filename):
        # confirm that the file is a qcd jet file
        assert np.all(ak_arr["label"] == 0)
        labels_dict["label_QCD"] = np.ones(len(ak_arr))
        labels_dict["label_Tbqq"] = np.zeros(len(ak_arr))
    else:
        raise ValueError(
            "File is not a top or qcd jet file. Two things are checked:"
            "1. The label is 1 for top jets and 0 for qcd jets."
            "2. The filename contains 'TTBar' for top jets and 'ZJetsToNuNu' for qcd jets."
        )

    p4s = ak.zip(
        {
            "px": ak_arr["part_px"],
            "py": ak_arr["part_py"],
            "pz": ak_arr["part_pz"],
            "energy": ak_arr["part_energy"],
        },
        with_name="Momentum4D",
    )
    jets = ak.sum(p4s, axis=1)

    particle_features = {
        "part_px": p4s.px,
        "part_py": p4s.py,
        "part_pz": p4s.pz,
        "part_energy": p4s.energy,
        "part_pt": p4s.pt,
        "part_eta": p4s.eta,
        "part_phi": p4s.phi,
        "part_deta": p4s.deltaeta(jets),
        "part_dphi": p4s.deltaphi(jets),
        "part_etarel": p4s.deltaeta(jets),
        "part_phirel": p4s.deltaphi(jets),
        "part_ptrel": p4s.pt / jets.pt,
        "part_erel": p4s.energy / jets.energy,
        "part_deltaR": p4s.deltaR(jets),
    }

    jet_features = {
        "jet_pt": jets.pt,
        "jet_eta": jets.eta,
        "jet_phi": jets.phi,
        "jet_energy": jets.energy,
        "jet_mass": jets.mass,
        "jet_nparticles": ak.num(p4s),
    }

    logger.info(f"Saving to {save_path}")

    with uproot.recreate(save_path) as f:
        f["tree"] = particle_features | jet_features | labels_dict


def safe_load_features_from_ak_array(
    ak_array: ak.Array,
    features: list,
    load_zeros_if_not_present: bool = False,
    verbose: bool = True,
) -> ak.Array:
    """Load features from an awkward array, checking if they are present.

    Parameters
    ----------
    ak_array : ak.Array
        The awkward array to load the features from.
    features : list
        List of features to load.
    load_zeros_if_not_present : bool, optional
        If True, load zeros for features that are not present. Default is False.
    verbose : bool, optional
        If True, print information about the features being loaded. Default is True.

    Returns
    -------
    ak.Array
        An awkward array with the requested features.
    """
    available_features = [f for f in features if f in ak_array.fields]
    if not available_features:
        raise ValueError("No requested features are present in the awkward array.")

    if verbose:
        logger.info(f"Loading features: {available_features} from the awkward array.")
        logger.info(f"Available features in the awkward array: {ak_array.fields}.")

    missing_features = set(features) - set(available_features)

    if load_zeros_if_not_present:
        if len(missing_features) != 0:
            logger.warning(
                f"Features {missing_features} are not present in the awkward array. "
                "Loading zeros for these features."
            )
        for feature in missing_features:
            ak_array[feature] = ak.zeros_like(ak_array[available_features[0]])
        return ak_array[features]
    else:
        logger.warning(
            f"Features {missing_features} are not present in the awkward array. "
            "They will not be loaded."
        )
        return ak_array[available_features]
