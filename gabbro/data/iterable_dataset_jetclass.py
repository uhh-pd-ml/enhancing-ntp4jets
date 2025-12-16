import glob
from copy import deepcopy
from typing import Optional

import awkward as ak
import lightning as L
import numpy as np
import torch
import vector
from torch.distributed import get_rank, get_world_size
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from gabbro.data.loading import (
    read_cms_open_data_file,
    read_jetclass_file,
    read_tokenized_jetclass_file,
)
from gabbro.utils.arrays import (
    ak_pad,
    ak_select_and_preprocess,
    ak_to_np_stack,
)
from gabbro.utils.pylogger import get_pylogger
from gabbro.utils.utils import translate_bash_range

vector.register_awkward()


class CustomIterableDataset(IterableDataset):
    """Custom IterableDataset that loads data from multiple files."""

    def __init__(
        self,
        files_dict: dict,
        n_files_at_once: int = None,
        n_jets_per_file: int = None,
        max_n_files_per_type: int = None,
        shuffle_files: bool = True,
        shuffle_data: bool = True,
        seed: int = 4697,
        seed_shuffle_data: int = 3838,
        pad_length: int = 128,
        pad_fill_value: float = 0.0,
        logger_name: str = "CustomIterableDataset",
        feature_dict: dict = None,
        feature_dict_labels_particles: dict = None,
        feature_dict_jet: dict = None,
        labels_to_load: list = None,
        token_reco_cfg: dict = None,
        token_id_cfg: dict = None,
        load_only_once: bool = False,
        shuffle_only_once: bool = False,
        dont_shuffle_files_list_at_beginning: bool = False,
        random_seed_for_per_file_shuffling: int = None,
        collate: bool = False,
        dataset_type: str = "jetclass",
        shuffle_particles: bool = False,
        verbose: bool = False,
        **kwargs,
    ):
        """
        Parameters
        ----------
        files_dict : dict
            Dict with the file names for each type. Can be e.g. a dict like
            {"tbqq": ["tbqq_0.root", ...], "qcd": ["qcd_0.root", ...], ...}.
        n_files_at_once : int, optional
            Number of files to load at once. If None, one file per files_dict key
            is loaded.
        n_jets_per_file : int, optional
            Number of jets loaded from each individual file. Defaults to None, which
            means that all jets are loaded.
        max_n_files_per_type : int, optional
            Maximum number of files to use per type. If None, all files are used.
            Can be used to use e.g. always the first file from the sorted list of files
            in validation.
        shuffle_files : bool, optional
            Whether to shuffle the list of files.
        shuffle_data : bool, optional
            Whether to shuffle the data after loading.
        seed : int, optional
            Random seed.
        seed_shuffle_data : int, optional
            Random seed for shuffling the data. This is useful if you want to shuffle
            the data in the same way for different datasets (e.g. train and val).
            The default value is 3838.
        pad_length : int, optional
            Maximum number of particles per jet. If a jet has more particles, the
            first pad_length particles are used, the rest is discarded.
        pad_fill_value : float, optional
            Value to fill the padded values with.
        logger_name : str, optional
            Name of the logger.
        feature_dict : dict, optional
            Dictionary with the particle features to load. The keys are the names of the features
            and the values are the preprocessing parameters passed to the
            `ak_select_and_preprocess` function.
        feature_dict_labels_particles : dict, optional
            Dictionary with the particle labels to load. The keys are the names of the features
            and the values are the preprocessing parameters passed to the
            `ak_select_and_preprocess` function.
        feature_dict_jet : dict, optional
            Dictionary with the jet features to load. The keys are the names of the features
            and the values are the preprocessing parameters passed to the
            `ak_select_and_preprocess` function.
        feature_dict_jet : dict, optional
            Dictionary with the jet features to load. The keys are the names of the features
        labels_to_load : list, optional
            List with the jet_type labels to load.
        token_reco_cfg : dict, optional
            Dictionary with the configuration to reconstruct the tokenized jetclass files.
            If None, this is not used.
        token_id_cfg : dict, optional
            Dictionary with the tokenization configuration, this is to be used when the
            token-id data is to be loaded. If None, this is ignored.
        load_only_once : bool, optional
            If True, the data is loaded only once and then returned in the same order
            in each iteration. NOTE: this is only useful if the whole dataset fits into
            memory. If the dataset is too large, this will lead to a memory error.
        shuffle_only_once : bool, optional
            If True, the data is shuffled only once and then returned in the same order
            in each iteration. NOTE: this should only be used for val/test.
        dont_shuffle_files_list_at_beginning : bool, optional
            If True, the files list is not shuffled at the beginning. Default is False.
        random_seed_for_per_file_shuffling : int, optional
            Random seed for shuffling the jets within a file. This is useful if you want
            to only load a subset of the jets from a file and want to choose different
            jets in different training runs.
            If load_only_once is False, this is ignored.
        collate: bool, optional
            Whether to use the collate function to perform padding up to the max
            length of each batch only. Default is False. Note that this will result
            in different padding lengths between batches, care needs to be taken for
            further operations that require same-length sequences across all batches.
        shuffle_particles: bool, optional
            Whether to shuffle the particles within each jet. Default is False.
        verbose: bool, optional
            Whether to print verbose output. Default is False.
        **kwargs
            Additional keyword arguments.
        """
        if feature_dict is None:
            raise ValueError("feature_dict must be provided.")
        if labels_to_load is None:
            raise ValueError("labels_to_load must be provided.")

        worker_info = get_worker_info()
        rank = get_rank() if torch.distributed.is_initialized() else 0
        world_size = get_world_size() if torch.distributed.is_initialized() else 1

        self.multi_gpu_info = {
            "num_gpus": torch.cuda.device_count(),
            "process_rank": rank,
            "world_size": world_size,
            "device": f"cuda:{rank}" if torch.cuda.is_available() else "cpu",
            "worker_id": worker_info.id if worker_info is not None else 0,
            "num_workers": worker_info.num_workers if worker_info is not None else 1,
        }

        # initialize the logger of this dataset (by default with rank=None and then
        # later call this again and get rank the lightning trainer)
        self.logger_name = logger_name
        self.setup_logger(rank=None)

        self.logger.info(f"{[f'{key}={value}' for key, value in self.multi_gpu_info.items()]}")

        self.seed = seed
        self.verbose = verbose
        self.pad_length = pad_length
        self.pad_fill_value = pad_fill_value
        self.shuffle_data = shuffle_data
        self.shuffle_files = shuffle_files
        self.shuffle_files_counter = 0
        self.shuffle_particles = shuffle_particles
        self.processed_files_counter = 0
        self.max_n_files_per_type = max_n_files_per_type
        self.n_jets_per_file = n_jets_per_file
        self.feature_dict = feature_dict
        self.feature_dict_jet = feature_dict_jet
        self.labels_to_load = labels_to_load
        self.particle_features_list = [
            feat for feat in self.feature_dict.keys() if feat.startswith("part")
        ]
        self.feature_dict_labels_particles = feature_dict_labels_particles
        self.labels_particles_list = (
            None
            if self.feature_dict_labels_particles is None
            else [
                feat
                for feat in self.feature_dict_labels_particles.keys()
                if feat.startswith("part")
            ]
        )
        self.jet_features_list = (
            None
            if self.feature_dict_jet is None
            else [feat for feat in self.feature_dict_jet.keys() if feat.startswith("jet")]
        )
        self.seed_shuffle_data = seed_shuffle_data
        self.load_only_once = load_only_once
        self.shuffle_only_once = shuffle_only_once
        self.data_shuffled = False
        self.dont_shuffle_files_list_at_beginning = dont_shuffle_files_list_at_beginning
        self.random_seed_for_per_file_shuffling = random_seed_for_per_file_shuffling
        self.collate = collate
        # raise error if particle-level labels are provided and collate is True
        if self.feature_dict_labels_particles is not None and self.collate:
            raise ValueError(
                "Particle-level labels are provided, but collate is True. "
                "This is not supported atm (in the shuffling etc). "
                "Please set collate to False."
            )
        self.dataset_type = dataset_type
        valid_dataset_types = ["jetclass", "aoj"]
        if self.dataset_type not in valid_dataset_types:
            raise ValueError(f"Invalid dataset_type. Must be one of: {valid_dataset_types}")

        if self.random_seed_for_per_file_shuffling is not None:
            if not self.load_only_once:
                self.logger.warning(
                    "random_seed_for_per_file_shuffling is only used if load_only_once is True."
                )
                self.random_seed_for_per_file_shuffling = None
            else:
                self.logger.info(
                    f"Using random seed {self.random_seed_for_per_file_shuffling} for per-file shuffling."
                )

        # check that either shuffle_files or shift_files is used
        if not self.shuffle_files:
            self.logger.warning(
                "shuffle_files is False. This means that the files list will not be shuffled."
            )

        self.logger.info(f"Using seed {self.seed}")
        self.logger.info(f"Using the following labels: {self.labels_to_load}")
        self.logger.info(f"Using the following particle features: {self.particle_features_list}")
        self.logger.info(f"Using the following jet features: {self.jet_features_list}")
        self.logger.info(f"pad_length {self.pad_length} for the number of particles per jet.")
        self.logger.info(f"Using the following jet features: {self.jet_features_list}")
        self.logger.info(f"shuffle_data={self.shuffle_data}")
        self.logger.info(f"shuffle_files={self.shuffle_files}")
        self.logger.info(
            "Number of jets loaded per file: "
            f"{self.n_jets_per_file if self.n_jets_per_file is not None else 'all'}"
        )
        self.logger.info("Using the following particle features:")
        for feat, params in self.feature_dict.items():
            self.logger.info(f"- {feat}: {params}")
        if self.feature_dict_jet is not None:
            self.logger.info("Using the following jet features:")
            for feat, params in self.feature_dict_jet.items():
                self.logger.info(f"- {feat}: {params}")
        self.files_dict = {}
        for jet_type, files in files_dict.items():
            expanded_files = []
            for file in files:
                translated_files = translate_bash_range(file)
                for t_file in translated_files:
                    expanded_files.extend(sorted(list(glob.glob(t_file))))
            self.files_dict[jet_type] = (
                expanded_files
                if max_n_files_per_type is None
                else expanded_files[:max_n_files_per_type]
            )

            self.logger.info(f"Files for jet_type {jet_type}:")
            for file in self.files_dict[jet_type]:
                self.logger.info(f" - {file}")

        if self.load_only_once:
            self.logger.warning(
                "load_only_once is True. This means that there will only be the initial data loading."
            )

        # add all files from the dict to a list (the values are lists of files)
        if not self.dont_shuffle_files_list_at_beginning:
            self.shuffle_file_list_for_each_jet_type()
        self.create_file_list()

        # if not specified how many files to use at once, use one file per jet_type
        if n_files_at_once is None:
            self.logger.warning("n_files_at_once not specified.")
            n_used_jet_types = len(list(self.files_dict.keys()))
            self.logger.warning(f"Setting n_files_at_once to {n_used_jet_types}.")
            self.n_files_at_once = n_used_jet_types
        else:
            if n_files_at_once > len(self.file_list):
                self.logger.warning(
                    f"n_files_at_once={n_files_at_once} is larger than the number of files in the"
                    f" dataset ({len(self.file_list)})."
                )
                self.logger.info("File list:")
                for f in self.file_list:
                    self.logger.info(f" - {f}")

                self.logger.warning(f"Setting n_files_at_once to {len(self.file_list)}.")
                self.n_files_at_once = len(self.file_list)
                if len(self.file_list) == 0:
                    files_dict_string = "\n".join(
                        [f"{jet_type}: {files}" for jet_type, files in files_dict.items()]
                    )
                    raise ValueError(
                        f"No files found. Make sure to check the file paths:\n{files_dict_string}"
                    )
            else:
                self.n_files_at_once = n_files_at_once

        self.logger.info(f"Will load {self.n_files_at_once} files at a time and combine them.")

        self.file_indices = np.array([0, self.n_files_at_once])
        self.file_iterations = len(self.file_list) // self.n_files_at_once
        if self.load_only_once:
            self.file_iterations = 1

        self.current_part_data = None
        self.current_part_mask = None
        self.current_jet_data = None
        self.current_labels_particle_data = None
        self.token_reco_cfg = token_reco_cfg
        self.token_id_cfg = token_id_cfg

        # check that only one of the token configs is used
        if (self.token_reco_cfg is not None) and (self.token_id_cfg is not None):
            raise ValueError("Only one of token_reco_cfg, token_id_cfg can be used.")

        if self.shuffle_particles and self.token_id_cfg is not None:
            return ValueError("shuffle_particles is only supported for JetClass dataset loading.")

        self.logger.info("Dataset initialized.")

    def setup_logger(self, rank: int = None) -> None:
        self.logger = get_pylogger(f"{__name__}-{self.logger_name}", rank=rank)
        self.logger.info("Logger set up (potentially with new rank information).")

    def shuffle_file_list_for_each_jet_type(self):
        """Either shuffle the files list of each jet type."""

        n_print = 10

        if not self.shuffle_files:
            self.logger.warning(
                "shuffle_files is False. Will not shuffle files and continue with "
                "the previous order."
            )
            return

        self.logger.info("Shuffling files")
        self.shuffle_files_counter += 1
        # shuffle the lists in the files_dict
        for i, (jet_type, files) in enumerate(self.files_dict.items()):
            # printout before shuffling
            self.logger.info(f"Shuffling files for jet_type {jet_type}")
            self.logger.info(f"Number of files in the list: {len(files)}")
            self.logger.info(f"First {n_print} entries of the file list before shuffling:")
            for file in self.files_dict[jet_type][:n_print]:
                self.logger.info(f" - {file.split('/')[-1]}")

            # do the shuffling
            rng = np.random.default_rng(self.seed + i + self.shuffle_files_counter)
            permutation = rng.permutation(len(files))
            files = [files[i] for i in permutation]
            self.files_dict[jet_type] = files

            # printout after shuffling
            self.logger.info(f"First {n_print} entries of the file list after shuffling:")
            for file in self.files_dict[jet_type][:n_print]:
                self.logger.info(f" - {file.split('/')[-1]}")

    def create_file_list(self, n_print=10):
        # merge the shuffled lists into a single list, but do one file at a time
        # to ensure that it's the same number of files per jet_type if the number
        # of files loaded at once is divisible by the number of jet_types
        self.file_list = []

        # get maximum number of files per type
        # (we do this because if there are fewer files for one type than for another,
        # we want to only use the number of files for the type with the fewest files)
        # this is to ensure that the number of files per type is the same
        n_files_per_type = np.array([len(files) for files in self.files_dict.values()])
        max_n_files = n_files_per_type.max()
        min_n_files = n_files_per_type.min()

        if max_n_files != min_n_files:
            self.logger.warning(
                f"Number of files per jet type is not the same for all jet types."
                f"max_n_files={max_n_files}, min_n_files={min_n_files}."
            )

        for i_file in range(min_n_files):
            for jet_type in self.files_dict.keys():
                if i_file < len(self.files_dict[jet_type]):
                    self.file_list.append(self.files_dict[jet_type][i_file])
                else:
                    self.logger.warning(f"No file {i_file} for jet_type {jet_type}")

        n_files_print = n_print if len(self.file_list) > n_print else len(self.file_list)

        self.logger.info(f"Number of files in the list: {len(self.file_list)}")
        self.logger.info(f"First {n_files_print} entries of the file list:")
        for file in self.file_list[:n_files_print]:
            self.logger.info(f" - {file.split('/')[-1]}")

        self.logger.info(f"Last {n_files_print} entries of the file list:")
        for file in self.file_list[-n_files_print:]:
            self.logger.info(f" - {file.split('/')[-1]}")

    def get_data(self):
        """Returns a generator (i.e. iterator) that goes over the current files list and returns
        batches of the corresponding data."""
        # Iterate over jet_type
        self.logger.info(">>> get_data() called")

        # Iterate over files
        for j in range(self.file_iterations):
            self.logger.debug(20 * "-")

            self.load_next_files()

            # loop over the current data
            for i in range(self.start_idx_this_gpu, self.end_idx_this_gpu):
                yield_data = {
                    "part_features": self.current_part_data[i],
                    "part_mask": self.current_part_mask[i],
                    "jet_type_labels_one_hot": self.current_jet_type_labels_one_hot[i],
                    "jet_type_labels": torch.argmax(self.current_jet_type_labels_one_hot[i]),
                }

                if self.current_jet_data is not None:
                    yield_data["jet_features"] = self.current_jet_data[i]

                if self.current_labels_particle_data is not None:
                    yield_data["part_labels"] = self.current_labels_particle_data[i]

                yield yield_data

    def __iter__(self):
        """Returns an iterable which represents an iterator that iterates over the dataset."""
        # get current global rank to make sure the logger is set up correctly and displays
        # the rank in the logs
        self.multi_gpu_info["process_rank"] = (
            get_rank() if torch.distributed.is_initialized() else 0
        )
        self.setup_logger(rank=self.multi_gpu_info["process_rank"])
        self.logger.info(">>> __iter__(self.get_data()) called")
        return iter(self.get_data())

    def load_next_files(self):
        """Load the next files from the file list."""

        if self.load_only_once:
            if self.current_part_data is not None:
                self.logger.warning(
                    "Trying to load next files, but data has already been loaded and "
                    "`load_only_once=True` --> will not load again."
                )
                self.shuffle_current_data()
                return

        if self.load_only_once:
            self.logger.warning("Loading data only once. Will not load other files afterwards.")
            self.logger.warning("--> This will be the data for all iterations.")

        if self.processed_files_counter > 0:
            self.logger.info(
                f"self.processed_files_counter={self.processed_files_counter} is larger than 0 "
                f"and smaller than the total number of files in the dataset ({len(self.file_list)})."
                " This means that the files list was not fully traversed in the previous "
                "iteration. Will continue with the current files list."
            )

        def load_particles(self):
            """
            Loads particles from current files

            Returns
            ----------
            jet_data_list, part_data_list, mask_data_list, jet_type_labels_list, labels_particles_data_list: lists

            """

            part_data_list = []
            labels_particles_data_list = []
            mask_data_list = []
            jet_data_list = []
            jet_type_labels_list = []

            self.current_files = self.file_list[: self.n_files_at_once]

            self.logger.info("Loading next files.")

            for i_file, filename in enumerate(self.current_files):
                self.logger.info(f"{i_file + 1} / {len(self.current_files)} : {filename}")

                if self.token_id_cfg is not None:
                    if self.token_id_cfg.get("feature_resolution", None) is not None:
                        allowed_feature_resolutions = ["raw", "tokenized"]
                        if (
                            self.token_id_cfg["feature_resolution"]
                            not in allowed_feature_resolutions
                        ):
                            raise ValueError(
                                f"feature_resolution must be one of {allowed_feature_resolutions}, "
                                f"but got {self.token_id_cfg['feature_resolution']}."
                            )

                        if self.token_id_cfg.get("particle_features", None) is None:
                            raise ValueError(
                                "If feature_resolution is provided, particle_features must "
                                "also be provided."
                            )
                        elif "particle_features_tokenized" in self.token_id_cfg:
                            raise ValueError(
                                "If feature_resolution is provided, particle_features_tokenized "
                                "must not be provided."
                            )

                        # translate new syntax to particle_features and particle_features_tokenized
                        if self.token_id_cfg.get("feature_resolution") == "tokenized":
                            particle_features_tokenized = self.token_id_cfg["particle_features"]
                            particle_features = None
                            # remove all features that include "part_token_id"
                            particle_features_tokenized = [
                                feat
                                for feat in particle_features_tokenized
                                if "part_token_id" not in feat
                            ]
                            if len(particle_features_tokenized) == 0:
                                particle_features_tokenized = None
                        elif self.token_id_cfg.get("feature_resolution") == "raw":
                            particle_features_tokenized = None
                            particle_features = self.token_id_cfg["particle_features"]
                            # remove all features that include "part_token_id"
                            particle_features = [
                                feat for feat in particle_features if "part_token_id" not in feat
                            ]
                            if len(particle_features) == 0:
                                particle_features = None

                    else:
                        particle_features_tokenized = self.token_id_cfg.get(
                            "particle_features_tokenized", None
                        )
                        particle_features = self.token_id_cfg.get("particle_features", None)

                    # This means that the token-id data is loaded, thus only the token ids
                    ak_x_particles, ak_jet_type_labels, ak_x_jet = read_tokenized_jetclass_file(
                        filename,
                        load_token_ids=self.token_id_cfg.get("load_token_ids", True),
                        particle_features=particle_features,
                        particle_features_tokenized=particle_features_tokenized,
                        labels=self.labels_to_load,
                        remove_start_token=self.token_id_cfg.get("remove_start_token", False),
                        remove_end_token=self.token_id_cfg.get("remove_end_token", False),
                        shift_tokens_minus_one=self.token_id_cfg.get(
                            "shift_tokens_minus_one", False
                        ),
                        add_padded_particle_features_start=self.token_id_cfg.get(
                            "add_padded_particle_features_start", False
                        ),
                        add_padded_particle_features_end=self.token_id_cfg.get(
                            "add_padded_particle_features_end", False
                        ),
                        n_load=self.n_jets_per_file,
                        random_seed=self.random_seed_for_per_file_shuffling,
                        jet_features=self.jet_features_list,
                        verbose=self.verbose,
                    )
                else:
                    if self.dataset_type == "aoj":
                        ak_x_particles, ak_x_jet, ak_jet_type_labels = read_cms_open_data_file(
                            filename,
                            particle_features=self.particle_features_list,
                            labels=self.labels_to_load,
                        )
                    else:
                        # read the data from the file (just normal JetClass file, not tokenization involved)
                        # can add jet features, labels, and p4s here
                        ak_x_particles, ak_x_jet, ak_jet_type_labels = read_jetclass_file(
                            filename,
                            particle_features=self.particle_features_list,
                            jet_features=self.jet_features_list,
                            labels=self.labels_to_load,
                            n_load=self.n_jets_per_file,
                            shuffle_particles=self.shuffle_particles,
                            random_seed=self.random_seed_for_per_file_shuffling,
                        )

                # make sure that only the first n_jets_per_file jets are loaded
                if self.n_jets_per_file is not None:
                    self.logger.info(
                        f"Selecting the first {self.n_jets_per_file} jets from the file."
                    )
                    if len(ak_x_particles) > self.n_jets_per_file:
                        ak_x_particles = ak_x_particles[: self.n_jets_per_file]
                        ak_jet_type_labels = ak_jet_type_labels[: self.n_jets_per_file]
                        if self.token_id_cfg is not None:
                            ak_x_jet = ak_x_jet[: self.n_jets_per_file]
                    else:
                        self.logger.info(
                            f"Number of jets from the file {len(ak_x_particles)} is "
                            f"smaller/equal than n_jets_per_file={self.n_jets_per_file}. "
                            "Depending on the configuration, the loading function already "
                            "takes care of loading only the requested number of jets."
                        )

                # copy the ak_x_particles to a new array, since we will modify it
                ak_x_particles_copy = deepcopy(ak_x_particles)

                ak_x_particles = ak_select_and_preprocess(ak_x_particles, self.feature_dict)
                if not self.collate:
                    # Padding happens already here, instead of in the collate function
                    ak_x_particles_padded, ak_mask_particles = ak_pad(
                        ak_x_particles,
                        self.pad_length,
                        return_mask=True,
                        fill_value=self.pad_fill_value,
                    )
                    # The following converts to np, and stacks across all features in the particle_features_list
                    np_x_particles_padded = ak_to_np_stack(
                        ak_x_particles_padded, names=self.particle_features_list
                    )
                    np_mask_particles = ak.to_numpy(ak_mask_particles)

                np_jet_type_labels = ak_to_np_stack(ak_jet_type_labels, names=self.labels_to_load)
                if self.feature_dict_jet is not None:
                    ak_x_jet = ak_select_and_preprocess(ak_x_jet, self.feature_dict_jet)
                    np_x_jet = ak_to_np_stack(ak_x_jet, names=self.jet_features_list)
                    jet_data_list.append(torch.tensor(np_x_jet))

                if self.feature_dict_labels_particles is not None:
                    ak_labels_particles = ak_select_and_preprocess(
                        ak_x_particles_copy, self.feature_dict_labels_particles
                    )
                    ak_labels_particles_padded, _ = ak_pad(
                        ak_labels_particles,
                        self.pad_length,
                        return_mask=True,
                        fill_value=self.pad_fill_value,
                    )
                    np_labels_particles = ak_to_np_stack(
                        ak_labels_particles_padded, names=self.labels_particles_list
                    )
                    labels_particles_data_list.append(torch.tensor(np_labels_particles))

                # add the data to the lists
                if not self.collate:
                    part_data_list.append(
                        torch.tensor(np_x_particles_padded)
                    )  # This is now a nice tensor
                    mask_data_list.append(torch.tensor(np_mask_particles, dtype=torch.bool))
                else:
                    part_data_list.append(ak_x_particles)  # Note that we still have ak here
                jet_type_labels_list.append(torch.tensor(np_jet_type_labels))

            return (
                jet_data_list,
                part_data_list,
                mask_data_list,
                jet_type_labels_list,
                labels_particles_data_list,
            )

        (
            self.jet_data_list,
            self.part_data_list,
            self.mask_data_list,
            self.jet_type_labels_list,
            self.labels_particles_data_list,
        ) = load_particles(self)
        self.logger.info("Loaded particles.")

        # concatenate the data from all files
        if not self.collate:
            self.current_part_data = torch.cat(self.part_data_list, dim=0)
            self.current_part_mask = torch.cat(self.mask_data_list, dim=0)
        else:
            self.current_part_data = ak.concatenate(self.part_data_list, axis=0)
            # Placeholder, since we will do padding and masking later
            self.current_part_mask = np.zeros(len(self.current_part_data))

        if self.feature_dict_labels_particles is not None:
            self.current_labels_particle_data = torch.cat(self.labels_particles_data_list, dim=0)

        if self.feature_dict_jet is not None:
            self.current_jet_data = torch.cat(self.jet_data_list, dim=0)

        self.current_jet_type_labels_one_hot = torch.cat(self.jet_type_labels_list, dim=0)
        self.data_shuffled = False

        self.shuffle_current_data()

        if not self.collate:
            self.logger.info(
                f">>> Data loaded. (self.current_part_data.shape = {self.current_part_data.shape})"
            )
        else:
            self.logger.info(
                f">>> Data loaded. (self.current_part_data = {self.current_part_data})"
            )
        if self.current_jet_data is not None:
            self.logger.info(
                f">>> Data loaded. (self.current_jet_data.shape = {self.current_jet_data.shape})"
            )
        self.set_indices_for_this_rank()

        # update the counter which keeps track of how many files have been processed
        self.processed_files_counter += self.n_files_at_once
        self.logger.info(
            "Updating self.processed_files_counter. The new value is "
            f"self.processed_files_counter = {self.processed_files_counter}."
        )
        self.logger.info(
            "Checking if all files in the current files list have been processed. "
            "If so, the file list will be shuffled (unless `shuffle_files=False`)"
            "such that the next iteration will proceed with a new file list."
        )
        self.traverse_file_list_and_shuffle_if_fully_traversed()

    def traverse_file_list_and_shuffle_if_fully_traversed(self):
        """Traverse the file list and shuffle the files list if all files have been processed."""

        if self.processed_files_counter >= len(self.file_list):
            self.logger.info(
                "All files in the current files list have been processed. "
                "Shuffling the files list within each jet type and creating new "
                "self.file_list before proceeding."
            )
            self.shuffle_file_list_for_each_jet_type()
            self.create_file_list()
            self.processed_files_counter = 0
            self.logger.info(
                "Resetting self.processed_files_counter to "
                f"self.processed_files_counter = {self.processed_files_counter}."
            )
        else:
            self.logger.info(
                f"Processed {self.processed_files_counter} / {len(self.file_list)} files."
            )
            self.logger.info(
                "Not all files in the current files list have been processed. "
                "Will continue with the current files list."
            )
            self.logger.info(
                "Shifting the current files list by "
                f"`n_files_at_once={self.n_files_at_once}` files."
            )
            self.file_list = (
                self.file_list[self.n_files_at_once :] + self.file_list[: self.n_files_at_once]
            )

    def set_indices_for_this_rank(self):
        """Set the start and end indices to load for this rank."""
        # set the indices to load for each gpu
        if self.multi_gpu_info["world_size"] > 1:
            # split the self.current_part_data over the gpus
            n_jets = len(self.current_part_data)
            n_jets_per_gpu = n_jets // self.multi_gpu_info["world_size"]
            self.start_idx_this_gpu = n_jets_per_gpu * self.multi_gpu_info["process_rank"]
            self.end_idx_this_gpu = n_jets_per_gpu * (self.multi_gpu_info["process_rank"] + 1)
        else:
            self.start_idx_this_gpu = 0
            self.end_idx_this_gpu = len(self.current_part_data)

        self.logger.info(
            f"Rank {self.multi_gpu_info['process_rank']} will load data from index "
            f"{self.start_idx_this_gpu} to {self.end_idx_this_gpu}"
        )

    def shuffle_current_data(self):
        """Shuffle the current data."""
        # shuffle the data
        if self.shuffle_only_once and self.data_shuffled:
            self.logger.info(
                "Data has already been shuffled and `shuffle_only_once=True` "
                "--> will not shuffle again."
            )

        elif self.shuffle_data:
            rng = np.random.default_rng()
            if self.seed_shuffle_data is not None:
                self.logger.info(f"Shuffling data with seed {self.seed_shuffle_data}")
                rng = np.random.default_rng(self.seed_shuffle_data)
            perm = rng.permutation(len(self.current_part_data))
            self.current_part_data = self.current_part_data[perm]
            self.current_labels_particle_data = (
                self.current_labels_particle_data[perm]
                if self.current_labels_particle_data is not None
                else None
            )
            self.current_part_mask = self.current_part_mask[perm]
            self.current_jet_type_labels_one_hot = self.current_jet_type_labels_one_hot[perm]
            self.data_shuffled = True
            if self.current_jet_data is not None:
                self.current_jet_data = self.current_jet_data[perm]

            self.logger.info("Data shuffled.")

        if not self.collate:
            self.logger.info(
                "The particle features of the first three particles in the first and last jet "
                f"are now: {self.current_part_data[0, :3, :]}, {self.current_part_data[-1, :3, :]}"
            )
            if self.current_jet_data is not None:
                self.logger.info(
                    "The jet features of the first three jets are now: "
                    f"{self.current_jet_data[:3, :]}"
                )
            if self.current_labels_particle_data is not None:
                self.logger.info(
                    "The particle labels of the first three particles in the first and last jet "
                    "are now: "
                    f"{self.current_labels_particle_data[0, :3, :]}, "
                    f"{self.current_labels_particle_data[-1, :3, :]}"
                )


class IterableDatamodule(L.LightningDataModule):
    def __init__(
        self,
        dataset_kwargs_train: dict,
        dataset_kwargs_val: dict,
        dataset_kwargs_test: dict,
        dataset_kwargs_common: dict,
        batch_size: int = 256,
        **kwargs,
    ):
        super().__init__()

        if isinstance(batch_size, int):
            self.batch_size_train = batch_size
            self.batch_size_val = batch_size
            self.batch_size_test = batch_size
        else:
            # then we expect a dict with the batch sizes for train, val, and test
            if "train" not in batch_size or "val" not in batch_size or "test" not in batch_size:
                raise ValueError(
                    "If batch_size is a dict, it must include keys 'train', 'val', 'test'"
                )
            self.batch_size_train = batch_size["train"]
            self.batch_size_val = batch_size["val"]
            self.batch_size_test = batch_size["test"]

        # save the parameters as attributes
        self.save_hyperparameters()
        self.logger = get_pylogger(f"{__name__}-{self.__class__.__name__}")

        self.logger.info(f"batch_size_train: {self.batch_size_train}")
        self.logger.info(f"batch_size_val: {self.batch_size_val}")
        self.logger.info(f"batch_size_test: {self.batch_size_test}")

        # merge dataset_kwargs_common with the other dataset_kwargs
        # if a parameter is present in both, the one in dataset_kwargs_train/val/test
        # will be used

        def print_warning(key, value_common, value_specific, train_val_test):
            self.logger.warning(
                f"Parameter {key} is present in both dataset_kwargs_common "
                f"(value={value_common}) and dataset_kwargs_{train_val_test}"
                f"(value={value_specific}). "
                f"Using the value from dataset_kwargs_{train_val_test}."
            )

        for key, value in dataset_kwargs_common.items():
            if key not in dataset_kwargs_train:
                dataset_kwargs_train[key] = value
            else:
                print_warning(key, value, dataset_kwargs_train[key], "train")
            if key not in dataset_kwargs_val:
                dataset_kwargs_val[key] = value
            else:
                print_warning(key, value, dataset_kwargs_val[key], "val")
            if key not in dataset_kwargs_test:
                dataset_kwargs_test[key] = value
            else:
                print_warning(key, value, dataset_kwargs_test[key], "test")

        self.hparams.dataset_kwargs_train = dataset_kwargs_train
        self.hparams.dataset_kwargs_val = dataset_kwargs_val
        self.hparams.dataset_kwargs_test = dataset_kwargs_test

    def prepare_data(self) -> None:
        """Prepare the data."""
        pass

    def pad_constituents_and_stack(self, batch):
        """
        Pad constituents (located in batch["part_features"]) to the max length
        event in the batch, generate a padding mask and prepare the batch for
        further processing.

        Parameters
        ----------
        batch : list
            Each element of the batch corresponds to one event, and each event
            has its own dictionary of 'part_features', 'part_mask' etc, as
            specified in the yield of CustomIterableDataset.get_data():
                - 'part_features' is expected to be an awkward Record, containing
                  fields corresponding to the items in the feature_dict under
                  the dataset_kwargs_common section of the config.
                - 'part_mask' is a placeholder float.
                - The remaining items are expected to already be tensors.
        batched_data : dict
            A dictionary of 'part_features', 'part_mask' etc, where each value
            is a tensor of size [batch_size, ...]
        """
        ak_batch = ak.Array(batch)

        # Get the field names from feature_dict, eg 'part_token_id_without_last'
        feature_dict = self.hparams.dataset_kwargs_common.get("feature_dict", {})
        if feature_dict == {}:
            raise ValueError("feature_dict must be specified")
        particle_features_list = [feat for feat in feature_dict.keys() if feat.startswith("part")]

        maxlen = int(ak.max(ak.num(ak_batch["part_features"][particle_features_list[0]])))

        # Pad
        pad_fill_value = self.hparams.dataset_kwargs_common.get("pad_fill_value", 0)
        ak_x_particles_padded, ak_mask_particles = ak_pad(
            ak_batch["part_features"],
            maxlen,
            return_mask=True,
            fill_value=pad_fill_value,
        )

        # Stack and convert to numpy
        np_x_particles_padded = ak_to_np_stack(ak_x_particles_padded, names=particle_features_list)
        np_mask_particles = ak.to_numpy(ak_mask_particles)

        # Convert to tensors
        new_part_features = torch.tensor(np_x_particles_padded)
        part_mask = torch.tensor(np_mask_particles, dtype=torch.bool)

        # "Transpose" the structure so a batch is a dictionary with batch_size
        # elements, rather than a list of events where each event has its own
        # dictionary.
        batched_data = {}
        for key in batch[0]:
            if key == "part_features" or key == "part_mask":  # We'll do these separately
                continue
            batched_data[key] = torch.Tensor(ak_batch[key])

        batched_data["part_features"] = new_part_features
        batched_data["part_mask"] = part_mask

        return batched_data

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit":
            self.train_dataset = CustomIterableDataset(
                **self.hparams.dataset_kwargs_train,
            )
            self.val_dataset = CustomIterableDataset(
                **self.hparams.dataset_kwargs_val,
            )
        elif stage == "test":
            self.test_dataset = CustomIterableDataset(
                **self.hparams.dataset_kwargs_test,
            )

    def train_dataloader(self):
        collate = self.hparams.dataset_kwargs_train.get("collate", False)
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size_train,
            collate_fn=self.pad_constituents_and_stack if collate else None,
        )

    def val_dataloader(self):
        collate = self.hparams.dataset_kwargs_val.get("collate", False)
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size_val,
            collate_fn=self.pad_constituents_and_stack if collate else None,
        )

    def test_dataloader(self):
        collate = self.hparams.dataset_kwargs_test.get("collate", False)
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size_test,
            collate_fn=self.pad_constituents_and_stack if collate else None,
        )
