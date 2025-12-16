"""Callback for evaluating the tokenization of particles."""

import math
import os

import awkward as ak
import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import vector

import gabbro.plotting.utils as plot_utils
from gabbro.data.iterable_dataset_jetclass import CustomIterableDataset
from gabbro.metrics.jet_substructure import JetSubstructure
from gabbro.plotting.feature_plotting import plot_features
from gabbro.utils.arrays import (
    ak_abs,
    ak_clip,
    ak_mean,
    ak_select_and_preprocess,
    ak_subtract,
    np_to_ak,
)
from gabbro.utils.jet_types import jet_types_dict

# from gabbro.plotting.plotting_functions import plot_p4s
from gabbro.utils.pylogger import get_pylogger
from gabbro.utils.utils import update_existing_dict_values

default_labels = {
    "pt": "$p_\\mathrm{T}$",
    "ptrel": "$p_\\mathrm{T}^\\mathrm{rel}$",
    "eta": "$\\eta$",
    "etarel": "$\\eta^\\mathrm{rel}$",
    "phi": "$\\phi$",
    "phirel": "$\\phi^\\mathrm{rel}$",
    "mass": "$m$",
}

pylogger = get_pylogger("TokenizationEvalCallback")
vector.register_awkward()


class TokenizationEvalCallback(L.Callback):
    def __init__(
        self,
        image_path: str = None,
        image_filetype: str = "png",
        no_trainer_info_in_filename: bool = False,
        save_result_arrays: bool = None,
    ):
        """Callback for evaluating the tokenization of particles.

        Parameters
        ----------
        image_path : str
            Path to save the images to. If None, the images are saved to the
            default_root_dir of the trainer.
        image_filetype : str
            Filetype to save the images as. Default is "png".
        no_trainer_info_in_filename : bool
            If True, the filename of the images will not contain the epoch and
            global step information. Default is False.
        save_result_arrays : bool
            If True, the results are saved as parquet file. Default is None.
        """
        super().__init__()
        self.comet_logger = None
        self.image_path = image_path
        self.image_filetype = image_filetype
        self.no_trainer_info_in_filename = no_trainer_info_in_filename
        self.save_results_arrays = save_result_arrays

    def on_validation_epoch_end(self, trainer, pl_module):
        pl_module.concat_validation_loop_predictions()
        self.plot(trainer, pl_module, stage="val")

    def on_test_epoch_end(self, trainer, pl_module):
        pl_module.concat_test_loop_predictions()
        self.plot(trainer, pl_module, stage="test")

    def plot(self, trainer, pl_module, stage="val"):
        plot_utils.set_mpl_style()
        if stage == "val" and not hasattr(pl_module, "val_x_original_concat"):
            pylogger.info("No validation predictions found. Skipping plotting.")
            return

        pylogger.info(
            f"Running TokenizationEvalCallback epoch: {trainer.current_epoch} step:"
            f" {trainer.global_step}"
        )
        # get loggers for saving the plots on comet/wandb later on in the script
        for logger in trainer.loggers:
            if isinstance(logger, L.pytorch.loggers.CometLogger):
                self.comet_logger = logger.experiment
            elif isinstance(logger, L.pytorch.loggers.WandbLogger):
                self.wandb_logger = logger.experiment

        # create the plot directory and set up base-filename of plots
        plot_dir = (
            self.image_path
            if self.image_path is not None
            else trainer.default_root_dir + "/plots/"
        )
        os.makedirs(plot_dir, exist_ok=True)
        if self.no_trainer_info_in_filename:
            plot_filename = f"{plot_dir}/evaluation_overview.{self.image_filetype}"
        else:
            if stage == "val":
                plot_filename = f"{plot_dir}/val_epoch{trainer.current_epoch}_gstep{trainer.global_step}_overview.{self.image_filetype}"
            elif stage == "test":
                plot_filename = f"{plot_dir}/test_overview.{self.image_filetype}"

        # get the results from the validation/test loop
        if stage == "val":
            x_recos = pl_module.val_x_reco_concat
            x_originals = pl_module.val_x_original_concat
            masks = pl_module.val_mask_concat
            labels = pl_module.val_labels_concat
            code_idx = pl_module.val_code_idx_concat
        elif stage == "test":
            # return and print that there are no test predictions if there are none
            if not hasattr(pl_module, "test_x_original_concat"):
                pylogger.info("No test predictions found. Skipping plotting.")
                return
            x_recos = pl_module.test_x_reco_concat
            x_originals = pl_module.test_x_original_concat
            masks = pl_module.test_mask_concat
            labels = pl_module.test_labels_concat
            code_idx = pl_module.test_code_idx_concat
        else:
            raise ValueError(f"stage {stage} not recognized")

        if stage == "test":
            pylogger.info(f"x_original_concat.shape: {x_originals.shape}")
            pylogger.info(f"x_reco_concat.shape: {x_recos.shape}")
            pylogger.info(f"masks_concat.shape: {masks.shape}")
            pylogger.info(f"labels_concat.shape: {labels.shape}")

        pp_dict = trainer.datamodule.hparams.dataset_kwargs_common.feature_dict

        # --- only use jets with more than 3 particles (otherwise the calculation
        # of the jet substructure will fail and the plotting will be wrong) ---
        more_than_3_particles_mask = np.sum(masks, axis=1) >= 3
        n_jets_removed = np.sum(~more_than_3_particles_mask)
        if n_jets_removed > 0:
            pylogger.warning(f"Removing {n_jets_removed} jets with less than 3 particles")

        x_recos = x_recos[more_than_3_particles_mask]
        x_originals = x_originals[more_than_3_particles_mask]
        masks = masks[more_than_3_particles_mask]
        labels = labels[more_than_3_particles_mask]
        # ----

        x_reco_ak_pp = np_to_ak(x_recos, mask=masks, names=pp_dict.keys())
        x_original_ak_pp = np_to_ak(x_originals, mask=masks, names=pp_dict.keys())
        x_reco_ak = ak_select_and_preprocess(x_reco_ak_pp, pp_dict=pp_dict, inverse=True)
        x_original_ak = ak_select_and_preprocess(x_original_ak_pp, pp_dict=pp_dict, inverse=True)

        eta_var = None
        phi_var = None
        pt_var = None
        mass_var = None

        possible_pt_keys = ["part_pt", "pt", "ptrel", "part_ptrel"]
        possible_eta_keys = ["part_eta", "eta", "etarel", "part_etarel"]
        possible_phi_keys = ["part_phi", "phi", "phirel", "part_phirel"]
        possible_mass_keys = ["part_mass", "mass"]

        for key in possible_eta_keys:
            if key in pp_dict:
                eta_var = key
                break
        for key in possible_pt_keys:
            if key in pp_dict:
                pt_var = key
                break
        for key in possible_phi_keys:
            if key in pp_dict:
                phi_var = key
                break
        for key in possible_mass_keys:
            if key in pp_dict:
                mass_var = key
                break

        p4s_reco_ak = ak.zip(
            {
                "pt": ak_clip(getattr(x_reco_ak, pt_var), clip_min=0.0),
                "eta": getattr(x_reco_ak, eta_var),
                "phi": getattr(x_reco_ak, phi_var),
                "mass": ak_clip(getattr(x_reco_ak, mass_var), clip_min=0)
                if mass_var is not None
                else ak.zeros_like(getattr(x_reco_ak, pt_var)),
            },
            with_name="Momentum4D",
        )
        p4s_original_ak = ak.zip(
            {
                "pt": getattr(x_original_ak, pt_var),
                "eta": getattr(x_original_ak, eta_var),
                "phi": getattr(x_original_ak, phi_var),
                "mass": getattr(x_original_ak, mass_var)
                if mass_var is not None
                else ak.zeros_like(getattr(x_original_ak, pt_var)),
            },
            with_name="Momentum4D",
        )

        p4s_jets_reco_ak = ak.sum(p4s_reco_ak, axis=1)
        p4s_jets_original_ak = ak.sum(p4s_original_ak, axis=1)

        if stage == "val":
            pl_module.val_p4s_reco_ak = p4s_reco_ak
            pl_module.val_p4s_original_ak = p4s_original_ak
        elif stage == "test":
            pl_module.test_p4s_reco_ak = p4s_reco_ak
            pl_module.test_p4s_original_ak = p4s_original_ak
            pl_module.test_p4s_jets_reco_ak = p4s_jets_reco_ak
            pl_module.test_p4s_jets_original_ak = p4s_jets_original_ak
        else:
            raise ValueError(f"stage {stage} not recognized")

        # --- plot the kinematic features on jet level ---
        fig, axarr = plot_features(
            ak_array_dict={
                "Original": p4s_jets_original_ak,
                "Reco": p4s_jets_reco_ak,
            },
            names={feat: default_labels[feat] for feat in ["pt", "eta", "phi", "mass"]},
            label_prefix="Jet",
            flatten=False,
        )
        fig.savefig(plot_filename)
        if self.comet_logger is not None:
            self.comet_logger.log_image(
                plot_filename, name=plot_filename.split("/")[-1], step=trainer.global_step
            )

        # calculate jet substructure if in test stage (takes some time, so only do it for val stage)
        jet_substructure_original = JetSubstructure(p4s_original_ak)
        jet_substructure_reco = JetSubstructure(p4s_reco_ak)

        # save the results
        results_ak_array = ak.Array(
            {
                "part_p4s_reco": p4s_reco_ak,
                "part_p4s_original": p4s_original_ak,
                "part_x_reco": x_reco_ak,
                "part_x_original": x_original_ak,
                "part_featurs_diff_to_original": ak_subtract(x_reco_ak, x_original_ak),
                "jet_p4s_reco": p4s_jets_reco_ak,
                "jet_p4s_original": p4s_jets_original_ak,
                "jet_substructure_reco": jet_substructure_reco.get_substructure_as_ak_array(),
                "jet_substructure_original": jet_substructure_original.get_substructure_as_ak_array(),
                "jet_substructure_diff_to_original": ak_subtract(
                    jet_substructure_reco.get_substructure_as_ak_array(),
                    jet_substructure_original.get_substructure_as_ak_array(),
                ),
                "labels": labels,
                "masks": masks,
            }
        )
        if self.image_path is not None and self.save_results_arrays:
            n_eval_jets = len(p4s_jets_reco_ak)
            out_file_name = f"{self.image_path}/eval_arrays_{n_eval_jets:_}.parquet"
            pylogger.info(f"Saving results to {out_file_name}")
            ak.to_parquet(results_ak_array, out_file_name)

        for i, (jet_type, jet_type_dict) in enumerate(jet_types_dict.items()):
            jet_type_mask = labels == jet_type_dict["label"]

            # extract the files_dict from the dataset to fix the jet type label
            # in case it's the iterable dataset with JetClass (not implemented for
            # other datasets)
            dataset_has_files_dict = False
            if stage == "val":
                dataset_has_files_dict = isinstance(
                    trainer.datamodule.val_dataset, CustomIterableDataset
                )
                files_dict = trainer.datamodule.val_dataset.files_dict
            elif stage == "test":
                files_dict = trainer.datamodule.test_dataset.files_dict
                dataset_has_files_dict = isinstance(
                    trainer.datamodule.test_dataset, CustomIterableDataset
                )

            if dataset_has_files_dict:
                if i >= len(files_dict):
                    continue
                jet_type = list(files_dict.keys())[i]
                jet_type_dict = jet_types_dict[jet_type]
                pylogger.info(f">>> Plotting jet type {jet_type}")
            else:
                pylogger.warning(
                    "The dataset is not an instance of CustomIterableDataset. "
                    "The jet type labels might be messed up in the evaluation plots."
                )
            jet_type_tex_label = jet_types_dict[jet_type]["tex_label"]

            # Plot the results from the jet substructure calculation
            ak_substructure_this_type_reco = results_ak_array["jet_substructure_reco"][
                jet_type_mask
            ]
            ak_substructure_this_type_original = results_ak_array["jet_substructure_original"][
                jet_type_mask
            ]
            ak_substructure_diff_this_type = results_ak_array["jet_substructure_diff_to_original"][
                jet_type_mask
            ]

            # print the fields
            pylogger.info(
                f"Fields in jet_substructure_reco: {ak_substructure_this_type_reco.fields}"
            )
            pylogger.info(
                f"Fields in jet_substructure_original: {ak_substructure_this_type_original.fields}"
            )

            fig_jet_features_this_type, axarr_jet_features_this_type = plot_features(
                ak_array_dict={
                    "Original": ak_substructure_this_type_original,
                    "Reco": ak_substructure_this_type_reco,
                },
                names={
                    "jet_pt": "Jet $p_\\mathrm{T}$ [GeV]",
                    "jet_eta": "Jet $\\eta$",
                    "jet_phi": "Jet $\\phi$",
                    "jet_mass": "Jet mass [GeV]",
                    "tau21": "$\\tau_{21}$",
                    "tau32": "$\\tau_{32}$",
                },
                flatten=False,
                ax_rows=2,
            )
            # plt.show()
            rep = "_overview"
            filename_jet_features_this_type = plot_filename.replace(
                rep, f"_jet_features_{jet_type}"
            )
            fig_jet_features_this_type.suptitle(
                f"Jet features of {jet_type_tex_label} jets", fontsize=16
            )
            fig_jet_features_this_type.tight_layout()
            fig_jet_features_this_type.savefig(filename_jet_features_this_type)
            pylogger.info(f"Saved jet features plot to {filename_jet_features_this_type}")

            # --- resolution / different to original plots
            labels_dict = {
                "jet_pt": "Jet $p_\\mathrm{T}^\\mathrm{reco} - p_\\mathrm{T}^\\mathrm{original}$ [GeV]",
                "jet_eta": "Jet $\\eta^\\mathrm{reco} - \\eta^\\mathrm{original}$",
                "jet_phi": "Jet $\\phi^\\mathrm{reco} - \\phi^\\mathrm{original}$",
                "jet_mass": "Jet $m^\\mathrm{reco} - m^\\mathrm{original}$ [GeV]",
                "tau21": "$\\tau_{21}^\\mathrm{reco} - \\tau_{21}^\\mathrm{original}$",
                "tau32": "$\\tau_{32}^\\mathrm{reco} - \\tau_{32}^\\mathrm{original}$",
                "d2": "$D_2^\\mathrm{reco} - D_2^\\mathrm{original}$",
            }

            fig_jet_features_res_this_type, axarr_jet_features_res_this_type = plot_features(
                ak_array_dict={"Reco. - Original": ak_substructure_diff_this_type},
                names=labels_dict,
                bins_dict={
                    "jet_pt": np.linspace(-30, 30, 100),  # -150, 150
                    "jet_eta": np.linspace(-0.1, 0.1, 100),
                    "jet_phi": np.linspace(-0.1, 0.1, 100),
                    "jet_mass": np.linspace(-20, 20, 100),  # -60, 60
                    "tau21": np.linspace(-0.5, 0.5, 100),
                    "tau32": np.linspace(-0.5, 0.5, 100),
                    "d2": np.linspace(-2, 2, 100),
                },
                flatten=False,
                ax_rows=2,
            )
            # draw vertical line at 0 on all axes
            for i in range(len(list(labels_dict.keys()))):
                axarr_jet_features_res_this_type[i].axvline(
                    0, color="black", linestyle="--", alpha=0.5
                )
            filename_jet_features_res_this_type = plot_filename.replace(
                rep, f"_jet_features_res_{jet_type}"
            )
            fig_jet_features_res_this_type.suptitle(
                f"Jet features resolution of {jet_type_tex_label} jets", fontsize=16
            )
            fig_jet_features_res_this_type.tight_layout()
            fig_jet_features_res_this_type.savefig(filename_jet_features_res_this_type)
            pylogger.info(
                f"Saved jet features resolution plot to {filename_jet_features_res_this_type}"
            )

            # --------------- Particle-level plots -------------------

            # get the particle features
            ak_part_features_this_type_reco = results_ak_array["part_x_reco"][jet_type_mask]
            ak_part_features_this_type_original = results_ak_array["part_x_original"][
                jet_type_mask
            ]
            # for each field in the particle features, log the min and max value for original and reco
            for field in ak_part_features_this_type_reco.fields:
                reco_min = ak.min(ak_part_features_this_type_reco[field])
                reco_max = ak.max(ak_part_features_this_type_reco[field])
                original_min = ak.min(ak_part_features_this_type_original[field])
                original_max = ak.max(ak_part_features_this_type_original[field])
                pylogger.info(
                    f"Field: {field} Reco: min: {reco_min:.2f} max: {reco_max:.2f} Original: min: {original_min:.2f} max: {original_max:.2f}"
                )

            # --- plot overall distribution of particle features
            labels_dict = {
                feat_name: feat_name for feat_name in ak_part_features_this_type_reco.fields
            }
            update_existing_dict_values(
                labels_dict,
                {
                    "part_pt": "Particle $p_\\mathrm{T}$ [GeV]",
                    "part_etarel": "Particle $\\eta^\\mathrm{rel}$",
                    "part_phirel": "Particle $\\phi^\\mathrm{rel}$",
                    "part_eta": "Particle $\\eta$",
                    "part_phi": "Particle $\\phi$",
                    "part_mass": "Particle mass [GeV]",
                    "part_ptrel": "Particle $p_\\mathrm{T}^\\mathrm{rel}$",
                    "part_charge": "Particle charge",
                    "part_isChargedHadron": "Particle isChargedHadron",
                    "part_isNeutralHadron": "Particle isNeutralHadron",
                    "part_isPhoton": "Particle isPhoton",
                    "part_isElectron": "Particle isElectron",
                    "part_d0val": "Particle $d_0$ [mm]",
                    "part_dzval": "Particle $d_z$ [mm]",
                    "part_d0err": "Particle $\\sigma_{d_0}$ [mm]",
                    "part_dzerr": "Particle $\\sigma_{d_z}$ [mm]",
                },
            )
            bins_dict = {feat_name: None for feat_name in ak_part_features_this_type_reco.fields}
            bins_dict.update(
                {
                    "part_pt": np.linspace(-1, 800, 102),
                    "part_etarel": np.linspace(-1, 1, 100),
                    "part_phirel": np.linspace(-1, 1, 100),
                    "part_eta": np.linspace(-3, 3, 100),
                    "part_phi": np.linspace(-3, 3, 100),
                    "part_mass": np.linspace(-0.1, 1.0, 111),
                    "part_ptrel": np.linspace(-0.1, 1.1, 121),
                    "part_charge": np.linspace(-1.1, 1.1, 56),
                    "part_isChargedHadron": np.linspace(-0.1, 1.1, 61),
                    "part_isNeutralHadron": np.linspace(-0.1, 1.1, 61),
                    "part_isPhoton": np.linspace(-0.1, 1.1, 61),
                    "part_isElectron": np.linspace(-0.1, 1.1, 61),
                    "part_isMuon": np.linspace(-0.1, 1.1, 61),
                    "part_d0val": np.linspace(-0.1, 0.1, 101),
                    "part_dzval": np.linspace(-0.1, 0.1, 101),
                    "part_d0err": np.linspace(-0, 0.1, 100),
                    "part_dzerr": np.linspace(-0, 0.1, 100),
                }
            )
            fig_part_features_this_type, axarr_part_features_this_type = plot_features(
                ak_array_dict={
                    "Original": ak_part_features_this_type_original,
                    "Reco": ak_part_features_this_type_reco,
                },
                names=labels_dict,
                flatten=True,
                ax_rows=math.ceil(len(ak_part_features_this_type_original.fields) / 3),
                bins_dict=bins_dict,
                logscale_features=["part_pt", "part_etarel", "part_phirel"],
            )
            fig_part_features_this_type.suptitle(
                f"Particle features of {jet_type_tex_label} jets", fontsize=16
            )
            fig_part_features_this_type.tight_layout()
            rep = "_overview"
            filename_part_features_this_type = plot_filename.replace(
                rep, f"_particle_features_{jet_type}"
            )
            fig_part_features_this_type.savefig(filename_part_features_this_type)
            pylogger.info(f"Saved particle features plot to {filename_part_features_this_type}")

            # --- plot difference to original of particle features
            ak_part_features_diff_this_type = results_ak_array["part_featurs_diff_to_original"][
                jet_type_mask
            ]
            bins_dict = {feat_name: None for feat_name in ak_part_features_diff_this_type.fields}
            bins_dict.update(
                {
                    "part_pt": np.linspace(-3, 3, 100),  # -5, 5, 100
                    "part_etarel": np.linspace(-0.1, 0.1, 100),
                    "part_phirel": np.linspace(-0.1, 0.1, 100),
                    "part_eta": np.linspace(-0.1, 0.1, 100),
                    "part_phi": np.linspace(-0.1, 0.1, 100),
                    "part_mass": np.linspace(-0.1, 0.1, 100),
                    "part_ptrel": np.linspace(-3, 3, 100),
                    "part_charge": np.linspace(-0.05, 0.05, 100),
                    "part_isChargedHadron": np.linspace(-0.05, 0.05, 100),
                    "part_isNeutralHadron": np.linspace(-0.05, 0.05, 100),
                    "part_isPhoton": np.linspace(-0.05, 0.05, 100),
                    "part_isElectron": np.linspace(-0.05, 0.05, 100),
                    "part_isMuon": np.linspace(-0.05, 0.05, 100),
                    "part_d0val": np.linspace(-0.05, 0.05, 100),
                    "part_dzval": np.linspace(-0.05, 0.05, 100),
                    "part_d0err": np.linspace(-0.05, 0.05, 100),
                    "part_dzerr": np.linspace(-0.05, 0.05, 100),
                }
            )
            # update the bins_dict for the values that have a specific binning
            labels_dict = {
                feat_name: feat_name for feat_name in ak_part_features_diff_this_type.fields
            }
            update_existing_dict_values(
                labels_dict,
                {
                    "part_pt": "Particle $p_\\mathrm{T}^\\mathrm{reco} - p_\\mathrm{T}^\\mathrm{original}$ [GeV]",
                    "part_etarel": "Particle $\\eta^\\mathrm{rel}_\\mathrm{reco} - \\eta^\\mathrm{reco}_\\mathrm{original}$",
                    "part_phirel": "Particle $\\phi^\\mathrm{rel}_\\mathrm{reco} - \\phi^\\mathrm{reco}_\\mathrm{original}$",
                    "part_mass": "Particle $m^\\mathrm{reco} - m^\\mathrm{original}$ [GeV]",
                    "part_charge": "Particle $\\mathrm{charge}^\\mathrm{reco} - \\mathrm{charge}^\\mathrm{original}$",
                    "part_isChargedHadron": "Particle $\\mathrm{isCH}^\\mathrm{reco} - \\mathrm{isCH}^\\mathrm{original}$",
                    "part_isNeutralHadron": "Particle $\\mathrm{isNH}^\\mathrm{reco} - \\mathrm{isNH}^\\mathrm{original}$",
                    "part_isPhoton": "Particle $\\mathrm{isPh}^\\mathrm{reco} - \\mathrm{isPh}^\\mathrm{original}$",
                    "part_isElectron": "Particle $\\mathrm{isEl}^\\mathrm{reco} - \\mathrm{isEl}^\\mathrm{original}$",
                    "part_isMuon": "Particle $\\mathrm{isMu}^\\mathrm{reco} - \\mathrm{isMu}^\\mathrm{original}$",
                    "part_d0val": "Particle $d_0^\\mathrm{reco} - d_0^\\mathrm{original}$ [mm]",
                    "part_dzval": "Particle $d_z^\\mathrm{reco} - d_z^\\mathrm{original}$ [mm]",
                    "part_d0err": "Particle $\\sigma_{d_0}^\\mathrm{reco} - \\sigma_{d_0}^\\mathrm{original}$ [mm]",
                    "part_dzerr": "Particle $\\sigma_{d_z}^\\mathrm{reco} - \\sigma_{d_z}^\\mathrm{original}$ [mm]",
                },
            )

            fig_part_features_res_this_type, axarr_part_features_res_this_type = plot_features(
                ak_array_dict={"Reco. - Original": ak_part_features_diff_this_type},
                names=labels_dict,
                flatten=True,
                ax_rows=math.ceil(len(ak_part_features_diff_this_type.fields) / 3),
                bins_dict=bins_dict,
            )
            # add vertical line at 0
            for i, ax in enumerate(axarr_part_features_res_this_type.flatten()):
                if i == len(ak_part_features_diff_this_type.fields):
                    break
                ax.axvline(0, color="black", linestyle="--", alpha=0.5)

            fig_part_features_res_this_type.suptitle(
                f"Particle features resolution of {jet_type_tex_label} jets", fontsize=16
            )
            fig_part_features_res_this_type.tight_layout()
            rep = "_overview"
            filename_part_features_res_this_type = plot_filename.replace(
                rep, f"_particle_features_res_{jet_type}"
            )
            fig_part_features_res_this_type.savefig(filename_part_features_res_this_type)
            pylogger.info(
                f"Saved particle features resolution plot to {filename_part_features_res_this_type}"
            )

            plt.show()

            # log the plots
            if self.comet_logger is not None:
                for fname in [
                    filename_jet_features_this_type,
                    filename_jet_features_res_this_type,
                    filename_part_features_this_type,
                    filename_part_features_res_this_type,
                ]:
                    self.comet_logger.log_image(
                        fname, name=fname.split("/")[-1], step=trainer.global_step
                    )

            plt.close()

        # calculate the mean abs error of the jet p4s
        ak_mean_abs_err_jet_features = ak_mean(
            ak_abs(results_ak_array["jet_substructure_diff_to_original"])
        )
        # calculate the mean (non-absolute) error of the jet p4s
        ak_mean_err_jet_features = ak_mean(results_ak_array["jet_substructure_diff_to_original"])

        # calculate per-feature mean abs error
        shape = x_recos.shape
        x_recos_reshaped = x_recos.reshape(-1, shape[-1])
        x_originals_reshaped = x_originals.reshape(-1, shape[-1])
        particle_feature_mean_absolute_error = np.mean(
            np.abs(x_recos_reshaped - x_originals_reshaped), axis=1
        )
        particle_feature_mean_error = np.mean(x_recos_reshaped - x_originals_reshaped, axis=1)

        # calculate codebook utilization
        n_codes = pl_module.model.vq_kwargs["num_codes"]
        codebook_utilization = len(np.unique(code_idx)) / n_codes

        # log the mean squared error
        if self.comet_logger is not None:
            # log the mean err of the jet p4s and substructure
            for field in ak_mean_abs_err_jet_features.keys():
                self.comet_logger.log_metric(
                    f"{stage}_mean_abserr_{field}",
                    ak_mean_abs_err_jet_features[field],
                    step=trainer.global_step,
                )
                self.comet_logger.log_metric(
                    f"{stage}_mean_err_{field}",
                    ak_mean_err_jet_features[field],
                    step=trainer.global_step,
                )
            self.comet_logger.log_metric(
                f"{stage}_codebook_utilization", codebook_utilization, step=trainer.global_step
            )
            for i, feature in enumerate(pp_dict.keys()):
                self.comet_logger.log_metric(
                    f"{stage}_mean_abserr_{feature}",
                    particle_feature_mean_absolute_error[i],
                    step=trainer.global_step,
                )
                self.comet_logger.log_metric(
                    f"{stage}_mean_err_{feature}",
                    particle_feature_mean_error[i],
                    step=trainer.global_step,
                )
