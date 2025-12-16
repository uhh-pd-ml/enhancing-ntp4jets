"""Backbone model with different heads."""

import time
from pathlib import Path
from typing import Any, Dict, Tuple

import awkward as ak
import lightning as L
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import vector
from omegaconf import OmegaConf
from tqdm import tqdm

from gabbro.data.loading import safe_load_features_from_ak_array
from gabbro.metrics.utils import calc_acc_from_logits, calc_accuracy
from gabbro.models.classifiers import ClassifierTransformer
from gabbro.models.vqvae import VQVAELightning
from gabbro.utils.arrays import (
    ak_pad,
    ak_select_and_preprocess,
    ak_to_np_stack,
    combine_ak_arrays,
    concat_up_to_n,
    convert_torch_token_sequence_with_stop_token_to_ak,
    fix_padded_logits,
    np_to_ak,
    p4s_from_ptetaphimass,
    replace_masked_positions,
    set_fraction_ones_to_zeros,
)
from gabbro.utils.pylogger import get_pylogger

from .backbone_base import (
    BackboneTransformer,
    MPMHead,
    TokenPredictionHead,
)

vector.register_awkward()

logger = get_pylogger(__name__)


def load_backbone_weights(model, ckpt_path, strict=True):
    """Load the backbone model weights.

    Parameters
    ----------
    model : LightningModule
        The lightning model.
    ckpt_path : str
        Path to the checkpoint file.
    strict : bool, optional
        Whether to load the weights strictly. (default is True)
    """
    # if attached to a trainer, save the state dict from before
    # loading to the default state dict
    if model._trainer is not None:
        logger.info("Saving state dict before loading weights")
        path = f"{model.trainer.default_root_dir}/state_dict_before_loading_backbone_weights.ckpt"
        logger.info(f"Saving state dict to {path}")
        torch.save(model.state_dict(), path)

    logger.info(f"Loading backbone weights from {ckpt_path}")
    device = next(model.parameters()).device
    ckpt = torch.load(ckpt_path, map_location=device)  # nosec
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    # drop all keys containing ".tril." in the state_dict to ensure backwards compatibility
    # with the old `tril` definition https://github.com/joschkabirk/gabbro/pull/167
    state_dict = {key: value for key, value in state_dict.items() if ".tril" not in key}
    # drop all keys *not* starting with "module." or "backbone."
    # state_dict = {key: value for key, value in state_dict.items() if "module." in key}
    # keep only keys starting with "module" and remove the "module."

    logger.info("The following keys are present in the state_dict:")
    for key in state_dict.keys():
        logger.info(f" - {key}")

    # if the model contains the start_token_continuous parameter,
    # we need to load it from the checkpoint as well
    if model.hparams.start_token_type == "trainable":
        if "start_token_continuous" in state_dict:
            model.start_token_continuous.data = state_dict["start_token_continuous"]
            logger.info("Loaded start_token_continuous from checkpoint.")
        else:
            raise ValueError(
                "The checkpoint does not contain the start_token_continuous parameter. "
                "Please make sure to load the correct checkpoint."
            )

    state_dict = {
        ".".join(key.split(".")[1:]): value
        for key, value in state_dict.items()
        if key.startswith("module.") or key.startswith("backbone.")
    }
    # if the current model has keys with ".tril" included, use them!
    state_dict_tril = {
        key: value for key, value in model.backbone.state_dict().items() if "tril" in key
    }
    state_dict.update(state_dict_tril)
    model.backbone.load_state_dict(state_dict, strict=strict)
    logger.info("Backbone weights loaded successfully.")

    if model._trainer is not None:
        logger.info("Saving state dict after loading weights")
        path = f"{model.trainer.default_root_dir}/state_dict_after_loading_backbone_weights.ckpt"
        logger.info(f"Saving state dict to {path}")
        torch.save(model.state_dict(), path)


class BackboneMultiHeadLightning(L.LightningModule):
    """PyTorch Lightning module for training the backbone model with potentially multiple heads."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler = None,
        backbone_cfg: dict = None,
        mpm_head_cfg: dict = None,
        gen_head_cfg: dict = None,
        class_head_cfg: dict = None,
        loss_term_weights: dict = None,
        exclude_padded_values_from_loss: bool = True,
        scheduler_lightning_kwargs: dict = None,
        use_continuous_input: bool = True,
        masked_input_treatment: str = "exclude",
        start_token_type: str = None,
        pos_encoding_type: str = None,
        causal_bidirectional_hybrid: bool = False,
        verbose=False,
        token_dir: str = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.pylogger = get_pylogger(__name__)

        if self.hparams.scheduler_lightning_kwargs is None:
            self.hparams.scheduler_lightning_kwargs = {
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            }

        # --- check inputs ---

        supported_start_token_types = [None, "zero_pad", "trainable"]
        if self.hparams.start_token_type not in supported_start_token_types:
            raise ValueError(
                f"Start token type {self.hparams.start_token_type} is not supported. "
                f"Please choose one of {supported_start_token_types}."
            )

        supported_masked_input_treatments = [
            "exclude",
            "keep",
            "zero_pad",
            "pad_trainable",
        ]
        if self.hparams.masked_input_treatment not in supported_masked_input_treatments:
            raise ValueError(
                f"Masked input treatment {self.hparams.masked_input_treatment} is not supported. "
                f"Please choose one of {supported_masked_input_treatments}."
            )

        supported_pos_encoding = [
            None,
            "sort_by_first_feature_descending",
            "sort_by_first_feature_descending_in_masked_subset",
        ]
        if self.hparams.pos_encoding_type not in supported_pos_encoding:
            raise ValueError(
                f"Positional encoding type {self.hparams.pos_encoding_type} is not supported. "
                f"Please choose one of {supported_pos_encoding}."
            )
        # ------------

        if token_dir is not None:
            self.pylogger.info("Overriding token_dir!")
            self.pylogger.info(f"Old token_dir: {backbone_cfg['token_dir']}")
            backbone_cfg["token_dir"] = token_dir
            self.pylogger.info(f"New token_dir: {backbone_cfg['token_dir']}")

        self.pylogger.info(f"Backbone cfg: {backbone_cfg}")
        self.token_dir = Path(backbone_cfg["token_dir"])

        # this is just used to simplify the `self.log(...)` calls later on
        self.log_dict = dict(on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.vocab_size = backbone_cfg["vocab_size"]

        self.backbone = BackboneTransformer(**backbone_cfg)

        if mpm_head_cfg is None:
            self.head_mpm = None
        elif self.hparams.loss_term_weights.get("mpm", 0) == 0:
            self.head_mpm = None
            self.pylogger.info(
                "MPM head is not used, setting it to None because the loss term weight is 0."
            )
        else:
            if mpm_head_cfg.get("use_old_head_definition", False):
                # remove "dim" entry from transformer_cfg if it exists
                if "dim" in mpm_head_cfg.get("transformer_cfg", {}):
                    mpm_head_cfg["transformer_cfg"].pop("dim", None)
                self.head_mpm = MPMHead(
                    input_dim=backbone_cfg["embedding_dim"],
                    output_dim=backbone_cfg["vocab_size"],
                    hidden_dims=mpm_head_cfg.get("hidden_dims", [128, 128]),
                    transformer_cfg=mpm_head_cfg.get("transformer_cfg", {}),
                    apply_causal_mask=mpm_head_cfg.get("apply_causal_mask", True),
                )
            else:
                self.head_mpm = TokenPredictionHead(
                    input_dim=backbone_cfg["embedding_dim"],
                    vocab_size=backbone_cfg["vocab_size"],
                    n_pred=mpm_head_cfg.get("n_pred", 1),
                    apply_causal_mask=mpm_head_cfg.get("apply_causal_mask", True),
                    transformer_cfg=mpm_head_cfg.get("transformer_cfg", None),
                    unembedding_mlp_cfg=mpm_head_cfg.get("unembedding_mlp_cfg", None),
                )

        if gen_head_cfg is None:
            self.head_gen = None
        elif self.hparams.loss_term_weights.get("gen", 0) == 0:
            self.head_gen = None
            self.pylogger.info(
                "Generation head is not used, setting it to None because the loss term weight is 0."
            )
        else:
            self.head_gen = TokenPredictionHead(
                input_dim=backbone_cfg["embedding_dim"],
                vocab_size=backbone_cfg["vocab_size"],
                n_pred=self.hparams.gen_head_cfg.get("n_pred", 1),
                apply_causal_mask=self.hparams.gen_head_cfg.get("apply_causal_mask", True),
                transformer_cfg=self.hparams.gen_head_cfg.get("transformer_cfg", None),
                unembedding_mlp_cfg=self.hparams.gen_head_cfg.get("unembedding_mlp_cfg", None),
            )
            # determine which mask to use for the multi-token prediction head
            if self.hparams.gen_head_cfg.get("positions_to_include_in_loss") not in [
                "only_survived",
                "only_masked",
                "all",
            ]:
                raise ValueError(
                    f"Unknown prediction type {self.hparams.gen_head_cfg.get('positions_to_include_in_loss')}. "
                    "Please choose either 'only_survived', 'all', or 'only_masked'."
                )

        if self.hparams.class_head_cfg is None:
            self.head_class = None
        elif self.hparams.loss_term_weights.get("class", 0) == 0:
            self.head_class = None
            self.pylogger.info(
                "Classification head is not used, setting it to None because the loss term weight is 0."
            )
        else:
            self.head_class = ClassifierTransformer(
                input_dim=backbone_cfg["embedding_dim"],
                hidden_dim=128,
                num_heads=8,
                num_class_blocks=2,
                num_enc_blocks=0,
                dropout_rate=0,
                fc_params=None,
                n_out_nodes=self.hparams.class_head_cfg.get("n_out_nodes", 2),
                self_attention_model_class="Normformer",
                cross_attention_model_class="ClassAttentionBlock",
                identity_init=False,
            )

        logger.info(
            f"Preprocessing dict for continuous features: {self.backbone.particle_features_dict}"
        )
        if self.head_gen is not None or self.head_mpm is not None:
            self.load_vqvae_weights(**backbone_cfg)

        # initialized masked feature vector as parameter, shape max_seq_len x embedding_dim
        self.masked_feature_vector = nn.Parameter(
            torch.randn(backbone_cfg["max_sequence_len"], backbone_cfg["embedding_dim"])
        )

        # initialize a trainable representation for the start token
        # we initialize it as a zero-padded vector, but allow those values to change
        # over the training (cause zero-padding is not necessarily the best choice)
        if self.hparams.start_token_type in ["trainable", "zero_pad"]:
            self.start_token_continuous = nn.Parameter(
                torch.zeros(self.backbone.part_features_input_dim),
                requires_grad=self.hparams.start_token_type == "trainable",
            )

            # raise an error if the start token is set to "trainable" but token-input is
            # used in the backbone
            if (
                "token_id" in self.backbone.embed_cfg.get("type")
                and self.hparams.start_token_type == "trainable"
            ):
                raise ValueError(
                    "The start token type 'trainable' is not supported when using "
                    "tokenized input in the backbone, as the start token is then "
                    "always 0. --> Please use 'zero_pad' instead."
                )

        # if the mask fraction is 0, we want to detach the masked feature vector
        if backbone_cfg.get("mask_fraction", 0.0) == 0.0:
            self.pylogger.info(
                "Mask fraction is 0 --> detaching the masked feature vector "
                "from the gradient computation."
            )
            self.masked_feature_vector.requires_grad = False

        self.criterion = nn.CrossEntropyLoss()

        self.verbose = verbose
        self.gen_model_pp_dict = None

        self.train_loss_history = []
        self.train_class_target_list = []
        self.train_class_preds_list = []

        # create empty lists to store the validation and test batches
        self.val_loss_list = []

        self.val_input_list = []
        self.val_input_list_jet = []
        self.val_mask_list = []
        self.val_valid_particle_mask_list = []
        self.val_valid_particle_but_masked_mask_list = []
        self.val_valid_particle_after_masking_mask_list = []
        self.val_mpm_pred_list = []
        self.val_mpm_target_list = []
        self.val_gen_target_list = []
        self.val_class_target_list = []
        self.val_class_preds_list = []

        # same for test
        self.test_input_list = []
        self.test_input_list_jet = []
        self.test_mask_list = []
        self.test_valid_particle_mask_list = []
        self.test_valid_particle_but_masked_mask_list = []
        self.test_valid_particle_after_masking_mask_list = []
        self.test_mpm_pred_list = []
        self.test_mpm_target_list = []
        self.test_gen_target_list = []
        self.test_class_target_list = []
        self.test_class_preds_list = []

        self.validation_cnt = 0
        self.validation_output = {}

        self.backbone_weights_path = backbone_cfg.get("backbone_weights_path", "None")

        self.pylogger.info(f"Backbone weights path: {self.backbone_weights_path}")

    def load_vqvae_weights(self, **backbone_cfg):
        """Load the VQ-VAE model weights.

        This is only used for the continuous input case.
        """

        # add the vqvae model to the lightning module but detach its parameters
        # the checkpoint is saved in the main directory of the data directory
        vqvae_ckpt = self.token_dir / "model_ckpt.ckpt"
        self.pylogger.info(f"Loading VQ-VAE model from {vqvae_ckpt}")
        self.vqvae_model = VQVAELightning.load_from_checkpoint(vqvae_ckpt)
        self.max_sequence_len = backbone_cfg["max_sequence_len"]

        self.pylogger.info(
            "Detaching the parameters of the VQ-VAE model from the gradient computation."
        )
        for param in self.vqvae_model.parameters():
            param.requires_grad = False

        # get the preprocessing dict used in the VQ-VAE model training
        vqvae_config_file = self.token_dir / "config.yaml"
        cfg = OmegaConf.load(vqvae_config_file)
        self.vqvae_pp_dict = OmegaConf.to_container(cfg.data.dataset_kwargs_common["feature_dict"])
        if "feature_dict_jet" in cfg.data.dataset_kwargs_common:
            if cfg.data.dataset_kwargs_common["feature_dict_jet"] is not None:
                self.vqvae_pp_dict_jet = OmegaConf.to_container(
                    cfg.data.dataset_kwargs_common["feature_dict_jet"]
                )
            else:
                self.vqvae_pp_dict_jet = None
        # the convention is that tokenized jets have vq-vae token + 1, which means
        # that 0 is the start token and num_codes + 1 is the stop token
        self.stop_token_value = self.vqvae_model.model.vq_kwargs["num_codes"] + 1

    def forward(
        self,
        x,
        valid_particle_mask=None,
        valid_particle_mask_corrupted=None,
        valid_particle_but_masked_mask=None,
        x_jet=None,
        return_logits_only=False,
    ):
        if valid_particle_mask is None:
            valid_particle_mask = torch.ones(x.shape[0], x.shape[1]).to(x.device)
            mask_for_backbone_input = valid_particle_mask
        elif self.hparams.masked_input_treatment == "keep":
            mask_for_backbone_input = valid_particle_mask
        elif self.hparams.masked_input_treatment == "exclude":
            mask_for_backbone_input = valid_particle_mask_corrupted
        elif self.hparams.masked_input_treatment == "zero_pad":
            # multiply the inputs with the corrupted mask to zero out the masked particles
            x = x * valid_particle_mask_corrupted.unsqueeze(-1)
            # then use the full mask, to still allow attention to masked (but padded) particles
            mask_for_backbone_input = valid_particle_mask

        # raise error if only one of valid_particle_mask_corrupted or
        # valid_particle_but_masked_mask is None
        if (valid_particle_mask_corrupted is None) != (valid_particle_but_masked_mask is None):
            raise ValueError(
                "Either both or none of valid_particle_mask_corrupted and "
                "valid_particle_but_masked_mask have to be provided."
            )

        if valid_particle_mask_corrupted is None and valid_particle_but_masked_mask is None:
            valid_particle_mask_corrupted = valid_particle_mask
            valid_particle_but_masked_mask = valid_particle_mask * 0

        if self.hparams.causal_bidirectional_hybrid:
            self.backbone.apply_causal_mask = True
            backbone_out_nonmasked_causal = self.backbone(
                x,
                mask=valid_particle_mask,
                x_jet=x_jet,
            )
            self.backbone.apply_causal_mask = False
            backbone_out = self.backbone(
                x,
                mask=mask_for_backbone_input,
                x_jet=x_jet,
            )
            # make sure to set it back to what is specified in the config
            self.backbone.apply_causal_mask = self.hparams.backbone_cfg["apply_causal_mask"]
        else:
            backbone_out = self.backbone(
                x,
                mask=mask_for_backbone_input,
                x_jet=x_jet,
            )

        if self.hparams.pos_encoding_type is None:
            pos_encoding_type = None
            pos_encoding_feature = None
        elif self.hparams.pos_encoding_type == "sort_by_first_feature_descending":
            pos_encoding_type = "sort_descending_all"
            pos_encoding_feature = x[:, :, 0]
        elif self.hparams.pos_encoding_type == "sort_by_first_feature_descending_in_masked_subset":
            pos_encoding_type = "sort_descending_in_masked_subset"
            pos_encoding_feature = x[:, :, 0]

        # this masking helper function modifies the backbone output in-place
        replace_masked_positions(
            x=backbone_out,
            mask_is_valid=valid_particle_mask,
            mask_is_valid_corrupted=valid_particle_mask_corrupted,
            mask_is_valid_but_masked=valid_particle_but_masked_mask,
            vectors_to_insert=self.masked_feature_vector,
            pos_encoding_type=pos_encoding_type,
            pos_encoding_feature=pos_encoding_feature,
        )

        if self.head_class is not None:
            backbone_out_for_class = backbone_out.clone()
            if self.hparams.class_head_cfg.get("detach_backbone_grad_before_class", False):
                backbone_out_for_class = backbone_out_for_class.detach()
            logits_class = self.head_class(backbone_out_for_class, mask=valid_particle_mask)
        else:
            logits_class = None

        if self.head_mpm is not None:
            if self.hparams.mpm_head_cfg.get("use_old_head_definition", False):
                logits_mpm = self.head_mpm(backbone_out, mask=valid_particle_mask)
            else:
                logits_mpm = self.head_mpm(backbone_out, mask=valid_particle_mask)[..., 0]
        else:
            logits_mpm = None

        if self.head_gen is not None:
            # check which mask to use for the multi-token prediction head
            if self.hparams.causal_bidirectional_hybrid:
                logits_gen = self.head_gen(backbone_out_nonmasked_causal, mask=valid_particle_mask)
            else:
                logits_gen = self.head_gen(backbone_out, mask=valid_particle_mask)
        else:
            logits_gen = None

        if self.verbose:
            self.pylogger.info("Logits shape: ", logits_mpm.shape)

        return logits_gen, logits_mpm, logits_class

    def model_step(self, batch):
        """Perform a single model step on a batch of data.

        Parameters
        ----------
        batch : dict
            A batch of data as a dictionary containing the input and target tensors,
            as well as the mask.
        """

        if self.head_gen is not None or self.head_mpm is not None:
            # check that the shape of `batch["part_labels"]` is as expected
            if batch["part_labels"].shape[-1] != 2:
                raise ValueError(
                    "Invalid shape for part_labels. Shape in currently used particle labels "
                    f"tensor is {batch['part_labels'].shape}. Should have exactly 2 "
                    "particle labels. The first is used for MPM and the second is "
                    "used for generation."
                )

        X = batch["part_features"].clone()
        X_jet = batch["jet_features"] if self.backbone.jet_features_input_dim > 0 else None
        mask = batch["part_mask"].clone()

        input = X

        # if a start token is used, then the input features at the first position
        # have to be set to the start token continuous vector
        if self.hparams.start_token_type in ["trainable", "zero_pad"]:
            input[:, 0, : self.start_token_continuous.shape[0]] = self.start_token_continuous

        mask = mask.int()

        # ------ masking of input particles ------
        valid_particle_mask = mask.clone()
        valid_particle_after_masking_mask = set_fraction_ones_to_zeros(
            mask,
            fraction=self.hparams.backbone_cfg.get("mask_fraction", 0.15),
        )
        valid_particle_but_masked_mask = valid_particle_mask * (
            1 - valid_particle_after_masking_mask
        )

        logits_gen, logits_mpm, logits_class = self.forward(
            input,
            valid_particle_mask_corrupted=valid_particle_after_masking_mask,
            valid_particle_mask=valid_particle_mask,
            valid_particle_but_masked_mask=valid_particle_but_masked_mask,
            x_jet=X_jet,
        )

        # ------ calculate mpm prediction loss ------

        targets_mpm = batch["part_labels"].clone().long()[:, :, 0]
        targets_clone_mpm = targets_mpm.clone()

        if self.head_mpm is not None:
            if self.hparams.exclude_padded_values_from_loss:
                logits_mpm = fix_padded_logits(
                    logits_mpm, valid_particle_but_masked_mask.bool(), factor=1e6
                )
                targets_mpm = targets_mpm * valid_particle_but_masked_mask

            B, T, C = logits_mpm.shape
            logits_reshaped_mpm = logits_mpm.view(B * T, C)
            targets_reshaped_mpm = targets_mpm.contiguous().view(B * T)

            loss_mpm = self.criterion(logits_reshaped_mpm, targets_reshaped_mpm)

            max_len_in_batch = logits_mpm.size(1)
            default_token_indices_to_calc = [0, 1, 2, 3, 10, 20, 30, 40, 50, 90]
            token_indices_to_calc = [
                index for index in default_token_indices_to_calc if index < max_len_in_batch
            ]

            acc_dict_mpm = calc_acc_from_logits(
                logits=logits_mpm,
                targets=targets_mpm,
                # only calculate accuracy for the particles that were masked
                mask=valid_particle_but_masked_mask,
                token_indices_to_calc=token_indices_to_calc,
            )
        else:
            logits_reshaped_mpm = None
            targets_reshaped_mpm = None
            loss_mpm = torch.tensor(0.0)
            acc_dict_mpm = {}

        # ------ calculate class prediction loss ------

        if self.head_class is not None:
            labels_class = F.one_hot(
                batch["jet_type_labels"].squeeze(),
                num_classes=self.head_class.n_out_nodes,
            ).float()
            loss_class = self.criterion(logits_class, labels_class)
        else:
            labels_class = None
            loss_class = torch.tensor(0)

        # -------- calculate generation loss --------

        if self.head_gen is not None:
            targets_gen = batch["part_labels"].clone().long()[:, :, 1]
            targets_gen_before_masking = targets_gen.clone()

            # compute the logits (i.e. the predictions for the next token)
            logits_mtp = logits_gen.clone()

            # calculate next-token-prediction loss
            logits_gen = logits_mtp[..., 0]

            # fix the logits for the multi-token prediction head
            if self.hparams.exclude_padded_values_from_loss:
                # determine which mask to use for the multi-token prediction head
                if (
                    self.hparams.gen_head_cfg.get("positions_to_include_in_loss")
                    == "only_survived"
                ):
                    mask_for_gen_loss = valid_particle_after_masking_mask
                elif (
                    self.hparams.gen_head_cfg.get("positions_to_include_in_loss") == "only_masked"
                ):
                    mask_for_gen_loss = valid_particle_but_masked_mask
                elif self.hparams.gen_head_cfg.get("positions_to_include_in_loss") == "all":
                    mask_for_gen_loss = valid_particle_mask
                else:
                    raise ValueError(
                        f"Unknown prediction type {self.hparams.gen_head_cfg.get('positions_to_include_in_loss')}. "
                        "Please choose either 'only_survived', 'all', or 'only_masked'."
                    )
                targets_gen = targets_gen * mask_for_gen_loss
                logits_gen = fix_padded_logits(
                    logits_gen,
                    mask=mask_for_gen_loss.bool(),
                    factor=1e6,
                )

            B, T, C = logits_gen.shape
            loss_next_token_prediction = self.criterion(
                logits_gen.reshape(B * T, C),
                targets_gen.contiguous().reshape(B * T),
            )

            # calculate multi-token prediction loss, i.e. predictions for the next tokens
            # after the first one
            start_indices = list(range(logits_mtp.size(-1)))
            end_indices = list(range(logits_mtp.size(-1)))
            start_indices[0] = end_indices[0] = None

            for i in range(1, len(start_indices)):
                end_indices[i] = int(-i)

            mtp_losses = {}

            if self.hparams.exclude_padded_values_from_loss:
                targets_gen = targets_gen * mask_for_gen_loss

            for i in range(1, len(start_indices)):
                logits_ith_token = logits_mtp[:, : end_indices[i], :, i]

                if self.hparams.exclude_padded_values_from_loss:
                    # fix the logits for the multi-token prediction head
                    logits_ith_token = fix_padded_logits(
                        logits_ith_token,
                        mask=mask_for_gen_loss[:, start_indices[i] :].bool(),
                        factor=1e6,
                    )

                mtp_losses[f"loss_ntp_{i + 1}"] = self.criterion(
                    logits_ith_token.reshape(B * (T - i), C),
                    targets_gen[:, start_indices[i] :].contiguous().reshape(B * (T - i)),
                )

            loss_gen = (loss_next_token_prediction + sum(mtp_losses.values())) / logits_mtp.size(
                -1
            )

            max_len_in_batch = logits_gen.size(1)
            default_token_indices_to_calc = [0, 1, 2, 3, 10, 20, 30, 40, 50, 100]
            token_indices_to_calc = [
                index for index in default_token_indices_to_calc if index < max_len_in_batch
            ]

            acc_dict_gen = calc_acc_from_logits(
                logits_gen, targets_gen, mask, token_indices_to_calc=token_indices_to_calc
            )
        else:
            logits_gen = None
            targets_gen = None
            targets_gen_before_masking = None
            loss_gen = torch.tensor(0.0)
            acc_dict_gen = {}

        # print(f"Targets gen [0, :] = {targets_gen[0, :]}")
        # print(f"Targets MPM [0, :] = {targets_mpm[0, :]}")

        # -------- calculate total loss --------
        loss = (
            self.hparams.loss_term_weights.get("mpm") * loss_mpm
            + self.hparams.loss_term_weights.get("gen") * loss_gen
            + self.hparams.loss_term_weights.get("class") * loss_class
        )

        return {
            # --- losses ---
            "loss": loss,
            "loss_mpm": loss_mpm,
            "loss_class": loss_class,
            "loss_gen": loss_gen,
            # loss terms multiplied with the weights
            "loss_mpm_weighted": self.hparams.loss_term_weights.get("mpm") * loss_mpm,
            "loss_gen_weighted": self.hparams.loss_term_weights.get("gen") * loss_gen,
            "loss_class_weighted": self.hparams.loss_term_weights.get("class") * loss_class,
            # --- general tensors ---
            "X": X,
            "mask": mask,
            # --- MPM ---
            "logits_reshaped_mpm": logits_reshaped_mpm,
            "targets_reshaped_mpm": targets_reshaped_mpm,
            "acc_dict_mpm": acc_dict_mpm,
            "logits_mpm": logits_mpm,
            "targets_mpm": targets_clone_mpm,
            "valid_particle_mask": valid_particle_mask,
            "valid_particle_but_masked_mask": valid_particle_but_masked_mask,
            "valid_particle_after_masking_mask": valid_particle_after_masking_mask,
            # --- classification ---
            "logits_class": logits_class,
            "targets_class": labels_class,
            # --- generation ---
            "logits_gen": logits_gen,
            "targets_gen": targets_gen,
            "targets_gen_before_masking": targets_gen_before_masking,
            "acc_dict_gen": acc_dict_gen,
        }

    def _shared_step(self, batch, stage: str) -> torch.Tensor:
        """Shared logic for training, validation, and test steps."""
        model_step_output = self.model_step(batch)

        if self.head_class is not None:
            acc_class = calc_accuracy(
                preds=torch.softmax(model_step_output["logits_class"], dim=1)
                .float()
                .detach()
                .cpu()
                .numpy(),
                labels=model_step_output["targets_class"].float().detach().cpu().numpy(),
            )
            self.log(f"{stage}_acc_class", acc_class, **self.log_dict)

        self.log(f"{stage}_loss", model_step_output["loss"].item(), **self.log_dict)
        self.log(f"{stage}_loss_mpm", model_step_output["loss_mpm"].item(), **self.log_dict)
        self.log(f"{stage}_loss_gen", model_step_output["loss_gen"].item(), **self.log_dict)
        self.log(f"{stage}_loss_class", model_step_output["loss_class"].item(), **self.log_dict)
        self.log(
            f"{stage}_loss_weighted_mpm",
            model_step_output["loss_mpm_weighted"].item(),
            **self.log_dict,
        )
        self.log(
            f"{stage}_loss_weighted_gen",
            model_step_output["loss_gen_weighted"].item(),
            **self.log_dict,
        )
        self.log(
            f"{stage}_loss_weighted_class",
            model_step_output["loss_class_weighted"].item(),
            **self.log_dict,
        )

        for key, value in model_step_output["acc_dict_mpm"].items():
            self.log(f"{stage}_mpm_" + key, value, **self.log_dict)

        for key, value in model_step_output["acc_dict_gen"].items():
            self.log(f"{stage}_gen_" + key, value, **self.log_dict)

        self._collect_batch_data(batch, model_step_output, stage)

        return model_step_output["loss"]

    def _collect_batch_data(self, batch, model_step_output, stage: str) -> None:
        """Collect batch data for validation or test stages."""

        class_target_list = getattr(self, f"{stage}_class_target_list")
        class_preds_list = getattr(self, f"{stage}_class_preds_list")
        if self.head_class is not None:
            class_target_list.append(
                model_step_output["targets_class"].float().detach().cpu().numpy()
            )
            class_preds_list.append(
                torch.softmax(model_step_output["logits_class"], dim=1)
                .float()
                .detach()
                .cpu()
                .numpy()
            )

        if stage == "train":
            return

        if self.head_gen is None and self.head_mpm is None:
            return

        input_list = getattr(self, f"{stage}_input_list")
        mask_list = getattr(self, f"{stage}_mask_list")
        valid_particle_mask_list = getattr(self, f"{stage}_valid_particle_mask_list")
        valid_particle_but_masked_mask_list = getattr(
            self, f"{stage}_valid_particle_but_masked_mask_list"
        )
        valid_particle_after_masking_mask_list = getattr(
            self, f"{stage}_valid_particle_after_masking_mask_list"
        )
        mpm_pred_list = getattr(self, f"{stage}_mpm_pred_list")
        mpm_target_list = getattr(self, f"{stage}_mpm_target_list")
        gen_target_list = getattr(self, f"{stage}_gen_target_list")

        input_list.append(batch["part_features"].float().detach().cpu().numpy())
        mask_list.append(batch["part_mask"].float().detach().cpu().numpy())
        if self.head_mpm is not None:
            valid_particle_mask_list.append(
                model_step_output["valid_particle_mask"].float().detach().cpu().numpy()
            )
            valid_particle_but_masked_mask_list.append(
                model_step_output["valid_particle_but_masked_mask"].float().detach().cpu().numpy()
            )
            valid_particle_after_masking_mask_list.append(
                model_step_output["valid_particle_after_masking_mask"]
                .float()
                .detach()
                .cpu()
                .numpy()
            )
            # calculate argmax of logits, but don't allow first or last token id
            # mpm_preds = torch.argmax(model_step_output["logits_mpm"][..., 1:-1], dim=-1) + 1
            # sample from the logits but don't allow first or last token id
            mpm_probs = F.softmax(model_step_output["logits_mpm"][..., 1:-1], dim=-1)
            mpm_preds = (
                torch.multinomial(mpm_probs.view(-1, mpm_probs.size(-1)), 1).view(
                    mpm_probs.size(0), mpm_probs.size(1)
                )
                + 1
            )
            mpm_pred_list.append(mpm_preds.float().detach().cpu().numpy())
            mpm_target_list.append(model_step_output["targets_mpm"].float().detach().cpu().numpy())

        if self.head_gen is not None:
            gen_target_list.append(
                model_step_output["targets_gen_before_masking"].float().detach().cpu().numpy()
            )

        if self.backbone.jet_features_input_dim > 0:
            getattr(self, f"{stage}_input_list_jet").append(
                batch["jet_features"].detach().cpu().numpy()
            )

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set."""
        return self._shared_step(batch, "train")

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        self._shared_step(batch, "val")

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        self._shared_step(batch, "test")

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        self.pylogger.info("`on_train_start` called.")
        if self.backbone_weights_path is not None:
            if self.backbone_weights_path != "None":
                load_backbone_weights(self, self.backbone_weights_path, strict=True)

    def on_train_epoch_start(self) -> None:
        """Reset the training loss history and class prediction lists at the start of each epoch."""
        self.pylogger.info("`on_train_epoch_start` called.")
        self.epoch_train_start_time = time.time()
        self.train_class_target_list = []
        self.train_class_preds_list = []

    def on_train_epoch_end(self):
        logger.info("`on_train_epoch_end` called.")
        logger.info(f"Epoch {self.trainer.current_epoch} finished.")
        self.epoch_train_end_time = time.time()
        if hasattr(self, "epoch_train_start_time"):
            duration = (self.epoch_train_end_time - self.epoch_train_start_time) / 60
            self.log(
                "epoch_duration_minutes",
                duration,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )
            if self.train_loss_history:
                self.pylogger.info(
                    f"Epoch {self.trainer.current_epoch} finished in {duration:.1f} minutes. "
                    f"Rank: {self.global_rank}"
                )

    # on val/test epoch start, reset the lists to empty
    def on_validation_epoch_start(self) -> None:
        self.pylogger.info("`on_validation_epoch_start` called.")
        self.val_input_list = []
        self.val_mask_list = []
        self.val_valid_particle_mask_list = []
        self.val_valid_particle_but_masked_mask_list = []
        self.val_valid_particle_after_masking_mask_list = []
        self.val_input_list_jet = []
        self.val_mpm_pred_list = []
        self.val_mpm_target_list = []
        self.val_gen_target_list = []
        self.val_class_target_list = []
        self.val_class_preds_list = []

    def on_test_epoch_start(self) -> None:
        self.pylogger.info("`on_test_epoch_start` called.")
        self.test_input_list = []
        self.test_mask_list = []
        self.test_valid_particle_mask_list = []
        self.test_valid_particle_but_masked_mask_list = []
        self.test_valid_particle_after_masking_mask_list = []
        self.test_input_list_jet = []
        self.test_mpm_pred_list = []
        self.test_mpm_target_list = []
        self.test_gen_target_list = []
        self.test_class_target_list = []
        self.test_class_preds_list = []

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configures optimizers and learning-rate schedulers to be used for training."""
        logger.info("`configure_optimizers` called.")

        # get all parameters that contain "backbone" in their name
        backbone_params = [param for name, param in self.named_parameters() if "backbone" in name]
        # get all parameters that do not contain "backbone" in their name
        other_params = [param for name, param in self.named_parameters() if "backbone" not in name]

        # extract the base learning rate from the optimizer
        base_lr = self.hparams.optimizer.keywords.get("lr")
        logger.info(f"Base learning rate: {base_lr}")

        if self.hparams.get("backbone_lr_factor") is not None:
            backbone_lr = base_lr * self.hparams.backbone_lr_factor
            logger.info(f"Learning rate factor for backbone: {self.hparams.backbone_lr_factor}")
            logger.info(f"--> Backbone learning rate: {backbone_lr}")
            optimizer = self.hparams.optimizer(
                [
                    {"params": backbone_params, "lr": backbone_lr},
                    {"params": other_params, "lr": base_lr},
                ]
            )
        else:
            optimizer = self.hparams.optimizer(self.parameters())

        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    **self.hparams.scheduler_lightning_kwargs,
                },
            }

        return {"optimizer": optimizer}

    # -------------------------------------------
    # Generation methods
    def generate_n_jets_batched(
        self,
        n_jets,
        batch_size,
        saveas=None,
        seed=None,
        x_jet=None,
        start_token=0,
        **kwargs,
    ):
        """Generate jets in batches.

        Parameters
        ----------
        n_jets : int
            Number of jets to generate.
        batch_size : int
            Batch size to use during generation (use as large as possible with memory.)
        start_token: int, optional
            If a custom start token should be used instead of 0.
        saveas : str, optional
            Path to save the generated jets to (in parquet format). (default is None)
        x_jet : torch.Tensor, optional
            The jet features to use as input. (default is None)

        Returns
        -------
        ak.Array
            The generated jets (i.e. their token ids, in the shape (n_jets, <var>).
        """

        self.pylogger.info(f"Generating {n_jets} jets in batches.")

        if x_jet is not None:
            self.pylogger.info("Using x_jet as input for generation.")
            self.pylogger.info(f"x_jet shape: {x_jet.shape}")
            if x_jet.size(0) < n_jets:
                self.pylogger.warning(
                    f"x_jet has fewer jets than specified n_jets={n_jets}. "
                    f"Will generate the only {x_jet.size(0)} jets."
                )
                n_jets = x_jet.size(0)

            if n_jets < batch_size:
                batch_size = n_jets
                self.pylogger.warning(
                    f"Batch size is larger than n_jets. Setting batch size to {batch_size}."
                )
            n_batches = n_jets // batch_size

        else:
            n_batches = (
                n_jets // batch_size + 1 if n_jets % batch_size != 0 else n_jets // batch_size
            )

        generated_jets = []

        self.start_token = start_token
        if seed is not None:
            L.seed_everything(seed)

        self.pylogger.info(
            f"Generating {n_batches * batch_size} jets in {n_batches} batches of size {batch_size}, starting from start token {self.start_token}"
        )

        for i in tqdm(range(n_batches)):
            if x_jet is not None:
                x_jet_batch = x_jet[batch_size * i : batch_size * (i + 1)]
            else:
                x_jet_batch = None
            gen_batch_ak = self.generate_batch_continuous(batch_size, x_jet=x_jet_batch)
            generated_jets.append(gen_batch_ak)

        # concatenate the generated batches
        generated_jets = ak.concatenate(generated_jets)[:n_jets]

        if saveas is not None:
            self.pylogger.info(f"Saving generated jets to {saveas}")
            ak.to_parquet(generated_jets, saveas)

        return generated_jets

    @torch.no_grad()
    def generate_batch_continuous(
        self,
        batch_size,
        return_more=False,
        verbose=False,
        x_jet=None,
    ):
        """Generate a batch of jet constituents autoregressively.

        Parameters
        ----------
        batch_size : int
            Number of jets to generate.
        return_more : bool, optional
            Whether to return more information than just the generated jets. (default is False)
        verbose : bool, optional
            Whether to print more information during generation. (default is False)
        x_jet : torch.Tensor, optional
            The jet features to use as input. (default is None)

        Returns
        -------
        ak.Array
            The generated jets (i.e. their token ids, in the shape (batch_size, <var>).
        """
        if x_jet is not None:
            assert x_jet.size(0) == batch_size, "x_jet must have the same batch size as batch_size"

        if verbose:
            self.pylogger.info(f"Generating a batch of {batch_size} jets.")

        # idx is (B, T) array of indices in the current context, initialized with the start token
        # thus idx has shape (B, 1) at the beginning
        device = next(self.backbone.parameters()).device  # get the device of the model
        # we start with a zero-padded tensor of the continuous features
        continuous_input = self.start_token_continuous.unsqueeze(0).repeat(batch_size, 1, 1)

        # if interactions are used, make sure that the continuous input is extended
        if self.backbone.interaction_cfg is not None:
            p4_start = torch.zeros(batch_size, 1, 4, device=device)
            continuous_input = torch.cat([continuous_input, p4_start], dim=-1)

        for i in range(self.max_sequence_len):
            # get the predictions for the next token
            if verbose:
                self.pylogger.info(f"Continuous input shape: {continuous_input.shape}")
            # initialize ones masks for all cases
            valid_particle_mask = torch.ones(
                batch_size, continuous_input.shape[1], dtype=torch.bool, device=device
            )
            valid_particle_after_masking_mask = valid_particle_mask.clone()
            valid_particle_but_masked_mask = torch.zeros_like(valid_particle_mask)
            logits, _, _ = self.forward(
                continuous_input,
                valid_particle_mask_corrupted=valid_particle_after_masking_mask,
                valid_particle_mask=valid_particle_mask,
                valid_particle_but_masked_mask=valid_particle_but_masked_mask,
                x_jet=x_jet,
            )

            # just use next-token prediction logits, even if we have a multi-token prediction head
            logits = logits[..., 0]

            if verbose:
                self.pylogger.info(f"Logit shape input for generation: {logits.shape}")

            # only look at next-token prediction of last token
            logits = logits[:, -1, :]  # (B, T, C) becomes (B, C)

            # we want to exclude token-id 0 in the prediction, because
            # we'll shift them by -1 later on to get the VQ-VAE tokens, and
            # if we predict 0, we'll get an invalid "-1" token
            probs = F.softmax(logits[:, 1:], dim=-1)  # (B, C-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx_next = idx_next + 1  # shift the indices by +1 to get the correct token
            # append sampled index to the running sequence
            if i == 0:
                idx = idx_next
            else:
                idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)

            if verbose:
                self.pylogger.info(f"max value in predicted idx: {torch.max(idx)}")

            if "token_id" in self.backbone.embed_cfg.get("type"):
                # using token ids as input --> no on-the-fly reconstruction of continuous features
                continuous_input = torch.cat(
                    [
                        continuous_input,
                        idx_next.unsqueeze(-1),
                    ],
                    dim=1,
                )
                continue

            # convert the token-ids to an awkward array
            gen_batch_tokens_ak = convert_torch_token_sequence_with_stop_token_to_ak(
                idx,
                stop_value=(self.vocab_size - 1),
            )
            if self.verbose:
                self.pylogger.info(
                    "Min and max length of gen_batch_tokens_ak: "
                    f"{ak.min(ak.num(gen_batch_tokens_ak))}, {ak.max(ak.num(gen_batch_tokens_ak))}"
                )
                self.pylogger.info(f"Unique generated idx: {torch.unique(idx).size()[0]}")

            # reconstruct the continuous features from the token-ids
            gen_batch_continuous_ak = self.vqvae_model.reconstruct_ak_tokens(
                gen_batch_tokens_ak - 1,  # -1 because the VQ-VAE tokens are shifted by -1
                pp_dict=self.vqvae_pp_dict,
                hide_pbar=True,
            )

            if self.hparams.backbone_cfg.interaction_cfg is not None:
                # in case interactions are used, we potentially need the
                # p4s of the particles (which are not used as node-input then)
                # line below assumes `part_pt`, `part_etarel`, `part_phirel`
                # and optionally `part_mass`
                # --> should be made more general
                p4s_vector = p4s_from_ptetaphimass(gen_batch_continuous_ak)
                p4s_ak = ak.Array(
                    {
                        "part_px_centered": p4s_vector.px,
                        "part_py_centered": p4s_vector.py,
                        "part_pz_centered": p4s_vector.pz,
                        # energy where it's larger than p, otherwise use p
                        "part_energy_centered": ak.where(
                            p4s_vector.E > p4s_vector.p,
                            p4s_vector.E,
                            p4s_vector.p,
                        ),
                    }
                )
                # raise error if those features are *not* in the preprocessing dict
                if not all(f in self.backbone.particle_features_dict for f in p4s_ak.fields):
                    raise NotImplementedError(
                        "The current implementation (in the generative model) expects the "
                        "interaction input features to be"
                        " `part_px_centered`, `part_py_centered`, `part_pz_centered`, "
                        "`part_energy_centered` but they are not in the feature dict."
                        " The feature dict includes: "
                        f"{list(self.backbone.particle_features_dict.keys())}"
                    )
                # print last particle px/py/pz/E
                # print(
                #     f"Last particle px/py/pz/E: {p4s_vector.px[0, -1]}, {p4s_vector.py[0, -1]}, {p4s_vector.pz[0, -1]}, {p4s_vector.E[0, -1]}"
                # )
                # print()
                gen_batch_continuous_ak = combine_ak_arrays(
                    gen_batch_continuous_ak,
                    p4s_ak,
                )

            # add zero-padded features if using features that are not part of the VQ-VAE
            gen_batch_continuous_ak = safe_load_features_from_ak_array(
                ak_array=gen_batch_continuous_ak,
                features=self.backbone.particle_features_dict.keys(),
                load_zeros_if_not_present=True,
                verbose=False,
            )

            # apply the preprocessing before feeding it into the model
            pp_dict = self.backbone.particle_features_dict
            # add p4s to the gen_batch_continuous_ak
            gen_batch_continuous_ak_preprocessed = ak_select_and_preprocess(
                ak_array=gen_batch_continuous_ak,
                pp_dict=pp_dict,
                suppress_warnings=True,
            )
            if verbose:
                self.pylogger.info(gen_batch_continuous_ak_preprocessed)

            # convert to torch tensor to put into the model
            # pad the continuous features to the maximum sequence length
            gen_batch_continuous_ak_padded, gen_batch_mask = ak_pad(
                gen_batch_continuous_ak_preprocessed,
                maxlen=i + 1,
                return_mask=True,
            )
            continuous_input_new = torch.from_numpy(
                ak_to_np_stack(
                    gen_batch_continuous_ak_padded,
                    names=gen_batch_continuous_ak_preprocessed.fields,
                )
            ).to(self.device)

            # add the latest cotinuous column to the continuous_input tensor
            continuous_input = torch.cat(
                [continuous_input, continuous_input_new[:, -1:, :]], dim=1
            )

        if "token_id" in self.backbone.embed_cfg.get("type"):
            # convert to ak and reconstruct the continuous features
            gen_batch_tokens_ak = convert_torch_token_sequence_with_stop_token_to_ak(
                idx,
                stop_value=(self.vocab_size - 1),
            )
            if self.verbose:
                self.pylogger.info(
                    "Min and max length of gen_batch_tokens_ak: "
                    f"{ak.min(ak.num(gen_batch_tokens_ak))}, {ak.max(ak.num(gen_batch_tokens_ak))}"
                )
                self.pylogger.info(f"Unique generated idx: {torch.unique(idx).size()[0]}")
            # reconstruct the continuous features from the token-ids
            gen_batch_continuous_ak = self.vqvae_model.reconstruct_ak_tokens(
                gen_batch_tokens_ak - 1,  # -1 because the VQ-VAE tokens are shifted by -1
                pp_dict=self.vqvae_pp_dict,
                hide_pbar=True,
            )

        # combine the token-ids and the continuous features
        gen_batch_particles_ak = combine_ak_arrays(
            ak.Array({"part_token_id": gen_batch_tokens_ak}),
            gen_batch_continuous_ak,
        )

        if return_more:
            return (
                gen_batch_particles_ak,
                gen_batch_mask,
                idx,
            )

        return gen_batch_particles_ak

    def convert_valtest_batches_to_ak(self, stage, max_n=None):
        """Convert the collected validation loop batches to awkward arrays.

        Parameters
        ----------
        stage : str
            Either 'val' or 'test'.
        max_n : int, optional
            Maximum number of elements to include along the first axis. If None, use all.

        Returns
        -------
        ak.Array
            The validation input as an awkward array.
        """
        if stage == "val":
            input_list = self.val_input_list
            target_gen_list = self.val_gen_target_list
            mask_list = self.val_mask_list
        elif stage == "test":
            input_list = self.test_input_list
            target_gen_list = self.test_gen_target_list
            mask_list = self.test_mask_list
        else:
            raise ValueError(f"Stage {stage} not recognized.")

        input = concat_up_to_n(input_list, max_n)
        target_gen = concat_up_to_n(target_gen_list, max_n)
        mask = concat_up_to_n(mask_list, max_n)

        print(f"Input shape: {input.shape}")
        print(f"Target gen shape: {target_gen.shape if target_gen is not None else None}")
        print(target_gen[:, :-1, None].shape)
        print(input[:, 1:].shape)

        if "token_id" in self.backbone.embed_cfg.get("type"):
            # input is just the token ids, so we have to reconstruct them once
            # like in the generation case
            token_ids = (
                np_to_ak(
                    input[:, 1:, :],
                    names=self.backbone.particle_features_dict.keys(),
                    mask=mask[:, 1:],
                )["part_token_id_without_last"]
                - 1
            )  # -1 because the VQ-VAE tokens are shifted by -1

            # reconstruct the continuous features from the token-ids
            ak_arr_input = self.vqvae_model.reconstruct_ak_tokens(
                token_ids,  # -1 because the VQ-VAE tokens are shifted by -1
                pp_dict=self.vqvae_pp_dict,
                hide_pbar=True,
            )
            # combine the token-ids and the continuous features
            ak_arr_input = combine_ak_arrays(
                ak.Array({"part_token_id": token_ids}),
                ak_arr_input,
            )
            return ak_arr_input

        ak_arr_input = np_to_ak(
            # exclude the start token / dummy continuous feature at the beginning
            np.concatenate(
                [
                    # remove the stop token
                    # (by cropping the slicing mask a few lines below)
                    target_gen[:, :-1, None],
                    input[:, 1:],  # exclude zero-padded dummy continuous feature at beginning
                ],
                axis=-1,
            ),
            names=["part_token_id"] + list(self.backbone.particle_features_dict.keys()),
            mask=mask[:, 1:],
        )
        # invert the preprocessing
        pp_dict = {"part_token_id": {}} | self.backbone.particle_features_dict
        ak_arr_input = ak_select_and_preprocess(ak_arr_input, pp_dict, inverse=True)
        return ak_arr_input
