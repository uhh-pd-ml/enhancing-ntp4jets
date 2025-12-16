import time
from typing import Any, Dict, Tuple

import awkward as ak
import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import vector
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from vqtorch.nn import VectorQuant

from gabbro.models.transformer import MLP, NormformerStack, Transformer
from gabbro.utils.arrays import (
    ak_pad,
    ak_select_and_preprocess,
    ak_to_np_stack,
    get_causal_mask,
    np_to_ak,
)
from gabbro.utils.pylogger import get_pylogger

vector.register_awkward()

logger = get_pylogger(__name__)


class VQVAEMLP(torch.nn.Module):
    def __init__(
        self,
        input_features_dict: Dict[str, Any] = None,
        input_dim: int = None,
        conditional_dim: int = 0,
        latent_dim=4,
        encoder_layers=None,
        decoder_layers=None,
        vq_kwargs={},
        **kwargs,
    ):
        """Initializes the VQ-VAE model.

        Parameters
        ----------
        input_features_dict : Dict[str, Any]
            Dictionary containing the input features and their preprocessing information.
        input_dim : int, optional
            The dimension of the input data. If not provided, it is inferred from
            `input_features_dict`.
        conditional_dim : int, optional
            The dimension of the conditional data. The default is 0.
        codebook_size : int, optional
            The size of the codebook. The default is 8.
        embed_dim : int, optional
            The dimension of the embedding space. The default is 2.
        input_dim : int, optional
            The dimension of the input data. The default is 2.
        encoder_layers : list, optional
            List of integers representing the number of units in each encoder layer.
            If None, a default encoder with a single linear layer is used. The default is None.
        decoder_layers : list, optional
            List of integers representing the number of units in each decoder layer.
            If None, a default decoder with a single linear layer is used. The default is None.
        """

        super().__init__()
        self.vq_kwargs = vq_kwargs
        self.embed_dim = latent_dim
        self.input_features_dict = input_features_dict
        if input_features_dict is not None:
            self.input_dim = len(self.input_features_dict)
        elif input_dim is not None:
            self.input_dim = input_dim
        else:
            raise ValueError("Either input_features_dict or input_dim must be provided.")
        self.conditional_dim = conditional_dim

        # --- Encoder --- #
        if encoder_layers is None:
            self.encoder = torch.nn.Linear(self.input_dim + self.conditional_dim, self.embed_dim)
        else:
            enc_layers = []
            enc_layers.append(
                torch.nn.Linear(self.input_dim + self.conditional_dim, encoder_layers[0])
            )
            enc_layers.append(torch.nn.ReLU())

            for i in range(len(encoder_layers) - 1):
                enc_layers.append(torch.nn.Linear(encoder_layers[i], encoder_layers[i + 1]))
                enc_layers.append(torch.nn.ReLU())
            enc_layers.append(torch.nn.Linear(encoder_layers[-1], self.embed_dim))

            self.encoder = torch.nn.Sequential(*enc_layers)

        # --- Vector-quantization layer --- #
        self.vqlayer = VectorQuant(feature_size=self.embed_dim, **vq_kwargs)

        # --- Decoder --- #
        if decoder_layers is None:
            self.decoder = torch.nn.Linear(self.embed_dim + self.conditional_dim, self.input_dim)
        else:
            dec_layers = []
            dec_layers.append(
                torch.nn.Linear(self.embed_dim + self.conditional_dim, decoder_layers[0])
            )
            dec_layers.append(torch.nn.ReLU())

            for i in range(len(decoder_layers) - 1):
                dec_layers.append(torch.nn.Linear(decoder_layers[i], decoder_layers[i + 1]))
                dec_layers.append(torch.nn.ReLU())
            dec_layers.append(torch.nn.Linear(decoder_layers[-1], self.input_dim))

            self.decoder = torch.nn.Sequential(*dec_layers)

        self.loss_history = []
        self.lr_history = []

    def forward(self, x, mask=None, x_conditional=None):
        # mask is there for compatibility with the transformer model
        if x_conditional is not None:
            x_conditional = x_conditional.unsqueeze(1).repeat(1, x.shape[1], 1)
            x = torch.cat([x, x_conditional], dim=-1) * mask.unsqueeze(-1)
        # encode
        z_embed = self.encoder(x)
        # quantize
        z_q2, vq_out = self.vqlayer(z_embed)
        if x_conditional is not None:
            z_q2 = torch.cat([z_q2, x_conditional], dim=-1) * mask.unsqueeze(-1)
        # decode
        x_reco = self.decoder(z_q2)
        return x_reco, vq_out


class VQVAETransformer(torch.nn.Module):
    """This is basically just a re-factor of the VQVAETransformer class, but with more modular
    model components, making it easier to use some components in other models."""

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        conditional_dim: int = 0,
        num_heads: int = 8,
        num_blocks: int = 2,
        vq_kwargs: dict = {},
        causal_decoder: bool = False,
        max_sequence_len: int = 128,
        input_features_dict: Dict[str, Any] = None,
        input_dim: int = None,
        old_transformer_implementation: bool = True,
        in_out_proj_cfg: Dict[str, Any] = None,
        latent_proj_cfg: Dict[str, Any] = None,
        transformer_cfg: dict = None,
        **kwargs,
    ):
        super().__init__()

        self.loss_history = []
        self.lr_history = []

        self.vq_kwargs = vq_kwargs
        self.input_features_dict = input_features_dict
        if input_features_dict is not None:
            self.input_dim = len(self.input_features_dict)
        elif input_dim is not None:
            self.input_dim = input_dim
        else:
            raise ValueError("Either input_features_dict or input_dim must be provided.")

        self.conditional_dim = conditional_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.causal_decoder = causal_decoder
        self.max_sequence_len = max_sequence_len
        self.old_transformer_implementation = old_transformer_implementation

        if transformer_cfg is None:  # old config from when this was not configurable
            transformer_cfg = {
                "attn_cfg": {
                    "num_heads": self.num_heads,
                    "dropout_rate": 0.0,
                    "norm_before": True,
                    "norm_after": False,
                },
                "mlp_cfg": {
                    "expansion_factor": 4,
                    "dropout_rate": 0.0,
                    "norm_before": True,
                    "activation": "ReLU",
                },
                "residual_cfg": {"gate_type": "local", "init_value": 1.0},
            }

        # Model components:
        if in_out_proj_cfg is None:
            self.input_projection = nn.Linear(
                self.input_dim + self.conditional_dim, self.hidden_dim
            )
        else:
            self.input_projection = MLP(
                input_dim=self.input_dim + self.conditional_dim,
                hidden_dims=in_out_proj_cfg.get("hidden_dims"),
                output_dim=self.hidden_dim,
                activation=in_out_proj_cfg.get("activation", "GELU"),
            )

        if not self.old_transformer_implementation:
            self.encoder = Transformer(
                n_blocks=self.num_blocks,
                dim=self.hidden_dim,
                attn_cfg=transformer_cfg["attn_cfg"],
                mlp_cfg=transformer_cfg["mlp_cfg"],
                residual_cfg=transformer_cfg["residual_cfg"],
                norm_after_blocks=False,
            )
        else:
            self.encoder_normformer = NormformerStack(
                hidden_dim=self.hidden_dim,
                num_heads=self.num_heads,
                num_blocks=self.num_blocks,
                mlp_expansion_factor=1,
            )

        if latent_proj_cfg is None:
            self.latent_projection_in = nn.Linear(self.hidden_dim, self.latent_dim)
        else:
            self.latent_projection_in = MLP(
                input_dim=self.hidden_dim,
                hidden_dims=latent_proj_cfg.get("hidden_dims"),
                output_dim=self.latent_dim,
                activation=latent_proj_cfg.get("activation", "GELU"),
            )

        self.vqlayer = VectorQuant(feature_size=self.latent_dim, **vq_kwargs)

        if latent_proj_cfg is None:
            self.latent_projection_out = nn.Linear(
                self.latent_dim + self.conditional_dim, self.hidden_dim
            )
        else:
            self.latent_projection_out = MLP(
                input_dim=self.latent_dim + self.conditional_dim,
                hidden_dims=latent_proj_cfg.get("hidden_dims"),
                output_dim=self.hidden_dim,
                activation=latent_proj_cfg.get("activation", "GELU"),
            )

        if not self.old_transformer_implementation:
            self.decoder = Transformer(
                n_blocks=self.num_blocks,
                dim=self.hidden_dim,
                attn_cfg=transformer_cfg["attn_cfg"],
                mlp_cfg=transformer_cfg["mlp_cfg"],
                residual_cfg=transformer_cfg["residual_cfg"],
                norm_after_blocks=False,
            )

        else:
            raise ValueError("Old implementation of VQ-VAE is not supported anymore.")

        if in_out_proj_cfg is None:
            self.output_projection = nn.Linear(hidden_dim, self.input_dim)
        else:
            self.output_projection = MLP(
                input_dim=self.hidden_dim,
                hidden_dims=in_out_proj_cfg.get("hidden_dims"),
                output_dim=self.input_dim,
                activation=in_out_proj_cfg.get("activation", "GELU"),
            )

    def encode(self, x, mask, x_conditional=None):
        """Encode input to latent embeddings."""
        if x_conditional is not None:
            # x_conditional is of shape (B, C)
            # x is of shape (B, S, F)
            # --> repeat x_conditional to match the shape of x
            x_conditional = x_conditional.unsqueeze(1).repeat(1, x.shape[1], 1)
            x = torch.cat([x, x_conditional], dim=-1) * mask.unsqueeze(-1)

        x = self.input_projection(x)
        if not self.old_transformer_implementation:
            x = self.encoder(x, mask=mask)
        else:
            x = self.encoder_normformer(x, mask)
        z_embed = self.latent_projection_in(x) * mask.unsqueeze(-1)
        return z_embed, x_conditional

    def quantize(self, z_embed):
        """Vector quantize the latent embeddings."""
        z, vq_out = self.vqlayer(z_embed)
        return z, vq_out

    def decode(self, z, mask, x_conditional=None):
        """Decode quantized latents to reconstructed output."""
        if x_conditional is not None:
            z = torch.cat([z, x_conditional], dim=-1) * mask.unsqueeze(-1)

        x_reco = self.latent_projection_out(z) * mask.unsqueeze(-1)
        if not self.old_transformer_implementation:
            if self.causal_decoder:
                attn_mask = (
                    get_causal_mask(x_reco, fill_value=float("-inf"))
                    .to(x_reco.device)
                    .unsqueeze(-1)
                )
            else:
                attn_mask = None
            x_reco = self.decoder(x_reco, mask=mask, attn_mask=attn_mask)
        else:
            x_reco = self.decoder_normformer(x_reco, mask)
        x_reco = self.output_projection(x_reco) * mask.unsqueeze(-1)
        return x_reco

    def forward(self, x, mask, x_conditional=None):
        """Forward pass through encode, quantize, and decode."""
        z_embed, x_conditional_repeated = self.encode(x, mask, x_conditional)
        z, vq_out = self.quantize(z_embed)
        x_reco = self.decode(z, mask, x_conditional_repeated)
        return x_reco, vq_out


class VQVAELightning(L.LightningModule):
    """PyTorch Lightning module for training a VQ-VAE."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler = None,
        model_kwargs={},
        model_type="Transformer",
        **kwargs,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        # --------------- load pretrained model --------------- #
        if model_type == "MLP":
            self.model = VQVAEMLP(**model_kwargs)
        elif model_type in [
            "VQVAETransformer",
            "VQVAENormFormer",  # <- for backwards compatibility with old models
        ]:
            self.model = VQVAETransformer(**model_kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        self.train_loss_history = []
        self.val_loss_list = []

        self.validation_cnt = 0
        self.validation_output = {}

        # for tracking best so far validation accuracy
        self.val_x_original = []
        self.val_x_reco = []
        self.val_mask = []

    def forward(
        self,
        x_particle,
        mask_particle,
        x_conditional=None,
    ):
        x_particle_reco, vq_out = self.model(
            x_particle, mask=mask_particle, x_conditional=x_conditional
        )
        return x_particle_reco, vq_out

    def model_step(self, batch, return_x=False):
        """Perform a single model step on a batch of data."""

        # x_particle, mask_particle, labels = batch
        x_particle = batch["part_features"]
        x_jet = batch.get("jet_features", None)
        mask_particle = batch["part_mask"]
        labels = batch["jet_type_labels"]

        # print(f"conditional_dim = {self.model.conditional_dim}")
        x_particle_reco, vq_out = self.forward(
            x_particle=x_particle,
            mask_particle=mask_particle,
            x_conditional=x_jet if self.model.conditional_dim > 0 else None,
        )

        reco_loss = torch.sum(
            (
                x_particle_reco * mask_particle.unsqueeze(-1)
                - x_particle * mask_particle.unsqueeze(-1)
            )
            ** 2
        ) / torch.sum(mask_particle)

        alpha = self.hparams["model_kwargs"]["alpha"]
        cmt_loss = vq_out["loss"]
        code_idx = vq_out["q"]
        loss = reco_loss + alpha * cmt_loss.mean()

        if return_x:
            return loss, x_particle, x_particle_reco, mask_particle, labels, code_idx

        return loss

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set."""
        loss = self.model_step(batch)

        self.train_loss_history.append(float(loss))
        self.log("train_loss", loss.item(), on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def on_train_start(self) -> None:
        logger.info("`on_train_start` called.")
        self.preprocessing_dict = (
            self.trainer.datamodule.hparams.dataset_kwargs_common.feature_dict
        )

    def on_train_epoch_start(self):
        logger.info(f"`on_train_epoch_start` called. Epoch {self.trainer.current_epoch} starting.")
        self.epoch_train_start_time = time.time()  # start timing the epoch

    def on_train_epoch_end(self):
        logger.info(f"`on_train_epoch_end` called. Epoch {self.trainer.current_epoch} finished.")
        self.epoch_train_end_time = time.time()
        if hasattr(self, "epoch_train_start_time"):
            duration = (self.epoch_train_end_time - self.epoch_train_start_time) / 60
            self.log(
                "epoch_train_duration_minutes",
                duration,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )
            if self.train_loss_history:
                logger.info(
                    f"Epoch {self.trainer.current_epoch} finished in {duration:.1f} minutes. "
                    f"Current step: {self.global_step}. Current loss: {self.train_loss_history[-1]}. "
                    f"Rank: {self.global_rank}"
                )

    def on_train_end(self):
        logger.info("`on_train_end` called.")

    def on_validation_epoch_start(self) -> None:
        logger.info("`on_validation_epoch_start` called.")
        self.val_x_original = []
        self.val_x_reco = []
        self.val_mask = []
        self.val_labels = []
        self.val_code_idx = []

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        loss, x_original, x_reco, mask, labels, code_idx = self.model_step(batch, return_x=True)

        # save the original and reconstructed data
        self.val_x_original.append(x_original.detach().cpu().numpy())
        self.val_x_reco.append(x_reco.detach().cpu().numpy())
        self.val_mask.append(mask.detach().cpu().numpy())
        self.val_labels.append(labels.detach().cpu().numpy())
        self.val_code_idx.append(code_idx.detach().cpu().numpy())

        self.log("val_loss", loss.item(), on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def on_test_epoch_start(self) -> None:
        logger.info("`on_test_epoch_start` called.")
        self.test_x_original = []
        self.test_x_reco = []
        self.test_mask = []
        self.test_labels = []
        self.test_code_idx = []

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        loss, x_original, x_reco, mask, labels, code_idx = self.model_step(batch, return_x=True)

        # save the original and reconstructed data
        self.test_x_original.append(x_original.detach().cpu().numpy())
        self.test_x_reco.append(x_reco.detach().cpu().numpy())
        self.test_mask.append(mask.detach().cpu().numpy())
        self.test_labels.append(labels.detach().cpu().numpy())
        self.test_code_idx.append(code_idx.detach().cpu().numpy())

        self.log("test_loss", loss.item(), on_step=True, on_epoch=True, prog_bar=True)

    def tokenize_ak_array(
        self,
        ak_arr,
        pp_dict,
        ak_arr_jet=None,
        pp_dict_jet=None,
        batch_size=256,
        pad_length=128,
        hide_pbar=False,
    ):
        """Tokenize an awkward array of jets.

        Parameters
        ----------
        ak_arr : ak.Array
            Awkward array of jets, shape (N_jets, <var>, N_features).
        pp_dict : dict
            Dictionary with preprocessing information.
        ak_arr_jet : ak.Array
            Awkward array of jet-level features, shape (N_jets, N_features_jet).
        pp_dict_jet : dict
            Dictionary with preprocessing information for jet-level features.
        batch_size : int, optional
            Batch size for the evaluation loop. The default is 256.
        pad_length : int, optional
            Length to which the tokens are padded. The default is 128.
        hide_pbar : bool, optional
            Whether to hide the progress bar. The default is False.

        Returns
        -------
        ak.Array
            Awkward array with the token-ids and the jet features if used. Thus,
            the field names are "part_token_id" and "jet_features".
        """

        # preprocess the ak_arrary
        ak_arr = ak_select_and_preprocess(ak_arr, pp_dict=pp_dict)
        ak_arr_padded, mask = ak_pad(ak_arr, maxlen=pad_length, return_mask=True)
        # convert to numpy
        arr = ak_to_np_stack(ak_arr_padded, names=pp_dict.keys())
        # convert to torch tensor
        x = torch.from_numpy(arr).float()
        mask = torch.from_numpy(mask.to_numpy()).float()

        if self.model.conditional_dim == 0:
            dataset = TensorDataset(x, mask)
            dataloader = DataLoader(dataset, batch_size=batch_size)
        else:
            print("Using jet-level features as conditional data.")
            ak_arr_jet_pp = ak_select_and_preprocess(ak_arr_jet, pp_dict=pp_dict_jet)
            np_arr_jet_pp = ak_to_np_stack(ak_arr_jet_pp, names=pp_dict_jet.keys())
            x_jet = torch.from_numpy(np_arr_jet_pp).float()
            dataset = TensorDataset(x, mask, x_jet)
            dataloader = DataLoader(dataset, batch_size=batch_size)

        codes = []
        z_qs = []

        with torch.no_grad():
            pbar = tqdm(dataloader) if not hide_pbar else dataloader
            for i, batch in enumerate(pbar):
                # move to device
                if self.model.conditional_dim == 0:
                    x_batch, mask_batch = batch
                    x_jet_batch = None
                else:
                    x_batch, mask_batch, x_jet_batch = batch
                    x_jet_batch = x_jet_batch.to(self.device)
                x_batch = x_batch.to(self.device)
                mask_batch = mask_batch.to(self.device)
                x_particle_reco, vq_out = self.forward(
                    x_batch, mask_batch, x_conditional=x_jet_batch
                )
                code = vq_out["q"]
                z_q = vq_out["z_q"]
                codes.append(code)
                z_qs.append(z_q)

        codes = torch.cat(codes, dim=0).detach().cpu().numpy()
        z_qs = torch.cat(z_qs, dim=0).squeeze(-2).detach().cpu().numpy()
        mask = mask.detach().cpu().numpy()

        if isinstance(self.model.vqlayer, VectorQuant):
            feature_names = ["part_token_id"]
        else:
            raise ValueError("Unknown VectorQuant type.")

        ak_arr_tokens = np_to_ak(codes, names=feature_names, mask=mask)
        ak_arr_zqs = np_to_ak(z_qs, names=[f"z_q_{i}" for i in range(z_qs.shape[-1])], mask=mask)

        if self.model.conditional_dim == 0:
            dict_with_jet_features = {}
        else:
            dict_with_jet_features = ak_arr_jet

        ak_arr = ak.Array(
            {
                "part_token_id": ak_arr_tokens,
                "z_q": ak_arr_zqs,
            }
            | dict_with_jet_features
        )
        return ak_arr

    def reconstruct_ak_tokens(
        self,
        tokens_ak,
        pp_dict,
        jets_ak=None,
        pp_dict_jet=None,
        batch_size=256,
        pad_length=128,
        hide_pbar=False,
    ):
        """Reconstruct tokenized awkward array.

        Parameters
        ----------
        tokens_ak : ak.Array
            Awkward array of tokens, shape (N_jets, <var>).
        pp_dict : dict
            Dictionary with preprocessing information.
        jets_ak : ak.Array
            Awkward array of jet-level features, shape (N_jets, N_features_jet).
        pp_dict_jet : dict
            Dictionary with preprocessing information for jet-level features.
        batch_size : int, optional
            Batch size for the evaluation loop. The default is 256.
        pad_length : int, optional
            Length to which the tokens are padded. The default is 128.
        hide_pbar : bool, optional
            Whether to hide the progress bar. The default is False.

        Returns
        -------
        ak.Array
            Awkward array of reconstructed jets, shape (N_jets, <var>, N_features).
        """

        self.model.eval()

        tokens, mask = ak_pad(tokens_ak, maxlen=pad_length, return_mask=True)
        if len(tokens.fields) == 0:
            tokens = torch.from_numpy(tokens.to_numpy()).long()
        else:
            tokens = torch.from_numpy(ak_to_np_stack(tokens, names=tokens_ak.fields)).long()
        mask = torch.from_numpy(mask.to_numpy()).float()

        if self.model.conditional_dim > 0:
            conditional_data = ak_select_and_preprocess(jets_ak, pp_dict=pp_dict_jet)
            conditional_data = ak_to_np_stack(conditional_data, names=pp_dict_jet.keys())
            x_conditional = torch.from_numpy(conditional_data).float()
            # concatenate the conditional data to the tokens
            x_conditional = x_conditional.unsqueeze(1).repeat(1, tokens.shape[1], 1)

        x_reco = []
        if self.model.conditional_dim == 0:
            dataset = TensorDataset(tokens, mask)
            dataloader = DataLoader(dataset, batch_size=batch_size)
        else:
            dataset = TensorDataset(tokens, mask, x_conditional)
            dataloader = DataLoader(dataset, batch_size=batch_size)

        codebook = self.model.vqlayer.codebook.weight

        # if the codebook has an affine transform, apply it
        # before using it to reconstruct the data
        # see https://github.com/minyoungg/vqtorch/blob/main/vqtorch/nn/vq.py#L102-L104
        if hasattr(self.model.vqlayer, "affine_transform"):
            codebook = self.model.vqlayer.affine_transform(codebook)

        last_batch = None
        with torch.no_grad():
            pbar = tqdm(dataloader) if not hide_pbar else dataloader
            for i, batch in enumerate(pbar):
                # move to device
                if self.model.conditional_dim == 0:
                    tokens_batch, mask_batch = batch
                else:
                    tokens_batch, mask_batch, x_conditional_batch = batch
                    x_conditional_batch = x_conditional_batch.to(self.device)

                tokens_batch = tokens_batch.to(self.device)
                mask_batch = mask_batch.to(self.device)
                try:
                    z_q = F.embedding(tokens_batch, codebook)
                    z_q = z_q.squeeze(-2)
                except Exception as e:  # noqa: E722
                    logger.info(f"Error in embedding: {e}")
                    logger.info("batch shape", tokens_batch.shape)
                    logger.info("batch max", tokens_batch.max())
                    logger.info("batch min", tokens_batch.min())

                # print(f"z_q shape: {z_q.shape}")

                # if conditioning is used, concatenate the conditional data to the tokens
                if self.model.conditional_dim > 0:
                    z_q = torch.cat([z_q, x_conditional_batch], dim=-1) * mask_batch.unsqueeze(-1)

                if last_batch is not None:
                    break

                if hasattr(self.model, "latent_projection_out"):
                    x_reco_batch = self.model.decode(z_q, mask=mask_batch)
                elif hasattr(self.model, "decoder"):
                    x_reco_batch = self.model.decoder(z_q)
                else:
                    raise ValueError("Unknown model structure. Cannot reconstruct.")
                x_reco.append(x_reco_batch)

        x_reco = torch.cat(x_reco, dim=0).detach().cpu().numpy()
        x_reco_ak = np_to_ak(x_reco, names=pp_dict.keys(), mask=mask.detach().cpu().numpy())
        x_reco_ak = ak_select_and_preprocess(x_reco_ak, pp_dict, inverse=True)

        return x_reco_ak

    def concat_validation_loop_predictions(self) -> None:
        self.val_x_original_concat = np.concatenate(self.val_x_original)
        self.val_x_reco_concat = np.concatenate(self.val_x_reco)
        self.val_mask_concat = np.concatenate(self.val_mask)
        self.val_labels_concat = np.concatenate(self.val_labels)
        self.val_code_idx_concat = np.concatenate(self.val_code_idx)

    def on_validation_end(self) -> None:
        """Lightning hook that is called when a validation loop ends."""
        logger.info("`on_validation_end` called.")
        self.concat_validation_loop_predictions()

    def on_test_end(self):
        logger.info("`on_test_end` called.")
        self.concat_test_loop_predictions()

    def concat_test_loop_predictions(self) -> None:
        self.test_x_original_concat = np.concatenate(self.test_x_original)
        self.test_x_reco_concat = np.concatenate(self.test_x_reco)
        self.test_mask_concat = np.concatenate(self.test_mask)
        self.test_labels_concat = np.concatenate(self.test_labels)
        self.test_code_idx_concat = np.concatenate(self.test_code_idx)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configures optimizers and learning-rate schedulers to be used for training."""
        optimizer = self.hparams.optimizer(params=self.parameters())
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
