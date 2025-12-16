import omegaconf
import torch
import torch.nn as nn
import vector

from gabbro.models.transformer import MLP, Transformer
from gabbro.models.weaver_particle_transformer import PairEmbed as PairEmbedParT
from gabbro.utils.arrays import get_causal_mask
from gabbro.utils.pylogger import get_pylogger
from gabbro.utils.utils import translate_transformer_cfg_to_old_syntax

vector.register_awkward()

logger = get_pylogger(__name__)


class IdentityXKwargs(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x


class ProjectAdd(nn.Module):
    """Embedding layer for token IDs."""

    def __init__(
        self,
        vocab_size: int = None,
        n_part_features: int = None,
        n_jet_features: int = None,
        embedding_dim: int = None,
        intermediate_part_dim: int = None,
    ):
        super().__init__()

        self.vocab_size = vocab_size

        if vocab_size is None:
            self.embed_part = (
                nn.Linear(n_part_features, embedding_dim) if n_part_features is not None else None
            )
        else:
            if intermediate_part_dim is not None:
                self.embed_part = nn.Sequential(
                    nn.Embedding(vocab_size, intermediate_part_dim),
                    nn.Linear(intermediate_part_dim, embedding_dim),
                )
            else:
                self.embed_part = nn.Embedding(vocab_size, embedding_dim)

        self.embed_jet = (
            nn.Linear(n_jet_features, embedding_dim, bias=False)
            if n_jet_features is not None
            else None
        )

    def forward(self, x, x_jet=None):
        # x is expected to be of shape (batch_size, seq_len, n_part_features)
        # x_jet is expected to be of shape (batch_size, n_jet_features)

        if self.vocab_size is not None:
            x_part_embed = self.embed_part(x.squeeze(-1).long())
        else:
            x_part_embed = self.embed_part(x)

        if x_jet is not None:
            x_jet = x_jet.unsqueeze(1).expand(-1, x.shape[1], -1)
            x_jet_embed = self.embed_jet(x_jet)

            return x_part_embed + x_jet_embed

        return x_part_embed


class NextTokenPredictionHead(nn.Module):
    """Head for predicting the next token in a sequence."""

    def __init__(self, embedding_dim, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.fc1 = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x):
        return self.fc1(x)


class TokenPredictionHead(nn.Module):
    """Head for predicting the next token in a sequence."""

    def __init__(
        self,
        input_dim: int,
        vocab_size: int,
        n_pred: int = 1,
        transformer_cfg: dict = None,
        unembedding_mlp_cfg: dict = None,
        dropout: float = 0.1,
        apply_causal_mask: bool = True,
    ):
        super().__init__()

        # validate input
        if n_pred < 1:
            raise ValueError(
                f"n_pred must be at least 1, but got {n_pred}. "
                "This is the number of independent heads for next token prediction."
            )
        if transformer_cfg is None:
            logger.warning(
                "transformer_cfg is None, which means that no transformer blocks "
                "will be used. The model will only use the linear layer for prediction."
            )

        self.vocab_size = vocab_size
        self.n_pred = n_pred
        self.transformer_cfg = transformer_cfg
        self.apply_causal_mask = apply_causal_mask
        self.unembedding_mlp_cfg = unembedding_mlp_cfg
        self.dropout = dropout

        base_dim = transformer_cfg["dim"] if transformer_cfg else input_dim

        # Add downprojection if the input dim is not equal to the transformer dim
        if transformer_cfg is None:
            self.projection_in = None
        elif transformer_cfg.get("n_blocks", 0) == 0:
            self.projection_in = None
        elif transformer_cfg.get("dim") != input_dim:
            self.projection_in = nn.Linear(input_dim, base_dim)
        else:
            self.projection_in = None

        # independent heads for next, next+1, next+2, ... prediction
        if transformer_cfg is None:
            self.transformer_blocks = None
        elif transformer_cfg.get("n_blocks", 0) == 0:
            self.transformer_blocks = None
        else:
            # New architecture: use the transformer_cfg for each head
            self.transformer_blocks = nn.ModuleList(
                [Transformer(**transformer_cfg) for _ in range(self.n_pred)]
            )

        if unembedding_mlp_cfg is not None:
            self.unembedding_mlp = MLP(
                input_dim=base_dim,
                output_dim=base_dim,
                **unembedding_mlp_cfg,
            )
        else:
            self.unembedding_mlp = nn.Identity()

        # Simple linear layer for unembedding
        self.unembedding_matrix = nn.Linear(base_dim, vocab_size)

    def __repr__(self):
        if self.transformer_blocks is not None:
            transformer_blocks_repr = self.transformer_blocks.__repr__().replace("\n", "\n  ")
        else:
            transformer_blocks_repr = "None"

        if self.unembedding_mlp is not None:
            unembedding_repr = self.unembedding_mlp.__repr__().replace("\n", "\n  ")
        else:
            unembedding_repr = "None"

        return (
            f"TokenPredictionHead(\n"
            f"  vocab_size={self.vocab_size},\n"
            f"  n_pred={self.n_pred},\n"
            f"  apply_causal_mask={self.apply_causal_mask},\n"
            f"  (projection_in): {self.projection_in},\n"
            f"  (transformer_blocks_per_token_pred): {transformer_blocks_repr},\n"
            f"  (unembedding_mlp): {unembedding_repr},\n"
            f"  (unembedding_matrix): {self.unembedding_matrix},\n"
            f")"
        )

    def forward(self, x, mask):
        # Apply downprojection if using new architecture
        if self.projection_in is not None:
            x = self.projection_in(x)

        # if there is
        if self.transformer_blocks is None:
            return (
                self.unembedding_matrix(self.unembedding_mlp(x)) * mask.unsqueeze(-1)
            ).unsqueeze(-1)

        if self.apply_causal_mask:
            attn_mask = get_causal_mask(x, fill_value=float("-inf")).to(x.device).unsqueeze(-1)
        else:
            attn_mask = None
        return torch.cat(
            [
                self.unembedding_matrix(
                    self.unembedding_mlp(head(x, mask=mask, attn_mask=attn_mask))
                ).unsqueeze(-1)
                for i, head in enumerate(self.transformer_blocks)
            ],
            dim=-1,
        )


class MPMHead(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        transformer_cfg: dict = None,
        hidden_dims: list = None,
        dropout: float = 0.1,
        apply_causal_mask: bool = False,
    ):
        super().__init__()

        self.transformer_cfg = transformer_cfg
        self.apply_causal_mask = apply_causal_mask
        self.hidden_dims = hidden_dims if hidden_dims is not None else [128, 128, 64]

        # downprojection to the desired dimension "hidden_dims[0]"
        self.downproject = nn.Linear(input_dim, self.hidden_dims[0])

        if self.transformer_cfg is not None:
            self.transformer_cfg["embedding_dim"] = self.hidden_dims[0]
            self.transformer_stack = Transformer(
                dim=self.hidden_dims[0],
                **self.transformer_cfg,
            )

        # create the MLP
        dims = hidden_dims + [output_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            # no relu and dropout after the last layer
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x, mask=None):
        x = self.downproject(x)
        if self.transformer_cfg is not None:
            if self.apply_causal_mask:
                attn_mask = get_causal_mask(x, fill_value=float("-inf")).to(x.device)
                num_heads = self.transformer_cfg["attn_cfg"]["num_heads"]
                attn_mask = attn_mask.unsqueeze(0).expand(x.shape[0] * num_heads, -1, -1)
                # also add the last dimension to the attention mask
                attn_mask = attn_mask.unsqueeze(-1)
            else:
                attn_mask = None
            x = self.transformer_stack(x, mask=mask, attn_mask=attn_mask)
        x = self.mlp(x)
        return x


class BackboneTransformer(nn.Module):
    """Transformer backbone model for continuous input features."""

    def __init__(
        self,
        embedding_dim: int,
        apply_causal_mask: bool = True,
        max_sequence_len: int = 256,
        vocab_size: int = 8194,
        n_registers: int = 0,
        embed_cfg: dict = None,
        transformer_cfg: dict = None,
        interaction_cfg: dict = None,
        particle_features_dict: dict = None,
        jet_features_dict: dict = None,
        feature_drop_cfg: dict = None,
        **kwargs,
    ):
        super().__init__()
        self.apply_causal_mask = apply_causal_mask
        if not self.apply_causal_mask:
            logger.warning(
                "NOT applying causal mask in the attention blocks. If you are using "
                "this model for an autoregressive generative task, this is probably "
                "not what you want."
            )

        self.max_sequence_len = max_sequence_len
        self.vocab_size = vocab_size
        self.n_registers = n_registers
        self.embedding_dim = embedding_dim
        self.transformer_cfg = transformer_cfg
        self.interaction_cfg = interaction_cfg
        self.embed_cfg = embed_cfg

        supported_embed_types = [
            "continuous_concat_project",
            "continuous_project_add",
            "token_id_project_add",
        ]
        if self.embed_cfg.get("type") not in supported_embed_types:
            raise ValueError(
                f"embed_cfg.type must be one of {supported_embed_types}, but "
                f"got {self.embed_cfg.get('type')}."
            )

        logger.info(f"Using input type: {self.embed_cfg.get('type')}")

        self.jet_features_dict = jet_features_dict
        self.jet_features_input_dim = (
            len(jet_features_dict) if jet_features_dict is not None else 0
        )
        logger.info(f"Jet features dict: {self.jet_features_dict}")
        logger.info(f"Jet features input dim: {self.jet_features_input_dim}")

        self.particle_features_dict = particle_features_dict
        if isinstance(self.particle_features_dict, omegaconf.DictConfig):
            self.particle_features_dict = omegaconf.OmegaConf.to_container(
                self.particle_features_dict, resolve=True
            )
        if (
            self.embed_cfg.get("type") == "continuous_concat_project"
            or self.embed_cfg.get("type") == "continuous_project_add"
        ):
            self.part_features_input_dim = len(self.particle_features_dict)
        elif self.embed_cfg.get("type") == "token_id_project_add":
            self.part_features_input_dim = 1

        logger.info(f"feature drop cfg: {feature_drop_cfg}")

        if feature_drop_cfg is not None:
            if isinstance(feature_drop_cfg, omegaconf.DictConfig):
                feature_drop_cfg = omegaconf.OmegaConf.to_container(feature_drop_cfg, resolve=True)
            self.feature_drop_rate = feature_drop_cfg.get("drop_rate", 0.0)
            logger.info(f"Feature drop rate: {self.feature_drop_rate}")
            # figure out which features are used in feature drop, and write out their
            # indices
            self.particle_feature_indices_used_for_feature_drop = []
            for i, feature_name in enumerate(self.particle_features_dict):
                if feature_name in feature_drop_cfg.get("features", []):
                    self.particle_feature_indices_used_for_feature_drop.append(i)
            logger.info(
                "Indices of particle features used for feature drop: "
                f"{self.particle_feature_indices_used_for_feature_drop}"
            )
            # if no features are specified, set the feature drop rate to 0.0
            if len(self.particle_feature_indices_used_for_feature_drop) == 0:
                self.feature_drop_rate = 0.0
                logger.warning(
                    "No particle features specified for feature drop, setting feature drop rate to 0.0."
                )

        else:
            self.feature_drop_rate = 0.0
            logger.info("No feature drop rate specified, using 0.0")

        # if the feature dict includes features that are intended for the interaction
        # module, use them
        if interaction_cfg is not None:
            self.n_part_features_used_for_interaction = interaction_cfg.get(
                "use_n_last_input_features", 0
            )
            logger.info(
                f"Using last {self.n_part_features_used_for_interaction} particle "
                "features for interaction"
            )
            if not interaction_cfg.get("also_use_interaction_features_as_node_input"):
                self.part_features_input_dim -= self.n_part_features_used_for_interaction
                logger.info(
                    "The interaction features are only used for the interaction "
                    "module, not as input to the transformer backbone."
                )
        logger.info(f"Particle features dict: {self.particle_features_dict}")
        logger.info(f"Particle features input dim: {self.part_features_input_dim}")

        if self.n_registers > 0:
            self.registers = nn.Parameter(
                torch.randn(1, self.n_registers, self.embedding_dim),
            )

        if "project_add" in self.embed_cfg.get("type"):
            # use a simple embedding layer for token IDs
            self.input_projection = ProjectAdd(
                vocab_size=self.vocab_size
                if self.embed_cfg.get("type") == "token_id_project_add"
                else None,
                embedding_dim=self.embedding_dim,
                n_part_features=self.part_features_input_dim,
                n_jet_features=self.jet_features_input_dim,
                intermediate_part_dim=self.embed_cfg.get("intermediate_dim", None),
            )
        elif self.embed_cfg.get("type") == "continuous_concat_project":
            project_input_dim = int(self.part_features_input_dim + self.jet_features_input_dim)
            self.input_projection = nn.Linear(project_input_dim, self.embedding_dim)
        else:
            raise NotImplementedError(
                f"Input type '{self.embed_cfg.get('type')}' is not supported."
            )

            # self.input_projection = MLP(
            #     input_dim=self.input_dim,
            #     output_dim=self.embedding_dim,
            #     hidden_dims=embed_cfg.get("hidden_dims", [64, 64]),
            #     dropout_rate=embed_cfg.get("dropout_rate", 0.0),
            #     activation=embed_cfg.get("activation", "GELU"),
            # )

        # initialize the transformer (support for old implementations as well for compatibility)
        if transformer_cfg is not None:
            # default: new implementation
            if transformer_cfg.get("transformer_implementation") is None:
                from gabbro.models.transformer import Transformer

                self.transformer = Transformer(**transformer_cfg)

            # old implementation based on Normformer
            elif transformer_cfg.get("transformer_implementation") == "normformer":
                from gabbro.models.transformer import NormformerStack

                self.transformer = NormformerStack(
                    **translate_transformer_cfg_to_old_syntax(transformer_cfg, "NormformerStack"),
                    # apply_mask_after_mlp=True,
                )
            else:
                raise ValueError(
                    f"Unknown transformer implementation: {transformer_cfg['transformer_implementation']}"
                )
        else:
            self.transformer = IdentityXKwargs()

        if interaction_cfg is not None:
            # self.interaction_mlp = PairEmbed(
            #     input_dim=continuous_input_dim, output_dim=self.n_heads
            # )
            interaction_type = interaction_cfg.get("type", "NotSet")
            if interaction_type == "PairEmbedParT":
                self.pair_embed = PairEmbedParT(
                    pairwise_lv_dim=4,
                    pairwise_input_dim=0,
                    dims=[
                        64,
                        64,
                        64,
                        transformer_cfg["attn_cfg"]["num_heads"]
                        * interaction_cfg.get("num_interaction_matrices", 1),
                    ],
                    remove_self_pair=False,
                    use_pre_activation_pair=True,
                )
            # elif self.interactions_module_class == "PairEmbed":
            #     self.pair_embed = PairEmbed(
            #         input_dim=self.interaction_input_dim,
            #         mlp_pointwise_dims=pair_embed_pointwise_dims,
            #         mlp_pair_dims=pair_embed_pair_dims,
            #         use_residual_point_cat=pair_embed_use_residual_point_cat,
            #         conv_dims=[64, 64, 64, self.n_heads],
            #         use_diff=pair_embed_use_diff,
            #     )
            # elif self.interactions_module_class == "PairEmbedDotProduct":
            #     self.pair_embed = PairEmbedDotProduct(
            #         input_dim=self.interaction_input_dim,
            #         mlp_pointwise_dims=pair_embed_pointwise_dims,
            #         n_interaction_matrices=self.n_heads,
            #     )
            else:
                raise ValueError(
                    f"Unknown interactions_module_class: {self.interactions_module_class}"
                )
        else:
            self.pair_embed = None

    def __repr__(self):
        transformer_repr = self.transformer.__repr__().replace("\n", "\n  ")
        pair_embed_repr = (
            self.pair_embed.__repr__().replace("\n", "\n  ")
            if self.pair_embed is not None
            else "None"
        )
        input_projection_repr = self.input_projection.__repr__().replace("\n", "\n  ")
        return (
            f"BackboneTransformer(\n"
            f"  dim={self.embedding_dim},\n"
            f"  part_features_input_dim={self.part_features_input_dim},\n"
            f"  jet_features_input_dim={self.jet_features_input_dim},\n"
            f"  max_sequence_len={self.max_sequence_len},\n"
            f"  vocab_size={self.vocab_size},\n"
            f"  n_registers={self.n_registers},\n"
            f"  apply_causal_mask={self.apply_causal_mask},\n"
            f"  (pair_embed): {pair_embed_repr},\n"
            f"  (input_projection): {input_projection_repr},\n"
            f"  (transformer): {transformer_repr},\n"
            f")"
        )

    def forward(self, x, mask=None, x_jet=None):
        # x.shape = (batch_size, seq_len, input_dim)
        mask_clone = mask.clone() if mask is not None else None

        # logger.info(f"x[0, :, 0] = {x[0, :, 0]}")

        # if feature drop is used, randomly drop some features of the specified indices
        if self.feature_drop_rate > 0.0 and self.training:
            x = x.clone()
            if mask is not None:
                mask_clone = mask_clone.clone()
            else:
                mask_clone = torch.ones(x.shape[0], x.shape[1]).to(x.device)

            # create a random mask for the features to drop
            feature_drop_mask = (
                torch.rand(
                    x.shape[0],
                    x.shape[1],
                    len(self.particle_feature_indices_used_for_feature_drop),
                ).to(x.device)
                < self.feature_drop_rate
            )

            # get indices as tensor for advanced indexing
            idx = torch.tensor(
                self.particle_feature_indices_used_for_feature_drop, device=x.device
            )

            # apply the feature drop mask to the selected particle features in one operation
            x[:, :, idx] = x[:, :, idx] * (~feature_drop_mask).float()

        # logger.info(f"x[0, :, 0] = {x[0, :, 0]}")

        # attention mask is
        # - responsible for masking future elements in (autoregressive) generative models
        # - adding the interaction features to the attention matrix if they are used
        attn_mask = None

        # if registers are used, add zero-padded entries at beginning of x
        # and mask
        if self.n_registers > 0:
            # concatenate the registers to the input (registers first)
            if mask is not None:
                mask = torch.cat(
                    [
                        torch.ones(x.shape[0], self.n_registers).to(mask.device),
                        mask,
                    ],
                    dim=-1,
                )
            x = torch.cat(
                [
                    torch.zeros(x.shape[0], self.n_registers, x.shape[2]).to(x.device),
                    x,
                ],
                dim=1,
            )

        if self.apply_causal_mask:
            attn_mask = get_causal_mask(x, fill_value=float("-inf")).to(x.device)

        if self.pair_embed is None:
            if attn_mask is not None:
                attn_mask = attn_mask.unsqueeze(-1)
                # last dim represents "block index" - without interaction features, we use
                # the same mask for all blocks (which is represented by only length 1, i.e.
                # a simple unsqueeze)
        else:
            x_interaction = self.pair_embed(
                x[:, :, -self.interaction_cfg["use_n_last_input_features"] :].transpose(1, 2),
            )
            if not self.interaction_cfg.get("also_use_interaction_features_as_node_input"):
                x = x[:, :, : -self.interaction_cfg["use_n_last_input_features"]]

            if self.interaction_cfg["num_interaction_matrices"] > 1:
                x_interaction = x_interaction.split(
                    self.interaction_cfg["num_interaction_matrices"], dim=1
                )
                x_interaction = torch.stack(x_interaction, dim=-1)
            else:
                # just unsqueeze the last dimension
                x_interaction = x_interaction.unsqueeze(-1)

            # x_interaction has shape (N, num_heads, L, L, num_interaction_matrices)
            # --> reshape such that attn mask is of shape (N * num_heads, L, L, num_interaction_matrices)
            # this way the first num_heads entries are the attention masks for the first
            # entry in the batch, then the next num_heads entries are the attention masks for the
            # second entry in the batch, ...
            x_interaction = x_interaction.view(
                x_interaction.shape[0] * x_interaction.shape[1],
                x_interaction.shape[2],
                x_interaction.shape[3],
                x_interaction.shape[4],
            )

            # if no attention mask is provided, create a zero mask
            if attn_mask is None:
                attn_mask = x_interaction
            else:
                # this means the attention mask is the causal mask of shape
                # (seq_len, seq_len)
                num_heads = self.transformer_cfg["attn_cfg"]["num_heads"]
                attn_mask = attn_mask.unsqueeze(0).expand(x.shape[0] * num_heads, -1, -1)
                # also add the last dimension to the attention mask
                attn_mask = attn_mask.unsqueeze(-1)
                # if we have multiple interaction matrices, we need to expand the attention mask
                # for each interaction matrix
                # this is done by repeating the attention mask for each interaction matrix
                # and adding the interaction features to the attention mask
                attn_mask = attn_mask.repeat(
                    1,
                    1,
                    1,
                    self.interaction_cfg.get("num_interaction_matrices", 1),
                )

                attn_mask = attn_mask + x_interaction

        if self.embed_cfg.get("type") == "continuous_concat_project":
            # concatenate the continuous input features with the jet features
            # the jet features should be repeated for each token in the sequence
            if x_jet is not None:
                x_jet = x_jet.unsqueeze(1).expand(-1, x.shape[1], -1)
                x = torch.cat([x, x_jet], dim=-1)

            x = self.input_projection(x)

        elif (
            self.embed_cfg.get("type") == "continuous_project_add"
            or self.embed_cfg.get("type") == "token_id_project_add"
        ):
            x = self.input_projection(x, x_jet=x_jet)

        if self.n_registers > 0:
            # concatenate the registers to the input (registers first)
            registers_repeat = self.registers.repeat(x.shape[0], 1, 1)
            x[:, : self.n_registers, :] = registers_repeat

        x = self.transformer(x, mask=mask, attn_mask=attn_mask)

        # drop the registers from the output (if used)
        if self.n_registers > 0:
            x = x[:, self.n_registers :, :]

        # apply mask to the output
        if mask is not None:
            x = x * mask_clone.unsqueeze(-1)

        return x
