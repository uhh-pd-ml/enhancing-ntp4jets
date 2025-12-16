"""Transformer/Normformer model implementation."""

import torch
import torch.nn as nn


class PointDrop(nn.Module):
    """Similar to Dropout, but the drop happens along the point dimension of a tensor
    of shape (batch_size, n_points, n_feats)."""

    def __init__(self, drop_rate: float = 0.1, fill_type: str = "zero"):
        super().__init__()

        # validate input
        supported_fill_types = ["zero", "normal"]
        if fill_type not in supported_fill_types:
            raise ValueError(
                f"Invalid fill_type: {fill_type}. Must be one of {supported_fill_types}."
            )
        if drop_rate < 0.0 or drop_rate > 1.0:
            raise ValueError(f"Invalid drop_rate: {drop_rate}. Must be between 0 and 1.")

        self.drop_rate = drop_rate
        self.fill_type = fill_type  # can be `zero` and `normal`

    def __repr__(self):
        return f"PointDrop(drop_rate={self.drop_rate}, fill_type={self.fill_type})"

    def forward(self, x):
        if self.training and self.drop_rate > 0.0:
            # create a mask for the points to keep
            mask = torch.rand(x.shape[0], x.shape[1], device=x.device) > self.drop_rate
            if self.fill_type == "zero":
                return x * mask.unsqueeze(-1).float()
            elif self.fill_type == "normal":
                noise = torch.randn_like(x)
                return x * mask.unsqueeze(-1).float() + noise * (1 - mask.unsqueeze(-1).float())
        return x


class NormformerBlock(nn.Module):
    def __init__(
        self,
        input_dim,
        mlp_expansion_factor: int = 4,
        num_heads: int = 8,
        dropout_rate=0.1,
        init_identity=True,
        apply_mask_after_mlp=False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.mlp_expansion_factor = mlp_expansion_factor
        self.apply_mask_after_mlp = apply_mask_after_mlp

        # define the MultiheadAttention layer with layer normalization
        self.norm1 = nn.LayerNorm(input_dim)
        self.attn = nn.MultiheadAttention(
            self.input_dim,
            self.num_heads,
            batch_first=True,
            dropout=self.dropout_rate,
        )
        self.norm2 = nn.LayerNorm(input_dim)

        # define the MLP with layer normalization
        self.mlp = nn.Sequential(
            nn.LayerNorm(self.input_dim),  # Add layer normalization
            nn.Linear(self.input_dim, self.input_dim * self.mlp_expansion_factor),
            nn.SiLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.input_dim * self.mlp_expansion_factor, self.input_dim),
        )

        # initialize weights of mlp[-1] and layer norm after attn block to 0
        # such that the residual connection is the identity when the block is
        # initialized
        if init_identity:
            nn.init.zeros_(self.mlp[-1].weight)
            nn.init.zeros_(self.mlp[-1].bias)
            nn.init.zeros_(self.norm1.weight)

    def forward(self, x, mask=None, return_attn_weights=False, attn_mask=None, **kwargs):
        # x: (B, S, F)
        # mask: (B, S)
        if mask is None:
            mask = torch.ones(x.size(0), x.size(1), device=x.device)
        x = x * mask.unsqueeze(-1)

        # calculate self-attention
        x_norm = self.norm1(x)
        attn_output, attn_weights = self.attn(
            x_norm,
            x_norm,
            x_norm,
            key_padding_mask=mask != 1,
            attn_mask=attn_mask,
        )
        # Add residual connection and permute back to (B, S, F)
        attn_res = self.norm2(attn_output) + x

        output = self.mlp(attn_res) + attn_res

        if self.apply_mask_after_mlp:
            output = output * mask.unsqueeze(-1)

        if return_attn_weights:
            return output, attn_weights

        # output shape: (B, S, F)
        return output


class NormformerStack(torch.nn.Module):
    def __init__(
        self,
        hidden_dim,
        num_heads=1,
        num_blocks=2,
        dropout_rate=0.1,
        mlp_expansion_factor=4,
        apply_causal_mask=False,
        init_identity=True,
        **kwargs,
    ):
        super().__init__()

        self.num_blocks = num_blocks
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.mlp_expansion_factor = mlp_expansion_factor
        self.apply_causal_mask = apply_causal_mask

        self.blocks = nn.ModuleList(
            [
                NormformerBlock(
                    input_dim=self.hidden_dim,
                    num_heads=self.num_heads,
                    dropout_rate=self.dropout_rate,
                    mlp_expansion_factor=self.mlp_expansion_factor,
                    init_identity=init_identity,
                    **kwargs,
                )
                for _ in range(num_blocks)
            ]
        )

    def forward(self, x, mask, **kwargs):
        attn_mask = kwargs.get("attn_mask", None)

        if mask is None:
            mask = torch.ones(x.size(0), x.size(1), device=x.device)
        for i, block in enumerate(self.blocks):
            x = block(x, mask=mask, attn_mask=attn_mask)
        return x * mask.unsqueeze(-1)


class MLP(nn.Module):
    """Simple MLP for embedding."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: list = None,
        dropout_rate: float = 0.0,
        activation: str = "GELU",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        act_fn = eval(f"torch.nn.{activation}")()

        dims = [self.input_dim] + list(self.hidden_dims) + [self.output_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(act_fn)
                layers.append(nn.Dropout(self.dropout_rate))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class Residual(nn.Module):
    """Residual block."""

    def __init__(
        self,
        module,
        gate_type: str = "global",
        init_value: float = 1.0,
        point_drop_cfg: dict = None,
    ):
        """
        Parameters
        ----------
        module: nn.Module
            Module to be wrapped in the residual block.
        gate_type: str
            Type of gate to use. Can be 'global' or 'local'.
            Local means the gate is a vector of the same dimension as the input/output
            of the module. Global means the gate is a scalar (i.e. same for all inputs).
        init_value: float
            Initial value for the gate. Only used if init_identity is False.
        point_drop_cfg: dict
            Configuration for point dropout.
        """
        super().__init__()
        self.module = module
        self.gate_type = gate_type
        self.init_value = init_value
        self.point_drop_cfg = point_drop_cfg

        if point_drop_cfg is not None:
            self.point_drop = PointDrop(**point_drop_cfg)
        else:
            self.point_drop = None

        if gate_type == "global":
            self.gate = nn.Parameter(torch.ones(1) * self.init_value, requires_grad=True)
        elif gate_type == "local":
            # module needs the dim attribute
            self.gate = nn.Parameter(torch.ones(module.dim) * self.init_value, requires_grad=True)
        elif gate_type == "constant":
            self.gate = nn.Parameter(torch.ones(1) * self.init_value, requires_grad=False)
        else:
            raise ValueError(
                f"Unknown gate type: {gate_type} -- must be 'global', 'local' or 'constant'."
            )

    def __repr__(self):
        inner_repr = self.module.__repr__().replace("\n", "\n  ")
        return (
            f"Residual(\n"
            f"  (inner_module): {inner_repr},\n"
            f"  gate_type={self.gate_type},\n"
            f"  init_value={self.init_value},\n"
            f"  point_drop={self.point_drop}\n"
            ")"
        )

    def forward(self, x, **kwargs):
        if self.point_drop is not None:
            x = self.point_drop(x)

        if kwargs.get("mask") is not None:
            x = x * kwargs["mask"].unsqueeze(-1)

        return x + self.module(x, **kwargs) * self.gate


class AttentionBlock(nn.Module):
    """Attention block with optional layernorm before and/or after."""

    def __init__(
        self,
        dim,
        num_heads=8,
        dropout_rate=0.0,
        norm_before: bool = True,
        norm_after: bool = False,
        need_weights: bool = False,
    ):
        """
        Parameters
        ----------
        dim: int
            Dimension of the input and output.
        num_heads: int
            Number of attention heads.
        dropout_rate: float
            Dropout rate for the attention module.
        norm_before: bool
            If True, apply layer normalization before the attention module.
        norm_after: bool
            If True, apply layer normalization after the attention module.
        need_weights: bool
            If True, return the attention weights as well.
        """
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.need_weights = need_weights

        self.pre_attn_norm = nn.LayerNorm(self.dim) if norm_before else None
        self.attn = nn.MultiheadAttention(
            embed_dim=self.dim,
            num_heads=self.num_heads,
            batch_first=True,
            dropout=self.dropout_rate,
        )
        self.post_attn_norm = nn.LayerNorm(self.dim) if norm_after else None

    def __repr__(self):
        attn_repr = (
            "torch.nn.MultiheadAttention(\n"
            "  Parameters:\n  "
            # add parameter names and counts
            + "\n  ".join(
                ["{}={}".format(k, torch.numel(v)) for k, v in self.attn.named_parameters()]
            )
            + "\n)"
        ).replace("\n", "\n  ")

        return (
            f"AttentionBlock(\n"
            f"  dim={self.dim}\n"
            f"  num_heads={self.num_heads}\n"
            f"  dropout_rate={self.dropout_rate}\n"
            f"  (pre_attn_norm): {self.pre_attn_norm}\n"
            f"  (post_attn_norm): {self.post_attn_norm}\n"
            f"  (attn_module): {attn_repr}\n"
            ")"
        )

    def forward(self, x, **kwargs):
        """Forward pass through the attention block."""

        mask = kwargs.get("mask", None)
        if mask is None:
            mask = torch.ones(x.size(0), x.size(1), device=x.device)

        attn_mask = kwargs.get("attn_mask", None)

        x = x * mask.unsqueeze(-1)

        if self.pre_attn_norm is not None:
            x = self.pre_attn_norm(x)

        attn_output, attn_weights = self.attn(
            x,
            x,
            x,
            key_padding_mask=torch.where(
                mask != 1,
                torch.tensor(float("-inf")),
                torch.tensor(0.0),
            ),
            attn_mask=attn_mask,
            need_weights=self.need_weights,
        )

        if self.post_attn_norm is not None:
            attn_output = self.post_attn_norm(attn_output)

        attn_output = attn_output * mask.unsqueeze(-1)

        if kwargs.get("return_attn_weights", False):
            return attn_output, attn_weights

        return attn_output


class MLPBlock(nn.Module):
    """MLP block with optional layernorm before."""

    def __init__(
        self,
        dim,
        expansion_factor: int = 2,
        dropout_rate: float = 0.0,
        norm_before: bool = True,
        norm_between: bool = False,
        activation: str = "GELU",
    ):
        super().__init__()
        self.dim = dim
        self.expansion_factor = expansion_factor
        self.dropout_rate = dropout_rate
        self.act_fn = eval(f"torch.nn.{activation}")

        self.pre_mlp_norm = nn.LayerNorm(self.dim) if norm_before else None
        self.norm_between = (
            nn.LayerNorm(self.dim * self.expansion_factor) if norm_between else None
        )

        layer_list = [
            nn.Linear(self.dim, self.dim * self.expansion_factor),
            self.act_fn(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.dim * self.expansion_factor, self.dim),
        ]

        # add the layernorm between the two linear layers if requested
        if self.norm_between is not None:
            layer_list.insert(2, self.norm_between)

        self.layers = nn.Sequential(*layer_list)

    def __repr__(self):
        layers_repr = self.layers.__repr__().replace("\n", "\n  ")
        return f"MLPBlock(\n  (pre_mlp_norm): {self.pre_mlp_norm}\n  (layers): {layers_repr}\n)"

    def forward(self, x, **kwargs):
        if self.pre_mlp_norm is not None:
            x = self.pre_mlp_norm(x)

        x = self.layers(x)

        if kwargs.get("mask") is not None:
            x = x * kwargs["mask"].unsqueeze(-1)

        return x


class EncoderBlock(nn.Module):
    """Residual MHA and residual MLP block."""

    def __init__(
        self,
        attn_cfg=None,
        mlp_cfg=None,
        residual_cfg=None,
        attn_point_drop_cfg=None,
    ):
        super().__init__()
        self.attn = Residual(
            AttentionBlock(**attn_cfg),
            **residual_cfg,
            point_drop_cfg=attn_point_drop_cfg,
        )
        self.mlp = Residual(MLPBlock(**mlp_cfg), **residual_cfg)

    def set_residuals_to_identity(self):
        """Will set the residuals to identity by setting the gate to 0.
        This should not be used for initialization, but for studies downstream
        when we want to see the effect of the residuals.
        """
        self.attn.gate.data.fill_(0)
        self.mlp.gate.data.fill_(0)

    def forward(self, x, **kwargs):
        x = self.attn(x, **kwargs)
        x = self.mlp(x)

        mask = kwargs.get("mask", None)

        if mask is not None:  # just to be sure apply mask again
            x = x * mask.unsqueeze(-1)
        return x


class ClassAttentionBlock(nn.Module):
    """Class attention block. I.e. cross-attention between a class token and the sequence.
    (+ an MLP block)."""

    def __init__(
        self,
        dim,
        mlp_cfg: dict,
        num_heads: int = 8,
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate

        # to make this consistent with ParT by default, set the corresponding
        # parameters in the MLP block
        if "norm_before" not in mlp_cfg:
            mlp_cfg["norm_before"] = True
        if "norm_between" not in mlp_cfg:
            mlp_cfg["norm_between"] = True
        if "activation" not in mlp_cfg:
            mlp_cfg["activation"] = "GELU"

        self.pre_attn_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim,
            num_heads,
            batch_first=True,
            dropout=self.dropout_rate,
        )
        self.post_attn_norm = nn.LayerNorm(dim)
        self.mlp = MLPBlock(dim=dim, **mlp_cfg)

    def __repr__(self):
        attn_repr = (
            "torch.nn.MultiheadAttention(\n"
            "  Parameters:\n  "
            # add parameter names and counts
            + "\n  ".join(
                ["{}={}".format(k, torch.numel(v)) for k, v in self.attn.named_parameters()]
            )
            + "\n)"
        ).replace("\n", "\n  ")
        mlp_repr = self.mlp.__repr__().replace("\n", "\n  ")

        return (
            f"ClassAttentionBlock(\n"
            f"  dim={self.dim}\n"
            f"  num_heads={self.num_heads}\n"
            f"  dropout_rate={self.dropout_rate}\n"
            f"  (pre_attn_norm): {self.pre_attn_norm}\n"
            f"  (post_attn_norm): {self.post_attn_norm}\n"
            f"  (attn_module): {attn_repr}\n"
            f"  (mlp_module): {mlp_repr}\n"
            ")"
        )

    def forward(self, x, class_token, mask=None, return_attn_weights=False):
        x = x * mask.unsqueeze(-1)

        x_pre_attn = self.pre_attn_norm(x)

        attn_output, attn_weights = self.attn(
            query=class_token,
            key=x_pre_attn,
            value=x_pre_attn,
            key_padding_mask=mask != 1,
        )

        x_pre_mlp = self.post_attn_norm(attn_output) + class_token

        x_post_mlp = x_pre_mlp + self.mlp(x_pre_mlp)

        return x_post_mlp


class Transformer(nn.Module):
    """Multiple transformer blocks with residual connections."""

    def __init__(
        self,
        n_blocks: int,
        dim: int,
        attn_cfg: dict,
        mlp_cfg: dict,
        residual_cfg: dict,
        attn_point_drop_cfg: dict = None,
        post_blocks_point_drop_cfg: dict = None,
        norm_after_blocks: bool = True,
        **kwargs,
    ):
        """
        Parameters
        ----------
        n_blocks: int
            Number of transformer blocks.
        dim: int
            Dimension of the input and output.
        attn_cfg: dict
            Configuration for the attention blocks.
        mlp_cfg: dict
            Configuration for the MLP blocks.
        residual_cfg: dict
            Configuration for the residual connections.
        norm_after_blocks: bool
            If True, apply layer normalization to the final output.
        kwargs: dict
            Additional keyword arguments.
        """
        super().__init__()

        self.n_blocks = n_blocks
        self.dim = dim

        self.blocks = nn.ModuleList(
            EncoderBlock(
                attn_cfg={**attn_cfg, "dim": dim},
                mlp_cfg={**mlp_cfg, "dim": dim},
                residual_cfg=residual_cfg,
                attn_point_drop_cfg=attn_point_drop_cfg,
            )
            for _ in range(n_blocks)
        )

        self.post_blocks_point_drop = (
            PointDrop(**post_blocks_point_drop_cfg) if post_blocks_point_drop_cfg else None
        )
        self.final_norm = nn.LayerNorm(dim) if norm_after_blocks else None

    def __repr__(self):
        blocks_repr = self.blocks.__repr__().replace("\n", "\n  ")
        return (
            f"Transformer(\n"
            f"  n_blocks={self.n_blocks},\n"
            f"  dim={self.dim},\n"
            f"  (blocks): {blocks_repr},\n"
            f"  (post_blocks_point_drop): {self.post_blocks_point_drop},\n"
            f"  (final_norm): {self.final_norm},\n"
            ")"
        )

    def set_residuals_to_identity(self):
        """Will set the residuals to identity by setting the gate to 0.
        This should not be used for initialization, but for studies downstream
        when we want to see the effect of the residuals.
        """
        for block in self.blocks:
            block.set_residuals_to_identity()

    def forward(self, x, **kwargs):
        attn_mask = kwargs.get("attn_mask", None)
        # remove attn_mask from kwargs if present to avoid passing twice
        kwargs.pop("attn_mask", None)

        for i, block in enumerate(self.blocks):
            if attn_mask is not None:
                block_attn_mask = (
                    attn_mask[..., i] if attn_mask.shape[-1] > 1 else attn_mask[..., 0]
                )
            else:
                block_attn_mask = None
            x = block(x, attn_mask=block_attn_mask, **kwargs)

        # apply the post-blocks point drop if it's defined
        if self.post_blocks_point_drop is not None:
            x = self.post_blocks_point_drop(x)

        # apply the final layer normalization if used
        if self.final_norm is not None:
            x = self.final_norm(x)

        return x
