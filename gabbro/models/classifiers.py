import torch
import torch.nn as nn

from gabbro.models.transformer import (
    ClassAttentionBlock,  # noqa: E402
    NormformerStack,
)
from gabbro.utils.pylogger import get_pylogger

logger = get_pylogger(__name__)


class NormformerCrossBlock(nn.Module):
    def __init__(self, input_dim, num_heads, dropout_rate=0.1, mlp_dim=None, **kwargs):
        super().__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate

        self.norm1 = nn.LayerNorm(input_dim)
        self.attn = nn.MultiheadAttention(input_dim, num_heads, batch_first=True, dropout=0.1)
        nn.init.zeros_(self.norm1.weight)

    def forward(self, x, class_token, mask=None, return_attn_weights=False):
        # x: (B, S, F)
        # mask: (B, S)
        x = x * mask.unsqueeze(-1)

        # calculate cross-attention
        x_norm = self.norm1(x)
        attn_output, attn_weights = self.attn(
            query=class_token, key=x_norm, value=x_norm, key_padding_mask=mask != 1
        )
        return attn_output


class NormformerCrossBlockv2(nn.Module):
    def __init__(
        self,
        input_dim,
        mlp_expansion_factor: int = 4,
        num_heads: int = 8,
        dropout_rate=0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate

        # TODO: re-implement the unused parameters (currently commented out)
        # define the MultiheadAttention layer with layer normalization
        self.norm1 = nn.LayerNorm(input_dim)
        self.attn = nn.MultiheadAttention(
            input_dim, num_heads, batch_first=True, dropout=self.dropout_rate
        )
        self.norm2 = nn.LayerNorm(input_dim)

        # define the MLP with layer normalization
        self.mlp = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, input_dim * mlp_expansion_factor),
            nn.Dropout(dropout_rate),
            nn.SiLU(),
            nn.LayerNorm(input_dim * mlp_expansion_factor),
            nn.Linear(input_dim * mlp_expansion_factor, input_dim),
            nn.Dropout(dropout_rate),
        )

        # initialize weights of mlp[-1] and layer norm after attn block to 0
        # such that the residual connection is the identity when the block is
        # initialized
        # nn.init.zeros_(self.mlp[-1].weight)
        # nn.init.zeros_(self.mlp[-1].bias)
        nn.init.zeros_(self.norm1.weight)

    def forward(self, x, class_token, mask=None, return_attn_weights=False):
        # x: (B, S, F)
        # mask: (B, S)
        x = x * mask.unsqueeze(-1)

        # calculate cross-attention
        x_pre_attn = self.norm1(x)
        x_post_attn, attn_weights = self.attn(
            query=class_token, key=x_pre_attn, value=x_pre_attn, key_padding_mask=mask != 1
        )
        x_pre_mlp = self.norm2(x_post_attn) + class_token
        x_post_mlp = x_pre_mlp + self.mlp(x_pre_mlp)
        return x_post_mlp


class ClassifierTransformer(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_heads: int = 1,
        num_enc_blocks: int = 0,
        num_class_blocks: int = 3,
        fc_params: list = None,
        n_out_nodes: int = None,
        dropout_rate: float = 0.1,
        self_attention_model_class: str = "GPT_Decoder",
        cross_attention_model_class: str = "NormformerCrossBlock",
        identity_init: bool = False,
        **kwargs,
    ):
        super().__init__()

        if n_out_nodes is None:
            self.n_out_nodes = 10
        else:
            self.n_out_nodes = n_out_nodes

        self.fc_params = fc_params
        self.dropout_rate = dropout_rate
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_enc_blocks = num_enc_blocks
        self.num_class_blocks = num_class_blocks
        self.class_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.identity_init = identity_init

        self.input_projection = nn.Linear(self.input_dim, self.hidden_dim)

        if self.num_enc_blocks == 0:
            self.self_attention_blocks = None
        else:
            if self_attention_model_class == "Normformer":
                self.self_attention_blocks = NormformerStack(
                    hidden_dim=self.hidden_dim,
                    num_heads=self.num_heads,
                    num_blocks=self.num_enc_blocks,
                    dropout_rate=self.dropout_rate,
                    init_identity=self.identity_init,
                )
            else:
                raise ValueError(
                    f"Self-attention model class {self_attention_model_class} not supported."
                    "Supported choices are 'Normformer' and 'GPT_Decoder'."
                )

        if cross_attention_model_class == "ClassAttentionBlock":
            class_attn_kwargs = dict(
                dim=self.hidden_dim,
                num_heads=self.num_heads,
                mlp_cfg=dict(
                    expansion_factor=4,
                    dropout_rate=0,
                    norm_before=True,
                    norm_between=True,
                    activation="GELU",
                ),
                dropout_rate=0,
            )
        else:
            class_attn_kwargs = dict(
                input_dim=self.hidden_dim,
                num_heads=self.num_heads,
                dropout_rate=self.dropout_rate,
                mlp_expansion_factor=4,
            )

        self.class_attention_blocks = nn.ModuleList(
            [
                eval(cross_attention_model_class)(**class_attn_kwargs)  # nosec
                for _ in range(self.num_class_blocks)
            ]
        )
        self.initialize_classification_head()

        self.loss_history = []
        self.lr_history = []

    def forward(self, x, mask):
        # encode
        x = self.input_projection(x)
        if self.self_attention_blocks is not None:
            x = self.self_attention_blocks(x, mask)
        # concatenate class token and x
        class_token = self.class_token.expand(x.size(0), -1, -1)
        mask_with_token = torch.cat([torch.ones(x.size(0), 1).to(x.device), mask], dim=1)

        # pass through class attention blocks, always use the updated class token
        for block in self.class_attention_blocks:
            x_class_token_and_x_encoded = torch.cat([class_token, x], dim=1)
            class_token = block(
                x=x_class_token_and_x_encoded, class_token=class_token, mask=mask_with_token
            )

        return self.final_mlp(class_token.squeeze(1))

    def initialize_classification_head(self):
        if self.fc_params is None:
            self.final_mlp = nn.Linear(self.hidden_dim, self.n_out_nodes)
        else:
            fc_params = [[self.hidden_dim, 0]] + self.fc_params
            layers = []

            for i in range(1, len(fc_params)):
                in_dim = fc_params[i - 1][0]
                out_dim = fc_params[i][0]
                dropout_rate = fc_params[i][1]
                layers.extend(
                    [
                        nn.Linear(in_dim, out_dim),
                        nn.Dropout(dropout_rate),
                        nn.ReLU(),
                    ]
                )
            # add final layer
            layers.extend([nn.Linear(fc_params[-1][0], self.n_out_nodes)])
            self.final_mlp = nn.Sequential(*layers)
