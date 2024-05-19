import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from models.uvit import UViT, timestep_embedding, unpatchify


class OutputHead(nn.Module):
    def __init__(
        self, embed_dim: int, patch_dim: int, in_chans: int, conv: bool = True
    ):
        super().__init__()
        self.in_chans = in_chans

        self.norm = nn.LayerNorm(embed_dim)
        self.decoder_pred = nn.Linear(embed_dim, patch_dim, bias=True)
        self.final_layer = (
            nn.Conv2d(in_chans, in_chans, 3, padding=1) if conv else nn.Identity()
        )

    def forward(self, x):
        x = self.norm(x)
        x = self.decoder_pred(x)
        x = x[:, 1:, :]  # Ignore time vector
        x = unpatchify(x, self.in_chans)
        x = self.final_layer(x)
        return x


class AttentionProbe(nn.Module):
    """
    TODO: come up with a more efficient classifier!
    With embed_dim = 512, it has 788993 parameters.
    """

    def __init__(self, embed_dim, num_heads=1):
        super().__init__()
        assert embed_dim % num_heads == 0
        head_dim = embed_dim // num_heads
        self.num_heads = num_heads

        # The query is a single vector that will be learned
        self.q = nn.Parameter(torch.zeros(1, num_heads, 1, head_dim))
        self.weight_kv = nn.Linear(embed_dim, 2 * embed_dim)
        self.classification = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, 1),
        )

    def get_qkv(self, x):
        # The query is learned and does not depend on the input
        q = self.q

        # Compute key and vector from input
        kv = self.weight_kv(x)
        k, v = rearrange(kv, "b l (k h hd) -> k b h l hd", k=2, h=self.num_heads)

        return q, k, v

    def forward(self, x):
        # Ignore time vector
        x = x[:, 1:, :]

        q, k, v = self.get_qkv(x)
        x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "b h l hd -> b l (h hd)")
        x = self.classification(x)
        x = x.squeeze()
        return x


class EarlyExitUViT(nn.Module):
    def __init__(self, uvit: UViT, exit_threshold=0.2):
        super().__init__()

        self.uvit = uvit
        self.exit_threshold = exit_threshold

        # Add classifiers (they tell us if we should exit)
        self.in_blocks_classifiers = nn.ModuleList(
            [
                AttentionProbe(embed_dim=uvit.embed_dim)
                for _ in range(len(uvit.in_blocks))
            ]
        )
        self.mid_block_classifier = AttentionProbe(embed_dim=uvit.embed_dim)
        self.out_blocks_classifiers = nn.ModuleList(
            [
                AttentionProbe(embed_dim=uvit.embed_dim)
                for _ in range(len(uvit.out_blocks))
            ]
        )

        # Add output heads (in training we'll use them all the time, in inference only if there's early exit)
        self.in_blocks_heads = nn.ModuleList(
            [
                OutputHead(
                    embed_dim=self.uvit.embed_dim,
                    patch_dim=self.uvit.patch_dim,
                    in_chans=self.uvit.in_chans,
                )
                for _ in range(len(uvit.in_blocks))
            ]
        )
        self.mid_block_head = OutputHead(
            embed_dim=self.uvit.embed_dim,
            patch_dim=self.uvit.patch_dim,
            in_chans=self.uvit.in_chans,
        )
        self.out_blocks_heads = nn.ModuleList(
            [
                OutputHead(
                    embed_dim=self.uvit.embed_dim,
                    patch_dim=self.uvit.patch_dim,
                    in_chans=self.uvit.in_chans,
                )
                for _ in range(len(uvit.out_blocks))
            ]
        )

    def forward(self, x, timesteps):
        classifier_outputs = []
        outputs = []

        x = self.uvit.patch_embed(x)

        time_token = self.uvit.time_embed(
            timestep_embedding(timesteps, self.uvit.embed_dim)
        )
        time_token = time_token.unsqueeze(dim=1)
        x = torch.cat((time_token, x), dim=1)
        x = x + self.uvit.pos_embed

        skips = []

        for blk, classifier, output_head in zip(
            self.uvit.in_blocks, self.in_blocks_classifiers, self.in_blocks_heads
        ):
            outputs.append(output_head(x))
            classifier_outputs.append(classifier(x))
            x = blk(x)
            skips.append(x)

        outputs.append(self.mid_block_head(x))
        classifier_outputs.append(classifier(x))
        x = self.uvit.mid_block(x)

        for blk, classifier, output_head in zip(
            self.uvit.out_blocks, self.out_blocks_classifiers, self.out_blocks_heads
        ):
            outputs.append(output_head(x))
            classifier_outputs.append(classifier(x))
            x = blk(x, skips.pop())

        x = self.uvit.norm(x)
        x = self.uvit.decoder_pred(x)
        x = x[:, self.uvit.extras :, :]
        x = unpatchify(x, self.uvit.in_chans)
        x = self.uvit.final_layer(x)
        return x, classifier_outputs, outputs

    @property
    def device(self):
        return next(self.parameters()).device
