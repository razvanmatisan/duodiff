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

    def forward(self, x, extras):
        x = self.norm(x)
        x = self.decoder_pred(x)
        x = x[:, extras:, :]  # Ignore time (and class) vector
        x = unpatchify(x, self.in_chans)
        x = self.final_layer(x)
        return x


class MLPProbe(nn.Module):
    def __init__(self, embed_dim):
        super(MLPProbe, self).__init__()
        self.classifier = nn.Sequential(nn.Linear(embed_dim, 1), nn.Sigmoid())

    def forward(self, x):
        return self.classifier(x).mean(dim=1).squeeze()


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


# Old Early Exit UViT for loading old checkpoints
# It works only for classifier_type:
#   - attention_probe
#   - mlp_probe_per_layer
class OldEarlyExitUViT(nn.Module):
    def __init__(
        self, uvit: UViT, classifier_type="attention_probe", exit_threshold=0.2
    ):
        super().__init__()

        self.uvit = uvit
        self.exit_threshold = exit_threshold

        # Add classifiers (they tell us if we should exit)
        self.in_blocks_classifiers = nn.ModuleList(
            [
                AttentionProbe(embed_dim=uvit.embed_dim)
                if classifier_type == "attention_probe"
                else MLPProbe(embed_dim=self.uvit.embed_dim)
                for _ in range(len(uvit.out_blocks))
            ]
        )
        self.mid_block_classifier = (
            AttentionProbe(embed_dim=uvit.embed_dim)
            if classifier_type == "attention_probe"
            else MLPProbe(embed_dim=self.uvit.embed_dim)
        )

        self.out_blocks_classifiers = nn.ModuleList(
            [
                AttentionProbe(embed_dim=uvit.embed_dim)
                if classifier_type == "attention_probe"
                else MLPProbe(embed_dim=self.uvit.embed_dim)
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


class EarlyExitUViT(nn.Module):
    def get_classifer(self, t, i) -> nn.Module:
        if self.classifier_type == "attention_probe":
            return self.matrix[f"{i}"]
        elif self.classifier_type == "mlp_probe_per_layer":
            return self.matrix[f"{i}"]
        elif self.classifier_type == "mlp_probe_per_timestep":
            return self.matrix[f"{t}"]
        elif self.classifier_type == "mlp_probe_per_layer_per_timestep":
            return self.matrix[f"{i}, {t}"]
        else:
            ValueError(f"Unknown classifier type: {self.classifier_type}")

    def __init__(
        self, uvit: UViT, classifier_type="attention_probe", exit_threshold=0.2
    ):
        super().__init__()

        self.uvit = uvit
        self.exit_threshold = exit_threshold
        self.classifier_type = classifier_type

        # !!! Change get_classifier as well if you change this!!!
        # TODO: FIXME (right now we have to modify this both in get_classifier and here)
        if classifier_type == "attention_probe":
            self.matrix = nn.ModuleDict(
                {
                    f"{i}": AttentionProbe(embed_dim=uvit.embed_dim)
                    for i in range(uvit.depth)
                }
            )
        elif classifier_type == "mlp_probe_per_layer":
            self.matrix = nn.ModuleDict(
                {f"{i}": MLPProbe(embed_dim=uvit.embed_dim) for i in range(uvit.depth)}
            )
        elif classifier_type == "mlp_probe_per_timestep":
            self.matrix = nn.ModuleDict(
                {f"{t}": MLPProbe(embed_dim=uvit.embed_dim) for t in range(1000)}
            )
        elif classifier_type == "mlp_probe_per_layer_per_timestep":
            self.matrix = nn.ModuleDict(
                {
                    f"{i}, {t}": MLPProbe(embed_dim=uvit.embed_dim)
                    for t in range(1000)
                    for i in range(uvit.depth)
                }
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

    def forward(self, x, timesteps, y=None):
        t = int(timesteps[0])
        if self.uvit.normalize_timesteps:
            timesteps = timesteps.float() / 1000

        classifier_outputs = []
        outputs = []

        x = self.uvit.patch_embed(x)

        time_token = self.uvit.time_embed(
            timestep_embedding(timesteps, self.uvit.embed_dim)
        )
        time_token = time_token.unsqueeze(dim=1)
        x = torch.cat((time_token, x), dim=1)
        if y is not None and self.uvit.label_emb is not None:
            label_emb = self.uvit.label_emb(y)
            label_emb = label_emb.unsqueeze(dim=1)
            x = torch.cat((label_emb, x), dim=1)
        x = x + self.uvit.pos_embed

        skips = []

        for blk, layer_id, output_head in zip(
            self.uvit.in_blocks, range(self.uvit.depth // 2), self.in_blocks_heads
        ):
            classifier = self.get_classifer(t, layer_id)
            outputs.append(output_head(x, self.uvit.extras))
            classifier_outputs.append(classifier(x))
            x = blk(x)
            skips.append(x)

        classifier = self.get_classifer(t, self.uvit.depth // 2)
        outputs.append(self.mid_block_head(x, self.uvit.extras))
        classifier_outputs.append(classifier(x))
        x = self.uvit.mid_block(x)

        for blk, layer_id, output_head in zip(
            self.uvit.out_blocks,
            range(self.uvit.depth // 2 + 1, self.uvit.depth),
            self.out_blocks_heads,
        ):
            classifier = self.get_classifer(t, layer_id)
            outputs.append(output_head(x, self.uvit.extras))
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
