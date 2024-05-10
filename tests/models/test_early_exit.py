import torch

from models.early_exit import AttentionProbe, OutputHead


def test_output_head():
    output_head = OutputHead(embed_dim=512, patch_dim=2**2 * 3, in_chans=3, conv=True)

    x = torch.zeros((16, 257, 512))

    y = output_head(x)
    assert y.shape == (16, 3, 32, 32)


def test_attention_probe():
    attention_probe = AttentionProbe(embed_dim=512, num_heads=1)

    x = torch.zeros((16, 257, 512))
    y = attention_probe(x)

    assert y.shape == (16,)
