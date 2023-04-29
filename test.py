import numpy as np
import torch
from model import ViT


def get_num_params(model):
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return num_params


def assert_tensors_equal(t1, t2):
    assert t1.shape == t2.shape
    assert t1.dtype == t2.dtype
    a1 = t1.detach().numpy()
    a2 = t2.detach().numpy()
    assert np.allclose(a1, a2)


config = {
    "img_size": 224,
    "patch_size": 74,
    "in_channels": 3,
    "num_classes": 1000,
    "embedding_dim": 768,
    "depth": 12,
    "n_heads": 12,
    "mlp_ratio": 4.0,
    "qkv_bias": True,
    "attn_p": 0.0,
    "p": 0.0,
}

model = ViT(**config)
model.eval()

# inputs = torch.randn(1, 3, 224, 224)
# outputs = model(inputs)
inputs = torch.randn(1, 3, 224, 224)
mask = torch.from_numpy(np.array([[1, 0, 1, 0, 1, 0, 0, 1, 0]]))
outputs = model(inputs, q_mask=mask)
