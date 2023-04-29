import numpy as np
import torch
import torch.nn as nn
from model import ViT, MaskedAutoencoderViT


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
    "img_size": 48,
    "patch_size": 16,
    "in_channels": 3,
    "num_classes": 2,
    "embedding_dim": 768,
    "depth": 12,
    "n_heads": 12,
    "mlp_ratio": 4.0,
    "qkv_bias": True,
    "attn_p": 0.0,
    "p": 0.0,
}

# model = ViT(**config)
# model.eval()

# # inputs = torch.randn(1, 3, 224, 224)
# # outputs = model(inputs)
# inputs = torch.randn(1, 3, 224, 224)
# mask = torch.from_numpy(np.array([[1, 0, 1, 0, 1, 0, 0, 1, 0]]))
# outputs = model(inputs, q_mask=mask)

model = MaskedAutoencoderViT(**config)
inputs = torch.randn(11, 3, 48, 48)
targets = torch.randint(0, config["num_classes"], (11,))

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
inputs = inputs.to(device)
targets = targets.to(device)

# pretrained testing
outputs = model(inputs, mask_ratio=0.5, stage="pretrain")
print("Pretrained outputs test passed!")
print("#" * 50 + "\n")

# fine-tuning testing
outputs = model(inputs, mask_ratio=0.5, stage="finetune")
print("Fine-tuning outputs test passed!")
print("#" * 50 + "\n")

# combine testing
outputs = model(inputs, mask_ratio=0.5, stage="combine")
print("Combine outputs test passed!")
print("#" * 50 + "\n")

# test pretrained backprop
print("Testing pretrained backprop...")
for _ in range(10):
    outputs = model(inputs, mask_ratio=0.5, stage="pretrain")
    loss = outputs["loss"]
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"Loss = {loss.item():.4f}")
print(f"Pretrained backprop test passed!")
print("#" * 50 + "\n")

# test fine-tuning backprop
print("Testing fine-tuning backprop...")
for _ in range(10):
    outputs = model(inputs, mask_ratio=0.5, stage="finetune")
    cls_loss = criterion(outputs["pred_cls"], targets)
    loss = cls_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"Cls loss = {cls_loss.item():.4f}")
print(f"Fine-tuning backprop test passed!")
print("#" * 50 + "\n")

# test combine backprop
print("Testing combine backprop...")
for _ in range(10):
    outputs = model(inputs, mask_ratio=0.5, stage="combine")
    cls_loss = criterion(outputs["pred_cls"], targets)
    loss = outputs["loss"] + cls_loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"Cls loss = {cls_loss.item():.4f}, Mask loss = {outputs['loss'].item():.4f}")
print(f"Combine backprop test passed!")
print("#" * 50 + "\n")

print(f"Pred: {model.predict(inputs)}")
print(f"Tgts: {targets}")
