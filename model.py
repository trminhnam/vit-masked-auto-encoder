import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbedding(nn.Module):
    def __init__(
        self,
        img_size,
        patch_size,
        in_channels=3,
        embedding_dim=768,
    ):
        super(PatchEmbedding, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        self.projection_layer = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embedding_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x):
        """Forward pass of patch embedding layer.

        Args:
            x (torch.Tensor): input image tensor, shape (B, C, H, W)
        """

        x = self.projection_layer(
            x
        )  # (batch_size, embedding_dim, n_patches ** 0.5, n_patches ** 0.5)
        x = x.flatten(2)  # (batch_size, embedding_dim, n_patches)
        x = x.transpose(1, 2)  # (batch_size, n_patches, embedding_dim)

        return x


class Attention(nn.Module):
    def __init__(
        self,
        d_model,
        n_heads,
        qkv_bias=False,
        attn_p=0.0,
        proj_p=0.0,
    ):
        super(Attention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim**-0.5

        self.qkv_linear = nn.Linear(d_model, d_model * 3, bias=qkv_bias)
        self.attn_dropout = nn.Dropout(attn_p)

        self.proj_linear = nn.Linear(d_model, d_model)
        self.proj_dropout = nn.Dropout(proj_p)

    def forward(self, x, q_mask=None):
        n_samples, n_patches, d_model = x.shape

        if d_model != self.d_model:
            raise ValueError(
                f"Input dimension {d_model} does not match layer dimension {self.d_model}"
            )

        qkv = self.qkv_linear(x)
        qkv = qkv.reshape(
            n_samples,
            n_patches,
            3,
            self.n_heads,
            self.head_dim,
        )  # (batch_size, n_patches, 3, n_heads, head_dim)
        qkv = qkv.permute(
            2, 0, 3, 1, 4
        )  # (3, batch_size, n_heads, n_patches, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (batch_size, n_heads, n_patches, head_dim)
        # print(f"q.shape: {q.shape}")

        dp = (
            q @ k.transpose(-2, -1)
        ) * self.scale  # (batch_size, n_heads, n_patches, n_patches)
        # print(f"dp.shape: {dp.shape}")

        if q_mask is not None:
            dp = dp.masked_fill(q_mask == 0, float("-inf"))

        attn = dp.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        weighted_avg = attn @ v  # (batch_size, n_heads, n_patches, head_dim)
        weighted_avg = weighted_avg.transpose(
            1, 2
        )  # (batch_size, n_patches, n_heads, head_dim)
        weighted_avg = weighted_avg.flatten(2)  # (batch_size, n_patches, d_model)

        x = self.proj_linear(weighted_avg)
        x = self.proj_dropout(x)

        return x


class MLP(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features,
        out_features,
        p=0.0,
    ):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(p)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        d_model,
        n_heads,
        mlp_ratio=4.0,
        qkv_bias=True,
        attn_p=0.0,
        proj_p=0.0,
    ):
        super(Block, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.mlp_ratio = mlp_ratio
        self.mlp_hidden_features = int(d_model * mlp_ratio)

        self.norm1 = nn.LayerNorm(d_model)
        self.attn = Attention(
            d_model=d_model,
            n_heads=n_heads,
            qkv_bias=qkv_bias,
            attn_p=attn_p,
            proj_p=proj_p,
        )

        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = MLP(
            in_features=d_model,
            hidden_features=self.mlp_hidden_features,
            out_features=d_model,
        )

    def forward(self, x, q_mask=None):
        x = x + self.attn(self.norm1(x), q_mask=q_mask)
        x = x + self.mlp(self.norm2(x))

        return x


class ViT(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_channels=3,
        num_classes=1000,
        embedding_dim=768,
        depth=12,
        n_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        attn_p=0.0,
        p=0.0,
        fine_tune=False,
    ):
        super(ViT, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.depth = depth

        # input
        self.patch_embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embedding_dim=embedding_dim,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, (img_size // patch_size) ** 2 + 1, embedding_dim)
        )
        self.pos_dropout = nn.Dropout(p=p)

        # transformer
        self.blocks = nn.ModuleList(
            [
                Block(
                    d_model=embedding_dim,
                    n_heads=n_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    attn_p=attn_p,
                    proj_p=p,
                )
                for _ in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(embedding_dim)
        self.head = nn.Linear(embedding_dim, num_classes)

    def forward(self, x, q_mask=None, get_features=False):
        batch_size = x.shape[0]
        x = self.patch_embed(x)

        # add cls token and position embedding
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        # print(f"cls_tokens.shape: {cls_tokens.shape}")
        # print(f"x.shape: {x.shape}")
        x = torch.cat((cls_tokens, x), dim=1)
        if q_mask is not None:
            q_mask = F.pad(q_mask, (1, 0), value=True)
        x = x + self.pos_embed
        x = self.pos_dropout(x)

        # transformer
        for block in self.blocks:
            x = block(x, q_mask=q_mask)

        transformer_features = self.norm(x)

        # classification head
        x = self.head(transformer_features[:, 0])

        if get_features:
            return x, transformer_features
        return x


# # implement ViT
# class ViT(nn.Module):
#     def __init__(
#         self,
#         image_size=224,
#         patch_size=16,
#         num_classes=1000,
#         dim=768,
#         depth=12,
#         heads=12,
#         mlp_dim=3072,
#         dropout=0.1,
#         emb_dropout=0.1,
#     ):
#         super(ViT, self).__init__()
#         num_patches = (image_size // patch_size) ** 2
#         patch_dim = 3 * patch_size**2

#         self.patch_size = patch_size
#         self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
#         self.patch_to_embedding = nn.Linear(patch_dim, dim)
#         self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
#         self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout)
#         self.to_cls_token = nn.Identity()

#         self.mlp_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))

#     def forward(self, x):
#         b, c, h, w = x.shape
#         assert h == w, "image height and width must be equal"
#         assert h % self.patch_size == 0, "image height must be multiple of patch size"
#         p = self.patch_size

#         # flatten the image
#         x = x.reshape(b, c, h // p, p, w // p, p)
#         x = x.permute(0, 2, 4, 1, 3, 5)
#         x = x.reshape(b, -1, c * p * p)

#         # patch to embedding
#         x = self.patch_to_embedding(x)

#         # concatenate cls token
#         cls_tokens = self.cls_token.expand(b, -1, -1)
#         x = torch.cat((cls_tokens, x), dim=1)

#         # add position embedding
#         x += self.pos_embedding[:, : (x.shape[1])]

#         # transformer
#         x = self.transformer(x)

#         # take cls token
#         x = self.to_cls_token(x[:, 0])

#         # mlp head
#         x = self.mlp_head(x)

#         return x
