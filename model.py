import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import get_2d_sincos_pos_embed


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

    def get_num_patches(self):
        return self.n_patches


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


class MaskedAutoencoderViT(nn.Module):
    """Masked Autoencoder with VisionTransformer backbone"""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_channels=3,
        num_classes=1000,
        embedding_dim=1024,
        depth=24,
        n_heads=16,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.0,
        qkv_bias=True,
        norm_layer=nn.LayerNorm,
        norm_pix_loss=False,
        attn_p=0.0,
        p=0.0,
    ):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbedding(
            img_size, patch_size, in_channels, embedding_dim
        )
        num_patches = self.patch_embed.get_num_patches()

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embedding_dim), requires_grad=False
        )  # fixed sin-cos embedding

        self.blocks = nn.ModuleList(
            [
                Block(
                    embedding_dim,
                    n_heads,
                    mlp_ratio,
                    qkv_bias=qkv_bias,
                    attn_p=attn_p,
                    proj_p=p,
                )
                for _ in range(depth)
            ]
        )
        self.norm = norm_layer(embedding_dim)
        self.cls_head = nn.Linear(embedding_dim, num_classes)  # classification head
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embedding_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False
        )  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList(
            [
                Block(
                    decoder_embed_dim,
                    decoder_num_heads,
                    mlp_ratio,
                    qkv_bias=qkv_bias,
                    attn_p=attn_p,
                    proj_p=p,
                )
                for _ in range(decoder_depth)
            ]
        )

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(
            decoder_embed_dim, patch_size**2 * in_channels, bias=True
        )  # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def forward(self, imgs, stage, mask_ratio=0.75):
        if stage == "pretrain":
            latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio, stage)
            pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
            loss = self.forward_loss(imgs, pred, mask)
            # return loss, pred, mask
            return {"loss": loss, "pred": pred, "mask": mask}

        elif stage == "finetune":
            latent, _, _ = self.forward_encoder(imgs, mask_ratio, stage)
            cls_token = latent[:, 0, :]
            pred_cls = self.cls_head(cls_token)
            # return pred_cls
            return {"pred_cls": pred_cls}

        elif stage == "combine":
            latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio, stage)
            pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
            loss = self.forward_loss(imgs, pred, mask)
            cls_token = latent[:, 0, :]
            pred_cls = self.cls_head(cls_token)
            # return loss, pred, mask, pred_cls
            return {
                "loss": loss,
                "pred": pred,
                "mask": mask,
                "pred_cls": pred_cls,
            }

        else:
            raise ValueError(
                "stage must be one of 'pretrain', 'finetune', 'combine'",
            )

    def classify(self, imgs):
        latent, _, _ = self.forward_encoder(imgs, mask_ratio=0.75, stage="finetune")
        cls_token = latent[:, 0, :]
        pred = self.cls_head(cls_token)
        pred = F.softmax(pred, dim=-1)
        pred = pred.argmax(dim=-1)
        return pred

    def predict(self, imgs):
        return self.classify(imgs)

    def forward_encoder(self, x, mask_ratio, stage):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        if stage != "finetune":
            x, mask, ids_restore = self.random_masking(x, mask_ratio)
        else:
            mask = None
            ids_restore = None

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        if stage == "finetune":
            q_mask = self.get_query_facial_mask(x)
        else:
            q_mask = None

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x, q_mask=q_mask)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(
            x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1
        )
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(
            x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2])
        )  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def initialize_weights(self):
        num_patches = self.patch_embed.get_num_patches()
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(num_patches**0.5),
            cls_token=True,
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1],
            int(num_patches**0.5),
            cls_token=True,
        )
        self.decoder_pos_embed.data.copy_(
            torch.from_numpy(decoder_pos_embed).float().unsqueeze(0)
        )

        w = self.patch_embed.projection_layer.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        torch.nn.init.normal_(self.cls_token, std=0.02)
        torch.nn.init.normal_(self.mask_token, std=0.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(
            noise, dim=1
        )  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum("nchpwq->nhwpqc", x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def get_query_facial_mask(self, x):
        N, L, D = x.shape
        single_mask = [1] + [1, 0, 1, 0, 1, 0, 0, 1, 0]  # cls + mask 9 patches

        # masks is repeated for each sample
        mask = torch.tensor(single_mask, device=x.device).repeat(N, 1, 1, 1)
        return mask.to(x.device)
