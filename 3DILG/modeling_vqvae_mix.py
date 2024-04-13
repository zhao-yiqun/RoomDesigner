import ipdb
import numpy as np

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils import weight_norm

from timm.models.registry import register_model
# from timm.models.layers import to_3tuple
from timm.models.layers import drop_path, trunc_normal_

from torch.nn import Sequential as Seq
from torch.nn import Linear as Lin
from torch.nn import ReLU
from torch.nn import BatchNorm1d as BN
from torch.nn import LayerNorm as LN

from torch_cluster import fps, knn
from torch_scatter import scatter_max, scatter_mean

from einops import rearrange


def _cfg(url='', **kwargs):
    return {
    }


class VectorQuantizer2(nn.Module):
    """
    Improved version over VectorQuantizer, can be used as a drop-in replacement. Mostly
    avoids costly matrix multiplications and allows for post-hoc remapping of indices.
    """

    # NOTE: due to a bug the beta term was applied to the wrong term. for
    # backwards compatibility we use the buggy version by default, but you can
    # specify legacy=False to fix it.
    def __init__(self, n_e, e_dim, beta=0.25, remap=None, unknown_index="random",
                 sane_index_shape=False, legacy=True):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.legacy = legacy

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        # self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
        self.embedding.weight.data.normal_(0, 1)

        self.remap = remap
        if self.remap is not None:
            self.register_buffer("used", torch.tensor(np.load(self.remap)))
            self.re_embed = self.used.shape[0]
            self.unknown_index = unknown_index  # "random" or "extra" or integer
            if self.unknown_index == "extra":
                self.unknown_index = self.re_embed
                self.re_embed = self.re_embed + 1
            print(f"Remapping {self.n_e} indices to {self.re_embed} indices. "
                  f"Using {self.unknown_index} for unknown indices.")
        else:
            self.re_embed = n_e

        self.sane_index_shape = sane_index_shape

    def remap_to_used(self, inds):
        ishape = inds.shape
        assert len(ishape) > 1
        inds = inds.reshape(ishape[0], -1)
        used = self.used.to(inds)
        match = (inds[:, :, None] == used[None, None, ...]).long()
        new = match.argmax(-1)
        unknown = match.sum(2) < 1
        if self.unknown_index == "random":
            new[unknown] = torch.randint(0, self.re_embed, size=new[unknown].shape).to(device=new.device)
        else:
            new[unknown] = self.unknown_index
        return new.reshape(ishape)

    def unmap_to_all(self, inds):
        ishape = inds.shape
        assert len(ishape) > 1
        inds = inds.reshape(ishape[0], -1)
        used = self.used.to(inds)
        if self.re_embed > self.used.shape[0]:  # extra token
            inds[inds >= self.used.shape[0]] = 0  # simply set to zero
        back = torch.gather(used[None, :][inds.shape[0] * [0], :], 1, inds)
        return back.reshape(ishape)

    def forward(self, z, temp=None, rescale_logits=False, return_logits=False):
        assert temp is None or temp == 1.0, "Only for interface compatible with Gumbel"
        assert rescale_logits == False, "Only for interface compatible with Gumbel"
        assert return_logits == False, "Only for interface compatible with Gumbel"
        # reshape z -> (batch, height, width, channel) and flatten
        # z = rearrange(z, 'b c h w -> b h w c').contiguous()
        z_flattened = z.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight ** 2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_flattened, rearrange(self.embedding.weight, 'n d -> d n'))

        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)
        perplexity = None
        min_encodings = None

        # compute loss for embedding
        if not self.legacy:
            loss = self.beta * torch.mean((z_q.detach() - z) ** 2) + \
                   torch.mean((z_q - z.detach()) ** 2)
        else:
            loss = torch.mean((z_q.detach() - z) ** 2) + self.beta * \
                   torch.mean((z_q - z.detach()) ** 2)

        # print(torch.mean(z_q**2), torch.mean(z**2))

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        # z_q = rearrange(z_q, 'b h w c -> b c h w').contiguous()

        if self.remap is not None:
            min_encoding_indices = min_encoding_indices.reshape(z.shape[0], -1)  # add batch axis
            min_encoding_indices = self.remap_to_used(min_encoding_indices)
            min_encoding_indices = min_encoding_indices.reshape(-1, 1)  # flatten

        if self.sane_index_shape:
            min_encoding_indices = min_encoding_indices.reshape(
                z_q.shape[0], z_q.shape[2], z_q.shape[3])

        return z_q, loss, perplexity, min_encoding_indices.view(z.shape[0], z.shape[1])

    def get_codebook_entry(self, indices, shape):
        # shape specifying (batch, height, width, channel)
        if self.remap is not None:
            indices = indices.reshape(shape[0], -1)  # add batch axis
            indices = self.unmap_to_all(indices)
            indices = indices.reshape(-1)  # flatten again

        # get quantized latent vectors
        z_q = self.embedding(indices)

        if shape is not None:
            z_q = z_q.view(shape)
            # reshape back to match original input shape
            z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        # x = self.drop(x)
        # commit this for the orignal BERT implement
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 attn_head_dim=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x):
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class VisionTransformer(nn.Module):
    def __init__(self,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=nn.LayerNorm,
                 init_values=0.,
                 ):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values)
            for i in range(depth)])

        self.norm = norm_layer(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {}

    def forward_features(self, x, pos_embed):
        # x = self.patch_embed(x)
        B, _, _ = x.size()

        x = x + pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        return x

    def forward(self, x, pos_embed):
        x = self.forward_features(x, pos_embed)
        return x


# class Embedding(nn.Module):
#     def __init__(self, query_channel=3, latent_channel=192):
#         super(Embedding, self).__init__()
#         # self.register_buffer('B', torch.randn((128, 3)) * 2)
#
#         self.l1 = weight_norm(nn.Linear(query_channel+latent_channel, 512))
#         self.l2 = weight_norm(nn.Linear(512, 512))
#         self.l3 = weight_norm(nn.Linear(512, 512))
#         self.l4 = weight_norm(nn.Linear(512, 512 - query_channel - latent_channel))
#         self.l5 = weight_norm(nn.Linear(512, 512))
#         self.l6 = weight_norm(nn.Linear(512, 512))
#         self.l7 = weight_norm(nn.Linear(512, 512))
#         self.l_out = weight_norm(nn.Linear(512, 1))
#
#     def forward(self, x, z):
#         # x: B x N x 3
#         # z: B x N x 192
#         input = torch.cat([x, z], dim=2)
#
#         h = F.relu(self.l1(input))
#         h = F.relu(self.l2(h))
#         h = F.relu(self.l3(h))
#         h = F.relu(self.l4(h))
#         h = torch.cat((h, input), axis=2)
#         h = F.relu(self.l5(h))
#         h = F.relu(self.l6(h))
#         h = F.relu(self.l7(h))
#         h = self.l_out(h)
#         return h

def embed(input, basis):
    # print(input.shape, basis.shape)
    projections = torch.einsum(
        'bnd,de->bne', input, basis)  # .permute(2, 0, 1)
    # print(projections.max(), projections.min())
    embeddings = torch.cat([projections.sin(), projections.cos()], dim=2)
    return embeddings  # B x N x E


class Embedding(nn.Module):
    def __init__(self, query_channel=3, latent_channel=192):
        super(Embedding, self).__init__()
        # self.register_buffer('B', torch.randn((128, 3)) * 2)

        self.embedding_dim = 48
        e = torch.pow(2, torch.arange(self.embedding_dim // 6)).float() * np.pi
        e = torch.stack([
            torch.cat([e, torch.zeros(self.embedding_dim // 6),
                       torch.zeros(self.embedding_dim // 6)]),
            torch.cat([torch.zeros(self.embedding_dim // 6), e,
                       torch.zeros(self.embedding_dim // 6)]),
            torch.cat([torch.zeros(self.embedding_dim // 6),
                       torch.zeros(self.embedding_dim // 6), e]),
        ])
        self.register_buffer('basis', e)  # 3 x 16

        self.l1 = weight_norm(nn.Linear(query_channel + latent_channel + self.embedding_dim, 512))
        self.l2 = weight_norm(nn.Linear(512, 512))
        self.l3 = weight_norm(nn.Linear(512, 512))
        self.l4 = weight_norm(nn.Linear(512, 512 - query_channel - latent_channel - self.embedding_dim))
        self.l5 = weight_norm(nn.Linear(512, 512))
        self.l6 = weight_norm(nn.Linear(512, 512))
        self.l7 = weight_norm(nn.Linear(512, 512))
        self.l_out = weight_norm(nn.Linear(512, 1))

    def forward(self, x, z):
        # x: B x N x 3
        # z: B x N x 192
        # input = torch.cat([x[:, :, None].expand(-1, -1, z.shape[1], -1), z[:, None].expand(-1, x.shape[1], -1, -1)], dim=-1)
        # print(x.shape, z.shape)

        pe = embed(x, self.basis)

        input = torch.cat([x, pe, z], dim=2)

        h = F.relu(self.l1(input))
        h = F.relu(self.l2(h))
        h = F.relu(self.l3(h))
        h = F.relu(self.l4(h))
        h = torch.cat((h, input), axis=2)
        h = F.relu(self.l5(h))
        h = F.relu(self.l6(h))
        h = F.relu(self.l7(h))
        h = self.l_out(h)
        return h


class ToRGB_layer(nn.Module):
    def __init__(self, query_channel=3, latent_channel=192):
        super(ToRGB_layer, self).__init__()
        self.embedding_dim = 48
        e = torch.pow(2, torch.arange(self.embedding_dim // 6)).float() * np.pi
        e = torch.stack([
            torch.cat([e, torch.zeros(self.embedding_dim // 6),
                       torch.zeros(self.embedding_dim // 6)]),
            torch.cat([torch.zeros(self.embedding_dim // 6), e,
                       torch.zeros(self.embedding_dim // 6)]),
            torch.cat([torch.zeros(self.embedding_dim // 6),
                       torch.zeros(self.embedding_dim // 6), e]),
        ])
        self.register_buffer('basis', e)  # 3 x 16

        self.l1 = weight_norm(nn.Linear(query_channel + latent_channel + self.embedding_dim, 512))
        self.l2 = weight_norm(nn.Linear(512, 512))
        self.l3 = weight_norm(nn.Linear(512, 512))
        self.l4 = weight_norm(nn.Linear(512, 512))
        self.l_out = weight_norm(nn.Linear(512, 3))

    def forward(self, x, z):
        pe = embed(x, self.basis)

        input = torch.cat([x, pe, z], dim=2)
        h = F.relu(self.l1(input))
        h = F.relu(self.l2(h))
        h = F.relu(self.l3(h))
        h = F.relu(self.l4(h))
        h = self.l_out(h)
        return h


class Decoder(nn.Module):
    def __init__(self, latent_channel=192):
        super().__init__()

        self.fc = Embedding(latent_channel=latent_channel // 2)
        self.to_rgb = ToRGB_layer(latent_channel=latent_channel // 2)

        self.log_sigma = nn.Parameter(torch.FloatTensor([3.0]))
        self.los_sigma_tex = nn.Parameter(torch.FloatTensor([3.0]))
        # self.register_buffer('log_sigma', torch.Tensor([-3.0]))

        self.embed = Seq(Lin(48 + 3, latent_channel))  # , nn.GELU(), Lin(128, 128))
        self.dim = latent_channel
        self.embedding_dim = 48
        e = torch.pow(2, torch.arange(self.embedding_dim // 6)).float() * np.pi
        e = torch.stack([
            torch.cat([e, torch.zeros(self.embedding_dim // 6),
                       torch.zeros(self.embedding_dim // 6)]),
            torch.cat([torch.zeros(self.embedding_dim // 6), e,
                       torch.zeros(self.embedding_dim // 6)]),
            torch.cat([torch.zeros(self.embedding_dim // 6),
                       torch.zeros(self.embedding_dim // 6), e]),
        ])
        self.register_buffer('basis', e)  # 3 x 16

        self.transformer = VisionTransformer(embed_dim=latent_channel,
                                             depth=6,
                                             num_heads=6,
                                             mlp_ratio=4.,
                                             qkv_bias=True,
                                             qk_scale=None,
                                             drop_rate=0.,
                                             attn_drop_rate=0.,
                                             drop_path_rate=0.1,
                                             norm_layer=partial(nn.LayerNorm, eps=1e-6),
                                             init_values=0.,
                                             )

    def get_surface(self, latents, centers, samples):
        embeddings = embed(centers, self.basis)
        embeddings = self.embed(torch.cat([centers, embeddings], dim=2))
        latents = self.transformer(latents, embeddings)

        pdist = (samples[:, :, None] - centers[:, None]).square().sum(dim=3)  # B x N x T
        sigma = torch.exp(self.log_sigma)
        weight = F.softmax(-pdist * sigma, dim=2)

        half_dim = self.dim // 2
        latents_s = torch.sum(weight[:, :, :, None] * latents[:, None, :, :half_dim], dim=2)  # B x N x 128
        preds = self.fc(samples, latents_s).squeeze(2)
        return preds, sigma

    def get_texture(self, latents, centers, samples):
        embeddings = embed(centers, self.basis)
        embeddings = self.embed(torch.cat([centers, embeddings], dim=2))
        # ipdb.set_trace()
        latents = self.transformer(latents, embeddings)
        pdist = (samples[:, :, None] - centers[:, None]).square().sum(dim=3)  # B x N x T
        # sigma = torch.exp(self.los_sigma_tex)
        sigma = torch.exp(self.log_sigma)
        weight = F.softmax(-pdist * sigma, dim=2)
        half_dim = self.dim // 2
        latents = torch.sum(weight[:, :, :, None] * latents[:, None, :, half_dim:], dim=2)
        #
        preds_tex = self.to_rgb(samples, latents).squeeze(2)
        return preds_tex, sigma

    def forward(self, latents, centers, samples, tex_out):
        # kernel average
        # samples: B x N x 3
        # latents: B x T x 320
        # centers: B x T x 3

        embeddings = embed(centers, self.basis)
        embeddings = self.embed(torch.cat([centers, embeddings], dim=2))
        latents = self.transformer(latents, embeddings)

        pdist = (samples[:, :, None] - centers[:, None]).square().sum(dim=3)  # B x N x T
        sigma = torch.exp(self.log_sigma)
        weight = F.softmax(-pdist * sigma, dim=2)

        # shape latents
        half_dim = self.dim // 2
        latents_s = torch.sum(weight[:, :, :, None] * latents[:, None, :, :half_dim], dim=2)  # B x N x 128
        preds = self.fc(samples, latents_s).squeeze(2)

        # Fixme: This may lead to out of memory
        pdist = (tex_out[:, :, None] - centers[:, None]).square().sum(dim=3)  # B x N x T
        # sigma = torch.exp(self.log_sigma)
        sigma_t = torch.exp(self.los_sigma_tex)
        weight = F.softmax(-pdist * sigma_t, dim=2)

        # texture latents

        B, N = weight.shape[:2]
        latents = torch.cat(
            [torch.sum(weight[i:i + 2, :, :, None] * latents[i:i + 2, None, :, half_dim:], dim=2) for i in
             range(0, B, 2)],
            dim=0)
        # import ipdb
        # ipdb.set_trace()

        # latents = torch.sum(weight[:, :, :, None] * latents[:, None, :, self.dim//2:], dim=2)  # B x N x 128
        preds_tex = self.to_rgb(tex_out, latents).squeeze(2)

        return preds, preds_tex, sigma, sigma_t


class PointConv(torch.nn.Module):
    def __init__(self, local_nn=None, global_nn=None):
        super(PointConv, self).__init__()
        self.local_nn = local_nn
        self.global_nn = global_nn

    def forward(self, pos, pos_dst, edge_index, basis=None):
        row, col = edge_index

        out = pos[row] - pos_dst[col]

        if basis is not None:
            embeddings = torch.einsum('bd,de->be', out, basis)
            embeddings = torch.cat([embeddings.sin(), embeddings.cos()], dim=1)
            out = torch.cat([out, embeddings], dim=1)

        if self.local_nn is not None:
            out = self.local_nn(out)

        out, _ = scatter_max(out, col, dim=0, dim_size=col.max().item() + 1)

        if self.global_nn is not None:
            out = self.global_nn(out)

        return out


class PointConv_texture(torch.nn.Module):
    def __init__(self, local_nn=None, global_nn=None):
        super(PointConv_texture, self).__init__()
        self.local_nn = local_nn
        self.global_nn = global_nn

    def forward(self, pos, pos_dst, edge_index, basis=None):
        row, col = edge_index

        out = pos[row, :3] - pos_dst[col, :3]

        if basis is not None:
            embeddings = torch.einsum('bd,de->be', out, basis)
            embeddings = torch.cat([embeddings.sin(), embeddings.cos()], dim=1)
            out = torch.cat([out, embeddings], dim=1)

        out = torch.cat((out, pos[row, 3:] - pos_dst[col, 3:]), dim=-1)

        if self.local_nn is not None:
            out = self.local_nn(out)

        # out, _ = scatter_max(out, col, dim=0, dim_size=col.max().item() + 1)

        out = scatter_mean(out, col, dim=0, dim_size=col.max().item() + 1)
        if self.global_nn is not None:
            out = self.global_nn(out)

        return out


class Encoder(nn.Module):
    def __init__(self, N, dim=128, M=2048):
        super().__init__()

        self.embed = Seq(Lin(48 + 3, dim * 2))  # , nn.GELU(), Lin(128, 128))

        self.embedding_dim = 48
        e = torch.pow(2, torch.arange(self.embedding_dim // 6)).float() * np.pi
        e = torch.stack([
            torch.cat([e, torch.zeros(self.embedding_dim // 6),
                       torch.zeros(self.embedding_dim // 6)]),
            torch.cat([torch.zeros(self.embedding_dim // 6), e,
                       torch.zeros(self.embedding_dim // 6)]),
            torch.cat([torch.zeros(self.embedding_dim // 6),
                       torch.zeros(self.embedding_dim // 6), e]),
        ])
        self.register_buffer('basis', e)  # 3 x 16

        # self.conv = PointConv(local_nn=Seq(weight_norm(Lin(3+self.embedding_dim, dim))))
        self.conv = PointConv(
            local_nn=Seq(weight_norm(Lin(3 + self.embedding_dim, 256)), ReLU(True), weight_norm(Lin(256, 256))),
            global_nn=Seq(weight_norm(Lin(256, 256)), ReLU(True), weight_norm(Lin(256, dim))),
        )
        self.affine_shape = 48

        self.affine_layer = Seq(Lin(3, self.affine_shape))
        self.conv_t = PointConv_texture(
            local_nn=Seq(weight_norm(Lin(3 + self.embedding_dim + self.affine_shape, 256)), ReLU(True),
                         weight_norm(Lin(256, 256))),
            global_nn=Seq(weight_norm(Lin(256, 256)), ReLU(True), weight_norm(Lin(256, dim))),
        )

        self.transformer = VisionTransformer(embed_dim=dim * 2,
                                             depth=6,
                                             num_heads=6,
                                             mlp_ratio=4.,
                                             qkv_bias=True,
                                             qk_scale=None,
                                             drop_rate=0.,
                                             attn_drop_rate=0.,
                                             drop_path_rate=0.1,
                                             norm_layer=partial(nn.LayerNorm, eps=1e-6),
                                             init_values=0.,
                                             )

        self.M = M
        self.ratio = N / M
        self.k = 32

        self.k_t = 64

    def forward(self, pc, tex_in):
        # pc: B x N x D
        B, N, D = pc.shape
        assert N == self.M  # 2048 points

        # flattened = pc.view(B*N, D)
        # #point cloud for flatten.
        #
        # batch = torch.arange(B).to(pc.device)
        # batch = torch.repeat_interleave(batch, N)
        #
        # pos = flattened
        #
        #
        # idx = fps(pos, batch, ratio=self.ratio) # 0.0625
        #
        # row, col = knn(pos, pos[idx], self.k, batch, batch[idx])
        # edge_index = torch.stack([col, row], dim=0)
        #
        # x = self.conv(pos, pos[idx], edge_index, self.basis)
        # x is the shape feature.
        # now we need to know the texture feature.
        B, N_t, T = tex_in.shape

        flattened_tex = tex_in.view(B * N_t, T)
        pos = flattened_tex[..., :3]
        batch_t = torch.arange(B).to(tex_in.device)
        batch_t = torch.repeat_interleave(batch_t, N_t)
        idx = fps(pos, batch_t, ratio=512 / 10000)

        row, col = knn(pos, pos[idx], self.k_t, batch_t, batch_t[idx])
        edge_index = torch.stack([col, row], dim=0)
        x = self.conv(pos, pos[idx], edge_index, self.basis)

        pos_t = torch.cat((pos, self.affine_layer(flattened_tex[..., 3:])), dim=-1)
        tex_x = self.conv_t(pos_t, pos_t[idx], edge_index, self.basis)

        # ipdb.set_trace()
        x = torch.cat((x, tex_x), dim=-1)
        pos, batch = pos[idx], batch_t[idx]

        x = x.view(B, -1, x.shape[-1])
        pos = pos.view(B, -1, 3)

        embeddings = embed(pos, self.basis)

        embeddings = self.embed(torch.cat([pos, embeddings], dim=2))

        # import ipdb
        # ipdb.set_trace()

        out = self.transformer(x, embeddings)

        return out, pos


class Autoencoder(nn.Module):
    def __init__(self, N, K=512, dim=256, M=2048, **kwargs):
        super().__init__()

        self.encoder = Encoder(N=N, dim=dim // 2, M=M)
        # Shape feature and texture feature should be dim//2
        self.dim = dim
        self.decoder = Decoder(latent_channel=dim)

        self.codebook_shape = VectorQuantizer2(K, dim // 2)
        self.codebook_texture = VectorQuantizer2(K, dim // 2)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {}

    def encode(self, x, tex_in, bins=256):
        B, _, _ = x.shape

        z_e_x, centers = self.encoder(x, tex_in)  # B x T x C, B x T x 3

        centers_quantized = ((centers + 1) / 2 * (bins - 1)).long()

        z_q_x_st, loss_vq, perplexity, encodings = self.codebook_shape(z_e_x[..., :self.dim // 2])
        z_q_x_st_tex, loss_vq_tex, perplexity_tex, encodings_tex = self.codebook_texture(z_e_x[..., self.dim // 2:])
        # print(z_q_x_st.shape, loss_vq.item(), perplexity.item(), encodings.shape)
        return z_e_x, torch.cat((z_q_x_st, z_q_x_st_tex), dim=-1), centers_quantized, loss_vq + loss_vq_tex, [
            perplexity, perplexity_tex], encodings, encodings_tex

    def forward(self, x, points, tex_in, tex_out):
        z_e_x, z_q_x_st, centers_quantized, loss_vq, perplexity, encodings, encodings_tex = self.encode(x, tex_in)

        centers = centers_quantized.float() / 255.0 * 2 - 1

        z_q_x = z_q_x_st

        z_q_x_st = z_q_x_st
        B, N, C = z_q_x_st.shape

        logits, logits_tex, sigma, sigma_t = self.decoder(z_q_x_st, centers, points, tex_out)

        return logits, logits_tex, z_e_x, z_q_x, sigma, sigma_t, loss_vq, perplexity


class AutoencoderKL(nn.Module):
    def __init__(self, N, dim=256, M=2048, latent_dim=256, Num_Cls=1, **kwargs):
        super().__init__()

        print(f'Autoencoder with KL-Divergence')

        # Comentts when opening pre/post operation
        # self.encoder = Encoder(N=N, dim=2*dim, M=M)
        self.encoder = Encoder(N=N, dim=dim, M=M)

        self.pre_fc = nn.Linear(dim, latent_dim * 2)
        self.post_fc = nn.Linear(latent_dim, dim)

        self.decoder = Decoder(latent_channel=dim)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {}

    def encode(self, x, return_kl=False):
        z_encode, centers = self.encoder(x)  # B x 512 x C, B x 512 x 3 centers in [-1,1]

        # Pre compression
        z_encode = self.pre_fc(z_encode)

        mu, logvar = torch.chunk(z_encode, 2, dim=-1)
        std = torch.exp(0.5 * logvar)

        z_sample = mu + std * torch.randn(mu.shape).to(device=x.device)

        kld = 0
        if return_kl:
            var = torch.exp(logvar)
            kld = 0.5 * torch.sum(
                torch.pow(mu, 2) + var - 1.0 - logvar, dim=[1, 2])
            kld = torch.sum(kld) / kld.shape[0]

        # Post compression
        z_sample = self.post_fc(z_sample)

        return 0, z_sample, centers, kld, 0, 0

    def forward(self, x, points):
        _, z, centers, kld, _, _ = self.encode(x, return_kl=True)

        logits, sigma = self.decoder(z, centers, points)

        return logits, 0, 0, sigma, 1e-5 * kld, 0


@register_model
def vaekl_512_256_2048(pretrained=False, **kwargs):
    model = AutoencoderKL(
        N=512,
        dim=256,
        M=2048,
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def vqvae_64_1024_2048(pretrained=False, **kwargs):
    model = Autoencoder(
        N=64,
        K=1024,
        M=2048,
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def vqvae_128_1024_2048(pretrained=False, **kwargs):
    model = Autoencoder(
        N=128,
        K=1024,
        M=2048,
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def vqvae_256_1024_2048(pretrained=False, **kwargs):
    model = Autoencoder(
        N=256,
        K=1024,
        M=2048,
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def vqvae_512_512_2048(pretrained=False, **kwargs):
    # import ipdb
    # ipdb.set_trace()
    model = Autoencoder(
        N=512,
        K=512,
        M=2048,
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def vqvae_512_1024_2048_tex(pretrained=False, **kwargs):
    # import ipdb
    # ipdb.set_trace()
    model = Autoencoder(
        N=512,
        K=1024,
        M=2048,
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def vqvae_1024_1024_4096(pretrained=False, **kwargs):
    model = Autoencoder(
        N=1024,
        K=1024,
        M=4096,
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def vqvae_64_1024_2048(pretrained=False, **kwargs):
    model = Autoencoder(
        N=64,
        K=1024,
        M=2048,
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model