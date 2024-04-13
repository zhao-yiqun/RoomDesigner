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
from unet import UNet
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
            self.unknown_index = unknown_index # "random" or "extra" or integer
            if self.unknown_index == "extra":
                self.unknown_index = self.re_embed
                self.re_embed = self.re_embed+1
            print(f"Remapping {self.n_e} indices to {self.re_embed} indices. "
                  f"Using {self.unknown_index} for unknown indices.")
        else:
            self.re_embed = n_e

        self.sane_index_shape = sane_index_shape

    def remap_to_used(self, inds):
        ishape = inds.shape
        assert len(ishape)>1
        inds = inds.reshape(ishape[0],-1)
        used = self.used.to(inds)
        match = (inds[:,:,None]==used[None,None,...]).long()
        new = match.argmax(-1)
        unknown = match.sum(2)<1
        if self.unknown_index == "random":
            new[unknown]=torch.randint(0,self.re_embed,size=new[unknown].shape).to(device=new.device)
        else:
            new[unknown] = self.unknown_index
        return new.reshape(ishape)

    def unmap_to_all(self, inds):
        ishape = inds.shape
        assert len(ishape)>1
        inds = inds.reshape(ishape[0],-1)
        used = self.used.to(inds)
        if self.re_embed > self.used.shape[0]: # extra token
            inds[inds>=self.used.shape[0]] = 0 # simply set to zero
        back=torch.gather(used[None,:][inds.shape[0]*[0],:], 1, inds)
        return back.reshape(ishape)

    def forward(self, z, temp=None, rescale_logits=False, return_logits=False):
        assert temp is None or temp==1.0, "Only for interface compatible with Gumbel"
        assert rescale_logits==False, "Only for interface compatible with Gumbel"
        assert return_logits==False, "Only for interface compatible with Gumbel"
        # reshape z -> (batch, height, width, channel) and flatten
        # z = rearrange(z, 'b c h w -> b h w c').contiguous()
        z_flattened = z.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_flattened, rearrange(self.embedding.weight, 'n d -> d n'))

        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)
        perplexity = None
        min_encodings = None

        # compute loss for embedding
        if not self.legacy:
            loss = self.beta * torch.mean((z_q.detach()-z)**2) + \
                   torch.mean((z_q - z.detach()) ** 2)
        else:
            loss = torch.mean((z_q.detach()-z)**2) + self.beta * \
                   torch.mean((z_q - z.detach()) ** 2)
        
        # print(torch.mean(z_q**2), torch.mean(z**2))

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        # z_q = rearrange(z_q, 'b h w c -> b c h w').contiguous()

        if self.remap is not None:
            min_encoding_indices = min_encoding_indices.reshape(z.shape[0],-1) # add batch axis
            min_encoding_indices = self.remap_to_used(min_encoding_indices)
            min_encoding_indices = min_encoding_indices.reshape(-1,1) # flatten

        if self.sane_index_shape:
            min_encoding_indices = min_encoding_indices.reshape(
                z_q.shape[0], z_q.shape[2], z_q.shape[3])

        return z_q, loss, perplexity, min_encoding_indices.view(z.shape[0], z.shape[1])

    def get_codebook_entry(self, indices, shape):
        # shape specifying (batch, height, width, channel)
        if self.remap is not None:
            indices = indices.reshape(shape[0],-1) # add batch axis
            indices = self.unmap_to_all(indices)
            indices = indices.reshape(-1) # flatten again

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
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

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
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
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

        self.norm =  norm_layer(embed_dim)

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

        self.l1 = weight_norm(nn.Linear(query_channel+latent_channel+self.embedding_dim, 512))
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
    def __init__(self,  query_channel=3, latent_channel=192):
        super(ToRGB_layer, self).__init__()
        self.embedding_dim = 24
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


# class Decoder(nn.Module):
#     def __init__(self, latent_channel=192):
#         super().__init__()
#
#         self.fc = Embedding(latent_channel=latent_channel//2)
#         self.to_rgb = ToRGB_layer(latent_channel = latent_channel//2)
#
#         self.log_sigma = nn.Parameter(torch.FloatTensor([3.0]))
#         self.los_sigma_tex = nn.Parameter(torch.FloatTensor([3.0]))
#         # self.register_buffer('log_sigma', torch.Tensor([-3.0]))
#
#         self.embed = Seq(Lin(48+3, latent_channel))#, nn.GELU(), Lin(128, 128))
#         self.dim = latent_channel
#         self.embedding_dim = 48
#         e = torch.pow(2, torch.arange(self.embedding_dim // 6)).float() * np.pi
#         e = torch.stack([
#             torch.cat([e, torch.zeros(self.embedding_dim // 6),
#                       torch.zeros(self.embedding_dim // 6)]),
#             torch.cat([torch.zeros(self.embedding_dim // 6), e,
#                       torch.zeros(self.embedding_dim // 6)]),
#             torch.cat([torch.zeros(self.embedding_dim // 6),
#                       torch.zeros(self.embedding_dim // 6), e]),
#         ])
#         self.register_buffer('basis', e)  # 3 x 16
#
#         self.transformer = VisionTransformer(embed_dim=latent_channel,
#                                             depth=6,
#                                             num_heads=6,
#                                             mlp_ratio=4.,
#                                             qkv_bias=True,
#                                             qk_scale=None,
#                                             drop_rate=0.,
#                                             attn_drop_rate=0.,
#                                             drop_path_rate=0.1,
#                                             norm_layer=partial(nn.LayerNorm, eps=1e-6),
#                                             init_values=0.,
#                                             )
#
#     def get_surface(self, latents, centers, samples):
#         embeddings = embed(centers, self.basis)
#         embeddings = self.embed(torch.cat([centers, embeddings], dim=2))
#         latents = self.transformer(latents, embeddings)
#
#         pdist = (samples[:, :, None] - centers[:, None]).square().sum(dim=3)  # B x N x T
#         sigma = torch.exp(self.log_sigma)
#         weight = F.softmax(-pdist * sigma, dim=2)
#
#         half_dim = self.dim // 2
#         latents_s = torch.sum(weight[:, :, :, None] * latents[:, None, :, :half_dim], dim=2)  # B x N x 128
#         preds = self.fc(samples, latents_s).squeeze(2)
#         return preds, sigma
#
#     def get_texture(self, latents, centers, samples):
#         embeddings = embed(centers, self.basis)
#         embeddings = self.embed(torch.cat([centers, embeddings], dim=2))
#         latents = self.transformer(latents, embeddings)
#         pdist = (samples[:, :, None] - centers[:, None]).square().sum(dim=3)  # B x N x T
#         # sigma = torch.exp(self.los_sigma_tex)
#         sigma = torch.exp(self.log_sigma)
#         weight = F.softmax(-pdist * sigma, dim=2)
#         half_dim = self.dim // 2
#         latents = torch.sum(weight[:, :, :, None] * latents[:, None, :, half_dim:], dim=2)
#         #
#         preds_tex = self.to_rgb(samples, latents).squeeze(2)
#         return preds_tex, sigma
#     def forward(self, latents, centers, samples, tex_out):
#         # kernel average
#         # samples: B x N x 3
#         # latents: B x T x 320
#         # centers: B x T x 3
#
#         embeddings = embed(centers, self.basis)
#         embeddings = self.embed(torch.cat([centers, embeddings], dim=2))
#         latents = self.transformer(latents, embeddings)
#
#         pdist = (samples[:, :, None] - centers[:, None]).square().sum(dim=3) # B x N x T
#         sigma = torch.exp(self.log_sigma)
#         weight = F.softmax(-pdist * sigma, dim=2)
#
#         # shape latents
#         half_dim = self.dim//2
#         latents_s = torch.sum(weight[:, :, :, None] * latents[:, None, :, :half_dim], dim=2) # B x N x 128
#         preds = self.fc(samples, latents_s).squeeze(2)
#
#
#         #Fixme: This may lead to out of memory
#         pdist = (tex_out[:, :, None] - centers[:, None]).square().sum(dim=3)  # B x N x T
#         # sigma = torch.exp(self.log_sigma)
#         sigma_t = torch.exp(self.los_sigma_tex)
#         weight = F.softmax(-pdist * sigma_t, dim=2)
#
#         # texture latents
#
#         B, N = weight.shape[:2]
#         latents = torch.cat(
#             [torch.sum(weight[i:i + 2, :, :, None] * latents[i:i + 2, None, :, half_dim:], dim=2) for i in range(0, B, 2)],
#             dim=0)
#         # import ipdb
#         # ipdb.set_trace()
#
#         # latents = torch.sum(weight[:, :, :, None] * latents[:, None, :, self.dim//2:], dim=2)  # B x N x 128
#         preds_tex = self.to_rgb(tex_out, latents).squeeze(2)
#
#         return preds, preds_tex, sigma, sigma_t

# def feature_pe(x, avals, bvals):
#
#     avals = avals.to(x.device)
#     bvals = bvals.to(x.device)
#     avals = avals[None].repeat(x.shape[0], 1, 1)
#     # import ipdb
#     # ipdb.set_trace()
#     # , avals[::2].transpose(0, 1)
#     # , avals[1::2].transpose(0, 1)
#     sp = torch.cat([
#         avals * torch.sin((2. * torch.pi * x) @ bvals)[:, None],
#         avals * torch.cos((2. * torch.pi * x) @ bvals)[:, None]
#     ], dim=-1)
#
#     return sp

def feature_pe(x, bvals):

    bvals = bvals.to(x.device)
    # import ipdb
    # ipdb.set_trace()
    # , avals[::2].transpose(0, 1)
    # , avals[1::2].transpose(0, 1)
    sp = torch.cat([
        torch.sin((2. * torch.pi * x[:, None]) * bvals.T)[:, None],
        torch.cos((2. * torch.pi * x[:, None]) * bvals.T)[:, None]
    ], dim=-1)

    return sp

class ToRGB_layer_feape(nn.Module):
    def __init__(self,  query_channel=3, latent_channel=192, coord_pe=False):
        super(ToRGB_layer_feape, self).__init__()
        self.embedding_dim = 16
        self.latent_channel = latent_channel
        # _avals = torch.ones(latent_channel, self.embedding_dim//2)
        _bvals = torch.randn(latent_channel, self.embedding_dim//2)
        # self.register_buffer('avals', _avals)
        self.register_buffer('bvals', _bvals)
        self.coord_pe = coord_pe
        if self.coord_pe:
            self.embedding_size = 24
            e = torch.pow(2, torch.arange(self.embedding_size // 6)).float() * np.pi
            e = torch.stack([
                torch.cat([e, torch.zeros(self.embedding_size // 6),
                           torch.zeros(self.embedding_size // 6)]),
                torch.cat([torch.zeros(self.embedding_size // 6), e,
                           torch.zeros(self.embedding_size // 6)]),
                torch.cat([torch.zeros(self.embedding_size // 6),
                           torch.zeros(self.embedding_size // 6), e]),
            ])
            self.register_buffer('basis', e)  # 3 x 16
            self.l1 = weight_norm(
                nn.Linear(query_channel + self.embedding_size + latent_channel + latent_channel * self.embedding_dim,
                          512))
        else:
            self.l1 = weight_norm(
                nn.Linear(latent_channel + latent_channel * self.embedding_dim,
                          512))
        # self.l1 = weight_norm(nn.Linear(latent_channel + latent_channel * self.embedding_dim , 512))
        self.l2 = weight_norm(nn.Linear(512, 512))
        self.l3 = weight_norm(nn.Linear(512, 512))
        self.l4 = weight_norm(nn.Linear(512, 512))
        self.l_out = weight_norm(nn.Linear(512, 3))
    def forward(self, x, z):
        # pe = embed(x, self.basis)
        B, N, _ = z.shape

        fea = z.reshape(-1, self.latent_channel)

        # pe = feature_pe(fea,  self.avals, self.bvals)
        pe = feature_pe(fea, self.bvals)
        # ipdb.set_trace()
        input = pe.reshape(B, N, -1)
        if self.coord_pe:
            pe_x = embed(x, self.basis)
            input = torch.cat([x, pe_x, z, input], dim=2)
        else:
            input = torch.cat([z, input], dim=2)

        h = F.relu(self.l1(input))
        h = F.relu(self.l2(h))
        h = F.relu(self.l3(h))
        h = F.relu(self.l4(h))
        h = self.l_out(h)
        return h



class Toocc_layer_feape(nn.Module):
    def __init__(self,  query_channel=3, latent_channel=192, coord_pe=False):
        super(Toocc_layer_feape, self).__init__()
        self.embedding_dim = 16
        self.latent_channel = latent_channel
        # _avals = torch.ones(latent_channel, self.embedding_dim//2)
        _bvals = torch.randn(latent_channel, self.embedding_dim//2)
        # self.register_buffer('avals', _avals)
        self.register_buffer('bvals', _bvals)
        self.coord_pe = coord_pe
        if self.coord_pe:
            self.embedding_size = 24
            e = torch.pow(2, torch.arange(self.embedding_size // 6)).float() * np.pi
            e = torch.stack([
                torch.cat([e, torch.zeros(self.embedding_size // 6),
                           torch.zeros(self.embedding_size // 6)]),
                torch.cat([torch.zeros(self.embedding_size // 6), e,
                           torch.zeros(self.embedding_size // 6)]),
                torch.cat([torch.zeros(self.embedding_size // 6),
                           torch.zeros(self.embedding_size // 6), e]),
            ])
            self.register_buffer('basis', e)  # 3 x 16
            self.l1 = weight_norm(
                nn.Linear(query_channel + self.embedding_size + latent_channel + latent_channel * self.embedding_dim,
                          512))
        else:
            self.l1 = weight_norm(
                nn.Linear(latent_channel + latent_channel * self.embedding_dim,
                          512))
        # self.l1 = weight_norm(nn.Linear(latent_channel + latent_channel * self.embedding_dim , 512))
        self.l2 = weight_norm(nn.Linear(512, 512))
        self.l3 = weight_norm(nn.Linear(512, 512))
        self.l4 = weight_norm(nn.Linear(512, 512))
        self.l_out = weight_norm(nn.Linear(512, 1))
    def forward(self, x, z):
        # pe = embed(x, self.basis)
        B, N, _ = z.shape

        fea = z.reshape(-1, self.latent_channel)

        # pe = feature_pe(fea,  self.avals, self.bvals)
        pe = feature_pe(fea, self.bvals)
        # ipdb.set_trace()
        input = pe.reshape(B, N, -1)
        if self.coord_pe:
            pe_x = embed(x, self.basis)
            input = torch.cat([x, pe_x, z, input], dim=2)
        else:
            input = torch.cat([z, input], dim=2)

        h = F.relu(self.l1(input))
        h = F.relu(self.l2(h))
        h = F.relu(self.l3(h))
        h = F.relu(self.l4(h))
        h = self.l_out(h)
        return h


class Decoder_shape(nn.Module):
    def __init__(self, latent_channel=192):
        super().__init__()

        self.fc = Embedding(latent_channel=latent_channel)
        self.log_sigma = nn.Parameter(torch.FloatTensor([3.0]))
        # self.register_buffer('log_sigma', torch.Tensor([-3.0]))

        self.embed = Seq(Lin(48 + 3, latent_channel))  # , nn.GELU(), Lin(128, 128))

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

    def forward(self, latents, centers, samples):
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

        latents = torch.sum(weight[:, :, :, None] * latents[:, None, :, :], dim=2)  # B x N x 128
        preds = self.fc(samples, latents).squeeze(2)

        return preds, sigma

class Decoder_texture(torch.nn.Module):

    def __init__(self, latent_dim=32, plane_type=['xy', 'xz', 'yz'], padding=0.1, **kwargs):
        super().__init__()
        self.unet = UNet(latent_dim, in_channels=latent_dim)
        # self.to_rgb = ToRGB_layer(query_channel=3, latent_channel=latent_dim)
        self.to_rgb = ToRGB_layer_feape(query_channel=3, latent_channel=latent_dim, coord_pe = kwargs['coord_pe'])
        self.plane_type = plane_type
        self.padding = padding

    def forward(self, query_points, feature):

        feature = feature.permute(0, 3, 1, 2)
        feature = self.unet(feature)

        feat = {}

        feat['xz'], feat['xy'], feat['yz'] = torch.chunk(feature, 3, dim=2)
        plane_feat = 0

        plane_feat += self.sample_plane_feature(query_points, feat['xz'], 'xz')
        plane_feat += self.sample_plane_feature(query_points, feat['xy'], 'xy')
        plane_feat += self.sample_plane_feature(query_points, feat['yz'], 'yz')

        plane_feat = plane_feat.transpose(1, 2)

        rgb = self.to_rgb(query_points, plane_feat)

        return rgb


    def normalize_coordinate(self, p, padding=0.1, plane='xz'):
        ''' Normalize coordinate to [0, 1] for unit cube experiments
                    Args:
                        p (tensor): point
                        padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
                        plane (str): plane feature type, ['xz', 'xy', 'yz']
                    '''
        if plane == 'xz':
            xy = p[:, :, [0, 2]]
        elif plane == 'xy':
            xy = p[:, :, [0, 1]]
        else:
            xy = p[:, :, [1, 2]]

        xy_new = xy / (1 + padding + 10e-6)  # (-0.5, 0.5)
        xy_new = xy_new + 0.5  # range (0, 1)

        # f there are outliers out of the range
        if xy_new.max() >= 1:
            xy_new[xy_new >= 1] = 1 - 10e-6
        if xy_new.min() < 0:
            xy_new[xy_new < 0] = 0.0
        return xy_new

    # sample_plane_feature function copied from /src/conv_onet/models/decoder.py
    # uses values from plane_feature and pixel locations from vgrid to interpolate feature
    def sample_plane_feature(self, query, plane_feature, plane):
        xy = self.normalize_coordinate(query.clone(), plane=plane, padding=self.padding)
        xy = xy[:, :, None].float()
        vgrid = 2.0 * xy - 1.0  # normalize to (-1, 1)
        sampled_feat = F.grid_sample(plane_feature, vgrid, padding_mode='border', align_corners=True,
                                     mode='bilinear').squeeze(-1)
        return sampled_feat







class PointConv(torch.nn.Module):
    def __init__(self, local_nn=None, global_nn=None):
        super(PointConv, self).__init__()
        self.local_nn = local_nn
        self.global_nn = global_nn

    def forward(self, pos, pos_dst, edge_index, basis=None):
        row, col = edge_index

        out = pos[row] - pos_dst[col]
        # ipdb.set_trace()

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


class texture_block(nn.Module):
    def __init__(self, size_in, size_out = None, size_h=None):
        super().__init__()
        if size_out is None:
            size_out = size_in

        if size_h is None:
            size_h = min(size_out, size_in)

        self.size_in = size_in
        self.size_out = size_out
        self.size_h = size_h
        self.fc_0 = nn.Linear(size_in, size_h)
        self.fc_1 = nn.Linear(size_h, size_out)
        self.actvn = nn.ReLU()
        self.shortcut = nn.Linear(size_in, size_out)

        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x):
        net = self.fc_0(self.actvn(x))
        dx = self.fc_1(self.actvn(net))
        residual = self.shortcut(x)
        return dx + residual





# This is a mixture encoder with 3DILG-like
# class Encoder(nn.Module):
#     def __init__(self, N, dim=128, M=2048):
#         super().__init__()
#
#         self.embed = Seq(Lin(48+3, dim*2))#, nn.GELU(), Lin(128, 128))
#
#         self.embedding_dim = 48
#         e = torch.pow(2, torch.arange(self.embedding_dim // 6)).float() * np.pi
#         e = torch.stack([
#             torch.cat([e, torch.zeros(self.embedding_dim // 6),
#                       torch.zeros(self.embedding_dim // 6)]),
#             torch.cat([torch.zeros(self.embedding_dim // 6), e,
#                       torch.zeros(self.embedding_dim // 6)]),
#             torch.cat([torch.zeros(self.embedding_dim // 6),
#                       torch.zeros(self.embedding_dim // 6), e]),
#         ])
#         self.register_buffer('basis', e)  # 3 x 16
#
#         # self.conv = PointConv(local_nn=Seq(weight_norm(Lin(3+self.embedding_dim, dim))))
#         self.conv = PointConv(
#             local_nn=Seq(weight_norm(Lin(3+self.embedding_dim, 256)), ReLU(True), weight_norm(Lin(256, 256)) ),
#             global_nn=Seq(weight_norm(Lin(256, 256)), ReLU(True), weight_norm(Lin(256, dim)) ),
#         )
#         self.affine_shape = 48
#
#         self.affine_layer = Seq(Lin(3, self.affine_shape))
#         self.conv_t = PointConv_texture(
#             local_nn=Seq(weight_norm(Lin(3 + self.embedding_dim + self.affine_shape, 256)), ReLU(True), weight_norm(Lin(256, 256))),
#             global_nn=Seq(weight_norm(Lin(256, 256)), ReLU(True), weight_norm(Lin(256, dim))),
#         )
#
#         self.transformer = VisionTransformer(embed_dim=dim * 2,
#                                             depth=6,
#                                             num_heads=6,
#                                             mlp_ratio=4.,
#                                             qkv_bias=True,
#                                             qk_scale=None,
#                                             drop_rate=0.,
#                                             attn_drop_rate=0.,
#                                             drop_path_rate=0.1,
#                                             norm_layer=partial(nn.LayerNorm, eps=1e-6),
#                                             init_values=0.,
#                                             )
#
#
#         self.M = M
#         self.ratio = N / M
#         self.k = 32
#
#         self.k_t = 64
#
#     def forward(self, pc, tex_in):
#         # pc: B x N x D
#         B, N, D = pc.shape
#         assert N == self.M # 2048 points
#
#
#         #x is the shape feature.
#         #now we need to know the texture feature.
#         B, N_t, T = tex_in.shape
#
#         flattened_tex = tex_in.view(B*N_t, T)
#         pos = flattened_tex[..., :3]
#         batch_t = torch.arange(B).to(tex_in.device)
#         batch_t = torch.repeat_interleave(batch_t, N_t)
#         idx = fps(pos, batch_t, ratio=512/10000)
#
#         row, col = knn(pos, pos[idx], self.k_t, batch_t, batch_t[idx])
#         edge_index = torch.stack([col, row], dim=0)
#         x = self.conv(pos, pos[idx], edge_index, self.basis)
#
#         pos_t = torch.cat((pos, self.affine_layer(flattened_tex[..., 3:])), dim=-1)
#         tex_x = self.conv_t(pos_t, pos_t[idx], edge_index, self.basis)
#
#         # ipdb.set_trace()
#         x = torch.cat((x, tex_x), dim=-1)
#         pos, batch = pos[idx], batch_t[idx]
#
#         x = x.view(B, -1, x.shape[-1])
#         pos = pos.view(B, -1, 3)
#
#         embeddings = embed(pos, self.basis)
#
#         embeddings = self.embed(torch.cat([pos, embeddings], dim=2))
#
#         # import ipdb
#         # ipdb.set_trace()
#
#         out = self.transformer(x, embeddings)
#
#         return out, pos

# This is an auto-encode for mixture of shape and texture with 3dilg-like.
class Autoencoder(nn.Module):
    def __init__(self, N, K=512, dim=256, M=2048, **kwargs):
        super().__init__()

        self.encoder = Encoder(N=N, dim=dim//2, M=M)
        # Shape feature and texture feature should be dim//2
        self.dim = dim
        self.decoder = Decoder(latent_channel=dim)

        self.codebook_shape = VectorQuantizer2(K, dim//2)
        self.codebook_texture = VectorQuantizer2(K, dim//2)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {}

    def encode(self, x, tex_in, bins=256):

        B, _, _ = x.shape

        z_e_x, centers = self.encoder(x, tex_in) # B x T x C, B x T x 3

        centers_quantized = ((centers + 1) / 2 * (bins-1)).long()

        z_q_x_st, loss_vq, perplexity, encodings = self.codebook_shape(z_e_x[..., :self.dim//2])
        z_q_x_st_tex, loss_vq_tex, perplexity_tex, encodings_tex = self.codebook_texture(z_e_x[..., self.dim // 2:])
        # print(z_q_x_st.shape, loss_vq.item(), perplexity.item(), encodings.shape)
        return z_e_x, torch.cat((z_q_x_st, z_q_x_st_tex), dim=-1), centers_quantized, loss_vq+loss_vq_tex, [perplexity, perplexity_tex], encodings, encodings_tex

    def forward(self, x, points, tex_in, tex_out):

        z_e_x, z_q_x_st, centers_quantized, loss_vq, perplexity, encodings, encodings_tex = self.encode(x, tex_in)

        centers = centers_quantized.float() / 255.0 * 2 - 1

        z_q_x = z_q_x_st

        z_q_x_st = z_q_x_st
        B, N, C = z_q_x_st.shape

        logits, logits_tex, sigma, sigma_t = self.decoder(z_q_x_st, centers, points, tex_out)

        return logits, logits_tex, z_e_x, z_q_x, sigma, sigma_t, loss_vq, perplexity


# This is for shape only encoder.
class Encoder_shape(nn.Module):
    def __init__(self, N, dim=128, M=2048):
        super().__init__()

        self.embed = Seq(Lin(48 + 3, dim))  # , nn.GELU(), Lin(128, 128))

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

        self.transformer = VisionTransformer(embed_dim=dim,
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
        assert N == self.M # 2048 points


        #x is the shape feature.
        #now we need to know the texture feature.
        B, N_t, T = tex_in.shape

        flattened_tex = tex_in.view(B*N_t, T)
        pos = flattened_tex[..., :3]
        batch_t = torch.arange(B).to(tex_in.device)
        batch_t = torch.repeat_interleave(batch_t, N_t)
        # idx = fps(pos, batch_t, ratio=512/20000)
        idx = fps(pos, batch_t, ratio=512 / 2048)

        # import ipdb
        # ipdb.set_trace()


        row, col = knn(pos, pos[idx], self.k, batch_t, batch_t[idx])
        edge_index = torch.stack([col, row], dim=0)
        x = self.conv(pos, pos[idx], edge_index, self.basis)

        # ipdb.set_trace()
        pos, batch = pos[idx], batch_t[idx]
        x = x.view(B, -1, x.shape[-1])
        pos = pos.view(B, -1, 3)


        embeddings = embed(pos, self.basis)



        embeddings = self.embed(torch.cat([pos, embeddings], dim=2))


        out = self.transformer(x, embeddings)

        return out, pos

class Encoder_texture(nn.Module):
    def __init__(self, N, dim=3+48, M=2048,  plane_resolution=128, plane_type=['xz', 'xy', 'yz'], padding=0.1, **kwargs):
        super().__init__()

        self.c_dim = kwargs['c_dim']
        self.hidden_dim = kwargs['hidden_dim']
        self.fc_pos = nn.Linear(dim,  2*kwargs['hidden_dim'])
        self.blocks = nn.ModuleList(
            [
                texture_block(2*self.hidden_dim, self.hidden_dim) for i in range(kwargs['n_blocks'])
            ]
        )
        self.fc_c = nn.Linear(self.hidden_dim, self.c_dim)
        kwargs_unet = {}

        self.aware = kwargs['aware']
        if self.aware:
            self.unet = UNet(self.c_dim, in_channels=self.c_dim*3, **kwargs_unet)
        else:
            self.unet = UNet(self.c_dim, in_channels=self.c_dim, **kwargs_unet)
        self.reso_plane = plane_resolution
        self.plane_type = plane_type
        self.padding = padding
        self.reso_min = kwargs['plane_min']

        self.scatter = scatter_max
        self.affine_layer = nn.Linear(3, kwargs['affine_size'])

    def forward(self, tex_in):
        B, T, D = tex_in.shape

        points = tex_in[..., :3]
        texture = tex_in[..., 3:]

        coord = {}
        index = {}

        if 'xz' in self.plane_type:
            coord['xz'] = self.normalize_coordinate(points.clone(), plane='xz', padding=self.padding)
            index['xz'] = self.coordinate2index(coord['xz'], self.reso_plane)
        if 'xy' in self.plane_type:
            coord['xy'] = self.normalize_coordinate(points.clone(), plane='xy', padding=self.padding)
            index['xy'] = self.coordinate2index(coord['xy'], self.reso_plane)
        if 'yz' in self.plane_type:
            coord['yz'] = self.normalize_coordinate(points.clone(), plane='yz', padding=self.padding)
            index['yz'] = self.coordinate2index(coord['yz'], self.reso_plane)

        texture = torch.cat((points, self.affine_layer(texture)), dim=-1)

        net = self.fc_pos(texture)

        net = self.blocks[0](net)
        for block in self.blocks[1:]:
            pooled = self.pool_local(coord, index, net)
            net = torch.cat([net, pooled], dim=2)
            net = block(net)

        c = self.fc_c(net)

        fea = {}
        plane_fea_sum = 0

        if self.reso_min:
            reso = torch.randint(self.reso_min, self.reso_plane+1, (1,)).item()
        else:
            reso = self.reso_plane

        if 'xz' in self.plane_type:
            fea['xz'] = self.generate_plane_features(points, c,
                                                     plane='xz', reso=reso)  # shape: batch, latent size, resolution, resolution (e.g. 16, 256, 64, 64)
            # plane_feat_sum += self.sample_plane_feature(query, fea['xz'], 'xz')
            # second_sum += self.sample_plane_feature(query2, fea['xz'], 'xz')
        if 'xy' in self.plane_type:
            fea['xy'] = self.generate_plane_features(points, c, plane='xy', reso=reso)
            # plane_feat_sum += self.sample_plane_feature(query, fea['xy'], 'xy')
            # second_sum += self.sample_plane_feature(query2, fea['xy'], 'xy')
        if 'yz' in self.plane_type:
            fea['yz'] = self.generate_plane_features(points, c, plane='yz', reso=reso)
            # plane_feat_sum += self.sample_plane_feature(query, fea['yz'], 'yz')

        if self.aware:
            feat_x = self.T3d_conv(fea['xz'], fea['xy'], fea['yz'])
            feat_y = self.T3d_conv(fea['xy'], fea['xz'], fea['yz'].transpose(2, 3))
            feat_z = self.T3d_conv(fea['yz'], fea['xy'].transpose(2, 3), fea['xz'])
            fea_plane = torch.cat((feat_x, feat_y, feat_z), dim=2)
        else:
            fea_plane = torch.cat((fea['xz'], fea['xy'], fea['yz']), dim=2)

        # fea_plane = torch.cat((fea['xz'], fea['xy'], fea['yz']), dim=2)
        if self.unet is not None:
            fea_plane = self.unet(fea_plane)
        # How should we add the 1e-5 kl in this
        return fea_plane

    def T3d_conv(self, x, y, z):
        _, _, H, W = x.shape
        y = torch.mean(y, dim=3, keepdim=True)
        y = y.repeat(1, 1, 1, W)
        z = torch.mean(z, dim=2, keepdim=True)
        z = z.repeat(1, 1, H, 1)
        return torch.cat((x, y, z), dim=1)


    def normalize_coordinate(self, p, padding=0.1, plane='xz'):
        ''' Normalize coordinate to [0, 1] for unit cube experiments
                    Args:
                        p (tensor): point
                        padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
                        plane (str): plane feature type, ['xz', 'xy', 'yz']
                    '''
        if plane == 'xz':
            xy = p[:, :, [0, 2]]
        elif plane == 'xy':
            xy = p[:, :, [0, 1]]
        else:
            xy = p[:, :, [1, 2]]

        xy_new = xy / (1 + padding + 10e-6)  # (-0.5, 0.5)
        xy_new = xy_new + 0.5  # range (0, 1)

        # f there are outliers out of the range
        if xy_new.max() >= 1:
            xy_new[xy_new >= 1] = 1 - 10e-6
        if xy_new.min() < 0:
            xy_new[xy_new < 0] = 0.0
        return xy_new

    def coordinate2index(self, x, reso):
        ''' Normalize coordinate to [0, 1] for unit cube experiments.
                        Corresponds to our 3D model
                    Args:
                        x (tensor): coordinate
                        reso (int): defined resolution
                        coord_type (str): coordinate type
                    '''
        x = (x * reso).long()
        index = x[:, :, 0] + reso * x[:, :, 1]
        index = index[:, None, :]
        return index

    # xy is the normalized coordinates of the point cloud of each plane
    # I'm pretty sure the keys of xy are the same as those of index, so xy isn't needed here as input
    def generate_plane_features(self, p, c, plane='xz', reso=128):
        # acquire indices of features in plane
        xy = self.normalize_coordinate(p.clone(), plane=plane,
                                       padding=self.padding)  # normalize to the range of (0, 1)


        index = self.coordinate2index(xy, reso)

        # scatter plane features from points
        fea_plane = c.new_zeros(p.size(0), self.c_dim, reso** 2)
        c = c.permute(0, 2, 1)  # B x 512 x T
        fea_plane = scatter_mean(c, index, out=fea_plane)  # B x 512 x reso^2
        fea_plane = fea_plane.reshape(p.size(0), self.c_dim, reso,
                                      reso)  # sparce matrix (B x 512 x reso x reso)

        # process the plane features with UNet
        if not reso == self.reso_plane:
            fea_plane = F.interpolate(fea_plane, (self.reso_plane, self.reso_plane), mode="nearest")
        return fea_plane


    def pool_local(self, xy, index, c):
        bs, fea_dim = c.size(0), c.size(2)
        keys = xy.keys()

        c_out = 0
        for key in keys:
            # scatter plane features from points
            fea = self.scatter(c.permute(0, 2, 1), index[key], dim_size=self.reso_plane ** 2)
            if self.scatter == scatter_max:
                fea = fea[0]
            # gather feature back to points
            fea = fea.gather(dim=2, index=index[key].expand(-1, fea_dim, -1))
            c_out += fea
        return c_out.permute(0, 2, 1) #






class Autoencoder_seperate(nn.Module):
    def __init__(self, N, K=512, dim=256, M=2048,  **kwargs):
        super().__init__()

        self.encoder_shape = Encoder_shape(N=N, dim=dim, M=M)
        # This should be revisited.
        dim_tex = kwargs['hidden_dim']
        self.encoder_texture = Encoder_texture(N, 3 + kwargs['affine_size'], plane_resolution=kwargs['reso'], **kwargs)
        # Shape feature and texture feature should be dim//2
        self.dim = dim
        self.pre_fc = nn.Linear(dim_tex, dim_tex * 2)
        self.post_fc = nn.Linear(dim_tex, dim_tex)
        self.decoder_shape = Decoder_shape(latent_channel=dim)
        self.decoder_texuture = Decoder_texture(latent_dim=kwargs['hidden_dim'],  **kwargs)


        self.codebook_shape = VectorQuantizer2(K, dim)
        # self.codebook_texture = VectorQuantizer2(K, dim//2)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {}

    def encode_shape(self, x, tex_in, bins=256):

        B, _, _ = x.shape

        z_e_x, centers = self.encoder_shape(x, tex_in) # B x T x C, B x T x 3

        centers_quantized = ((centers + 1) / 2 * (bins-1)).long()

        z_q_x_st, loss_vq, perplexity, encodings = self.codebook_shape(z_e_x)
        # print(z_q_x_st.shape, loss_vq.item(), perplexity.item(), encodings.shape)
        return z_e_x, z_q_x_st, centers_quantized, loss_vq, perplexity, encodings
        # return z_e_x, torch.cat((z_q_x_st, z_q_x_st_tex), dim=-1), centers_quantized, loss_vq+loss_vq_tex, [perplexity, perplexity_tex], encodings, encodings_tex

    def encode_texture(self, x, tex_in, bins=256):
        B, _, _ = x.shape

        feat = self.encoder_texture(tex_in)

        z_encode = feat.permute(0, 2, 3, 1)
        z_encode = self.pre_fc(z_encode)
        mu, logvar = torch.chunk(z_encode, 2, dim=3)
        std = torch.exp(0.5 * logvar)

        z_sample = mu + std *torch.randn(mu.shape).to(device=x.device)

        #Fixme: This should be revisited for different shape
        kld = 0
        var = torch.exp(logvar)
        kld = 0.5 * torch.sum(
            torch.pow(mu, 2) + var - 1.0 - logvar, dim=[1, 2, 3]
        )
        # import ipdb
        # ipdb.set_trace()
        kld = torch.sum(kld)/kld.shape[0]

        z_sample = self.post_fc(z_sample)

        return z_sample, kld



        # for k, v in feat.items():
        #     z_encode = self.pre_fc(v)
        #     mu, logvar = torch.chunk(z_encode)



    def forward(self, x, points, tex_in, tex_out):

        z_e_x, z_q_x_st, centers_quantized, loss_vq, perplexity, encodings = self.encode_shape(x, tex_in)
        z_sample, kld = self.encode_texture(x, tex_in)

        centers = centers_quantized.float() / 255.0 * 2 - 1

        z_q_x = z_q_x_st

        z_q_x_st = z_q_x_st
        B, N, C = z_q_x_st.shape

        logits, sigma = self.decoder_shape(z_q_x_st, centers, points)

        # logits, logits_tex, sigma, sigma_t = self.decoder(z_q_x_st, centers, points, tex_out)
        rgb = self.decoder_texuture(tex_out[..., :3], z_sample)

        return logits, rgb,  z_e_x, z_q_x, sigma, loss_vq, 1e-5 * kld, perplexity


        # return logits, logits_tex, z_e_x, z_q_x, sigma, sigma_t, loss_vq, perplexity


class Encoder_unet(nn.Module):
    def __init__(self, N, dim=3+48, M=2048,  plane_resolution=128, plane_type=['xz', 'xy', 'yz'], padding=0.1, **kwargs):
        super().__init__()

        self.c_dim = kwargs['c_dim']
        self.hidden_dim = kwargs['hidden_dim']
        self.fc_pos = nn.Linear(dim,  2*kwargs['hidden_dim'])
        self.blocks = nn.ModuleList(
            [
                texture_block(2*self.hidden_dim, self.hidden_dim) for i in range(kwargs['n_blocks'])
            ]
        )
        self.fc_c = nn.Linear(self.hidden_dim, self.c_dim)
        kwargs_unet = {}

        self.aware = kwargs['aware']
        if self.aware:
            self.unet = UNet(self.c_dim, in_channels=self.c_dim*3, **kwargs_unet)
        else:
            self.unet = UNet(self.c_dim, in_channels=self.c_dim, **kwargs_unet)
        self.reso_plane = plane_resolution
        self.plane_type = plane_type
        self.padding = padding
        self.reso_min = kwargs['plane_min']

        self.scatter = scatter_max
        self.affine_layer = nn.Linear(3, kwargs['affine_size'])

    def forward(self, tex_in):
        B, T, D = tex_in.shape

        points = tex_in[..., :3]
        texture = tex_in[..., 3:]

        coord = {}
        index = {}

        if 'xz' in self.plane_type:
            coord['xz'] = self.normalize_coordinate(points.clone(), plane='xz', padding=self.padding)
            index['xz'] = self.coordinate2index(coord['xz'], self.reso_plane)
        if 'xy' in self.plane_type:
            coord['xy'] = self.normalize_coordinate(points.clone(), plane='xy', padding=self.padding)
            index['xy'] = self.coordinate2index(coord['xy'], self.reso_plane)
        if 'yz' in self.plane_type:
            coord['yz'] = self.normalize_coordinate(points.clone(), plane='yz', padding=self.padding)
            index['yz'] = self.coordinate2index(coord['yz'], self.reso_plane)

        texture = torch.cat((points, self.affine_layer(texture)), dim=-1)

        net = self.fc_pos(texture)

        net = self.blocks[0](net)
        for block in self.blocks[1:]:
            pooled = self.pool_local(coord, index, net)
            net = torch.cat([net, pooled], dim=2)
            net = block(net)

        c = self.fc_c(net)

        fea = {}
        plane_fea_sum = 0

        if self.reso_min:
            reso = torch.randint(self.reso_min, self.reso_plane+1, (1,)).item()
        else:
            reso = self.reso_plane

        if 'xz' in self.plane_type:
            fea['xz'] = self.generate_plane_features(points, c,
                                                     plane='xz', reso=reso)  # shape: batch, latent size, resolution, resolution (e.g. 16, 256, 64, 64)
            # plane_feat_sum += self.sample_plane_feature(query, fea['xz'], 'xz')
            # second_sum += self.sample_plane_feature(query2, fea['xz'], 'xz')
        if 'xy' in self.plane_type:
            fea['xy'] = self.generate_plane_features(points, c, plane='xy', reso=reso)
            # plane_feat_sum += self.sample_plane_feature(query, fea['xy'], 'xy')
            # second_sum += self.sample_plane_feature(query2, fea['xy'], 'xy')
        if 'yz' in self.plane_type:
            fea['yz'] = self.generate_plane_features(points, c, plane='yz', reso=reso)
            # plane_feat_sum += self.sample_plane_feature(query, fea['yz'], 'yz')

        if self.aware:
            feat_x = self.T3d_conv(fea['xz'], fea['xy'], fea['yz'])
            feat_y = self.T3d_conv(fea['xy'], fea['xz'], fea['yz'].transpose(2, 3))
            feat_z = self.T3d_conv(fea['yz'], fea['xy'].transpose(2, 3), fea['xz'])
            fea_plane = torch.cat((feat_x, feat_y, feat_z), dim=2)
        else:
            fea_plane = torch.cat((fea['xz'], fea['xy'], fea['yz']), dim=2)

        # fea_plane = torch.cat((fea['xz'], fea['xy'], fea['yz']), dim=2)
        if self.unet is not None:
            fea_plane = self.unet(fea_plane)
        # How should we add the 1e-5 kl in this
        return fea_plane



    def T3d_conv(self, x, y, z):
        _, _, H, W = x.shape
        y = torch.mean(y, dim=3, keepdim=True)
        y = y.repeat(1, 1, 1, W)
        z = torch.mean(z, dim=2, keepdim=True)
        z = z.repeat(1, 1, H, 1)
        return torch.cat((x, y, z), dim=1)


    def normalize_coordinate(self, p, padding=0.1, plane='xz'):
        ''' Normalize coordinate to [0, 1] for unit cube experiments
                    Args:
                        p (tensor): point
                        padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
                        plane (str): plane feature type, ['xz', 'xy', 'yz']
                    '''
        if plane == 'xz':
            xy = p[:, :, [0, 2]]
        elif plane == 'xy':
            xy = p[:, :, [0, 1]]
        else:
            xy = p[:, :, [1, 2]]

        xy_new = xy / (1 + padding + 10e-6)  # (-0.5, 0.5)
        xy_new = xy_new + 0.5  # range (0, 1)

        # f there are outliers out of the range
        if xy_new.max() >= 1:
            xy_new[xy_new >= 1] = 1 - 10e-6
        if xy_new.min() < 0:
            xy_new[xy_new < 0] = 0.0
        return xy_new

    def coordinate2index(self, x, reso):
        ''' Normalize coordinate to [0, 1] for unit cube experiments.
                        Corresponds to our 3D model
                    Args:
                        x (tensor): coordinate
                        reso (int): defined resolution
                        coord_type (str): coordinate type
                    '''
        x = (x * reso).long()
        index = x[:, :, 0] + reso * x[:, :, 1]
        index = index[:, None, :]
        return index

    # xy is the normalized coordinates of the point cloud of each plane
    # I'm pretty sure the keys of xy are the same as those of index, so xy isn't needed here as input
    def generate_plane_features(self, p, c, plane='xz', reso=128):
        # acquire indices of features in plane
        xy = self.normalize_coordinate(p.clone(), plane=plane,
                                       padding=self.padding)  # normalize to the range of (0, 1)


        index = self.coordinate2index(xy, reso)

        # scatter plane features from points
        fea_plane = c.new_zeros(p.size(0), self.c_dim, reso** 2)
        c = c.permute(0, 2, 1)  # B x 512 x T
        fea_plane = scatter_mean(c, index, out=fea_plane)  # B x 512 x reso^2
        fea_plane = fea_plane.reshape(p.size(0), self.c_dim, reso,
                                      reso)  # sparce matrix (B x 512 x reso x reso)

        # process the plane features with UNet
        if not reso == self.reso_plane:
            fea_plane = F.interpolate(fea_plane, (self.reso_plane, self.reso_plane), mode="nearest")
        return fea_plane


    def pool_local(self, xy, index, c):
        bs, fea_dim = c.size(0), c.size(2)
        keys = xy.keys()

        c_out = 0
        for key in keys:
            # scatter plane features from points
            fea = self.scatter(c.permute(0, 2, 1), index[key], dim_size=self.reso_plane ** 2)
            if self.scatter == scatter_max:
                fea = fea[0]
            # gather feature back to points
            fea = fea.gather(dim=2, index=index[key].expand(-1, fea_dim, -1))
            c_out += fea
        return c_out.permute(0, 2, 1) #


class Decoder_unet(nn.Module):
    def __init__(self, latent_dim=32, plane_type=['xy', 'xz', 'yz'], padding=0.1, **kwargs):
        super().__init__()
        self.unet = UNet(latent_dim, in_channels=latent_dim)
        # self.to_rgb = ToRGB_layer(query_channel=3, latent_channel=latent_dim)
        self.to_occ = Toocc_layer_feape(query_channel=3, latent_channel=latent_dim, coord_pe=kwargs['coord_pe'])
        self.to_rgb = ToRGB_layer_feape(query_channel=3, latent_channel=latent_dim, coord_pe=kwargs['coord_pe'])
        self.plane_type = plane_type
        self.padding = padding

    def forward(self, query_points, feature, query_surf):

        feature = feature.permute(0, 3, 1, 2)
        feature = self.unet(feature)

        feat = {}

        feat['xz'], feat['xy'], feat['yz'] = torch.chunk(feature, 3, dim=2)
        plane_feat = 0

        plane_feat += self.sample_plane_feature(query_points, feat['xz'], 'xz')
        plane_feat += self.sample_plane_feature(query_points, feat['xy'], 'xy')
        plane_feat += self.sample_plane_feature(query_points, feat['yz'], 'yz')

        plane_feat = plane_feat.transpose(1, 2)
        rgb = self.to_rgb(query_points, plane_feat)

        plane_feat = 0
        plane_feat += self.sample_plane_feature(query_surf, feat['xz'], 'xz')
        plane_feat += self.sample_plane_feature(query_surf, feat['xy'], 'xy')
        plane_feat += self.sample_plane_feature(query_surf, feat['yz'], 'yz')

        plane_feat = plane_feat.transpose(1, 2)

        occ = self.to_occ(query_surf, plane_feat)
        return rgb, occ.squeeze(2)

    def normalize_coordinate(self, p, padding=0.1, plane='xz'):
        ''' Normalize coordinate to [0, 1] for unit cube experiments
                    Args:
                        p (tensor): point
                        padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
                        plane (str): plane feature type, ['xz', 'xy', 'yz']
                    '''
        if plane == 'xz':
            xy = p[:, :, [0, 2]]
        elif plane == 'xy':
            xy = p[:, :, [0, 1]]
        else:
            xy = p[:, :, [1, 2]]

        xy_new = xy / (1 + padding + 10e-6)  # (-0.5, 0.5)
        xy_new = xy_new + 0.5  # range (0, 1)

        # f there are outliers out of the range
        if xy_new.max() >= 1:
            xy_new[xy_new >= 1] = 1 - 10e-6
        if xy_new.min() < 0:
            xy_new[xy_new < 0] = 0.0
        return xy_new

    # sample_plane_feature function copied from /src/conv_onet/models/decoder.py
    # uses values from plane_feature and pixel locations from vgrid to interpolate feature
    def sample_plane_feature(self, query, plane_feature, plane):
        xy = self.normalize_coordinate(query.clone(), plane=plane, padding=self.padding)
        xy = xy[:, :, None].float()
        vgrid = 2.0 * xy - 1.0  # normalize to (-1, 1)
        sampled_feat = F.grid_sample(plane_feature, vgrid, padding_mode='border', align_corners=True,
                                     mode='bilinear').squeeze(-1)
        return sampled_feat


class Autoencoder_mixunet(nn.Module):
    def __init__(self, N, K=512, dim=256, M=2048, **kwargs):
        super().__init__()

        dim_tex = kwargs['hidden_dim']
        self.encoder = Encoder_unet(N, 3 + kwargs['affine_size'], plane_resolution=kwargs['reso'], **kwargs)
        # Shape feature and texture feature should be dim//2
        self.dim = dim
        self.decoder = Decoder_unet(latent_dim=kwargs['hidden_dim'],  **kwargs)
        # Shape feature and texture feature should be dim//2
        self.dim = dim
        self.pre_fc = nn.Linear(dim_tex, dim_tex * 2)
        self.post_fc = nn.Linear(dim_tex, dim_tex)
        # self.codebook_shape = VectorQuantizer2(K, dim // 2)
        # self.codebook_texture = VectorQuantizer2(K, dim // 2)

    def encode(self, x, tex_in, bins=256):
        B, _, _ = x.shape

        feat = self.encoder(tex_in)

        z_encode = feat.permute(0, 2, 3, 1)
        z_encode = self.pre_fc(z_encode)
        mu, logvar = torch.chunk(z_encode, 2, dim=3)
        std = torch.exp(0.5 * logvar)

        z_sample = mu + std *torch.randn(mu.shape).to(device=x.device)

        #Fixme: This should be revisited for different shape
        kld = 0
        var = torch.exp(logvar)
        kld = 0.5 * torch.sum(
            torch.pow(mu, 2) + var - 1.0 - logvar, dim=[1, 2, 3]
        )
        # import ipdb
        # ipdb.set_trace()
        kld = torch.sum(kld)/kld.shape[0]

        z_sample = self.post_fc(z_sample)

        return z_sample, kld
    def forward(self, x, points, tex_in, tex_out):
        z_sample, kld = self.encode(x, tex_in)

        rgb, occ = self.decoder(tex_out[..., :3], z_sample, points)
        # ipdb.set_trace()

        return occ, rgb, 0, 0, 0, 0, 1e-5*kld, 0

    @torch.jit.ignore
    def no_weight_decay(self):
        return {}



class Decoder_cross(torch.nn.Module):

    def __init__(self, latent_dim=32, plane_resolution=128, plane_type=['xy', 'xz', 'yz'], padding=0.1, **kwargs):
        super().__init__()
        self.unet = UNet(latent_dim, in_channels=latent_dim)
        # self.to_rgb = ToRGB_layer(query_channel=3, latent_channel=latent_dim)
        self.to_rgb = ToRGB_layer_feape(query_channel=3, latent_channel=latent_dim, coord_pe = kwargs['coord_pe'])
        self.plane_type = plane_type
        self.padding = padding
        self.reso_plane = plane_resolution
        self.down1 = nn.Linear(256, 256)
        self.down2 = nn.Linear(
            256, latent_dim
        )
        self.c_dim = kwargs['c_dim']

    def forward(self, query_points, feature, centers, latents):

        feature = feature.permute(0, 3, 1, 2)
        # This should be revisted.

        # ipdb.set_trace()
        feature_shape = {}
        latents = self.down2(F.relu(self.down1(latents)))

        if 'xz' in self.plane_type:
            feature_shape['xz'] = self.generate_plane_features(centers, latents, plane='xz', reso=self.reso_plane)
        if 'xy' in self.plane_type:
            feature_shape['xy'] = self.generate_plane_features(centers, latents, plane='xy', reso=self.reso_plane)
        if 'yz' in self.plane_type:
            feature_shape['yz'] = self.generate_plane_features(centers, latents, plane='yz', reso=self.reso_plane)


        feature_shape = torch.cat((feature_shape['xz'], feature_shape['xy'], feature_shape['yz']), dim=2)



        feature = feature + feature_shape

        feature = self.unet(feature)

        feat = {}

        feat['xz'], feat['xy'], feat['yz'] = torch.chunk(feature, 3, dim=2)
        plane_feat = 0

        plane_feat += self.sample_plane_feature(query_points, feat['xz'], 'xz')
        plane_feat += self.sample_plane_feature(query_points, feat['xy'], 'xy')
        plane_feat += self.sample_plane_feature(query_points, feat['yz'], 'yz')

        plane_feat = plane_feat.transpose(1, 2)

        rgb = self.to_rgb(query_points, plane_feat)

        return rgb

    def normalize_coordinate(self, p, padding=0.1, plane='xz'):
        ''' Normalize coordinate to [0, 1] for unit cube experiments
                    Args:
                        p (tensor): point
                        padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
                        plane (str): plane feature type, ['xz', 'xy', 'yz']
                    '''
        if plane == 'xz':
            xy = p[:, :, [0, 2]]
        elif plane == 'xy':
            xy = p[:, :, [0, 1]]
        else:
            xy = p[:, :, [1, 2]]

        xy_new = xy / (1 + padding + 10e-6)  # (-0.5, 0.5)
        xy_new = xy_new + 0.5  # range (0, 1)

        # f there are outliers out of the range
        if xy_new.max() >= 1:
            xy_new[xy_new >= 1] = 1 - 10e-6
        if xy_new.min() < 0:
            xy_new[xy_new < 0] = 0.0
        return xy_new

    def coordinate2index(self, x, reso):
        ''' Normalize coordinate to [0, 1] for unit cube experiments.
                        Corresponds to our 3D model
                    Args:
                        x (tensor): coordinate
                        reso (int): defined resolution
                        coord_type (str): coordinate type
                    '''
        x = (x * reso).long()
        index = x[:, :, 0] + reso * x[:, :, 1]
        index = index[:, None, :]
        return index

        # xy is the normalized coordinates of the point cloud of each plane
        # I'm pretty sure the keys of xy are the same as those of index, so xy isn't needed here as input

    def generate_plane_features(self, p, c, plane='xz', reso=128):
        # acquire indices of features in plane
        xy = self.normalize_coordinate(p.clone(), plane=plane,
                                       padding=self.padding)  # normalize to the range of (0, 1)

        index = self.coordinate2index(xy, reso)

        # scatter plane features from points
        fea_plane = c.new_zeros(p.size(0), self.c_dim, reso ** 2)
        c = c.permute(0, 2, 1)  # B x 512 x T
        fea_plane = scatter_mean(c, index, out=fea_plane)  # B x 512 x reso^2
        fea_plane = fea_plane.reshape(p.size(0), self.c_dim, reso,
                                      reso)  # sparce matrix (B x 512 x reso x reso)

        # process the plane features with UNet
        if not reso == self.reso_plane:
            fea_plane = F.interpolate(fea_plane, (self.reso_plane, self.reso_plane), mode="nearest")
        return fea_plane

    def pool_local(self, xy, index, c):
        bs, fea_dim = c.size(0), c.size(2)
        keys = xy.keys()

        c_out = 0
        for key in keys:
            # scatter plane features from points
            fea = self.scatter(c.permute(0, 2, 1), index[key], dim_size=self.reso_plane ** 2)
            if self.scatter == scatter_max:
                fea = fea[0]
            # gather feature back to points
            fea = fea.gather(dim=2, index=index[key].expand(-1, fea_dim, -1))
            c_out += fea
        return c_out.permute(0, 2, 1)  #

    def normalize_coordinate(self, p, padding=0.1, plane='xz'):
        ''' Normalize coordinate to [0, 1] for unit cube experiments
                    Args:
                        p (tensor): point
                        padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
                        plane (str): plane feature type, ['xz', 'xy', 'yz']
                    '''
        if plane == 'xz':
            xy = p[:, :, [0, 2]]
        elif plane == 'xy':
            xy = p[:, :, [0, 1]]
        else:
            xy = p[:, :, [1, 2]]

        xy_new = xy / (1 + padding + 10e-6)  # (-0.5, 0.5)
        xy_new = xy_new + 0.5  # range (0, 1)

        # f there are outliers out of the range
        if xy_new.max() >= 1:
            xy_new[xy_new >= 1] = 1 - 10e-6
        if xy_new.min() < 0:
            xy_new[xy_new < 0] = 0.0
        return xy_new

    # sample_plane_feature function copied from /src/conv_onet/models/decoder.py
    # uses values from plane_feature and pixel locations from vgrid to interpolate feature
    def sample_plane_feature(self, query, plane_feature, plane):
        xy = self.normalize_coordinate(query.clone(), plane=plane, padding=self.padding)
        xy = xy[:, :, None].float()
        vgrid = 2.0 * xy - 1.0  # normalize to (-1, 1)
        sampled_feat = F.grid_sample(plane_feature, vgrid, padding_mode='border', align_corners=True,
                                     mode='bilinear').squeeze(-1)
        return sampled_feat




#This is only used for 3DILG, Irregular field cross section decoder
class Autoencoder_cross(nn.Module):
    def __init__(self, N, K=512, dim=256, M=2048,  **kwargs):
        super().__init__()

        self.encoder_shape = Encoder_shape(N=N, dim=dim, M=M)
        # This should be revisited.
        dim_tex = kwargs['hidden_dim']
        self.encoder_texture = Encoder_texture(N, 3 + kwargs['affine_size'], plane_resolution=kwargs['reso'], **kwargs)
        # Shape feature and texture feature should be dim//2
        self.dim = dim
        self.pre_fc = nn.Linear(dim_tex, dim_tex * 2)
        self.post_fc = nn.Linear(dim_tex, dim_tex)
        self.decoder_shape = Decoder_shape(latent_channel=dim)
        self.decoder_texuture = Decoder_cross(latent_dim=kwargs['hidden_dim'],  **kwargs)


        self.codebook_shape = VectorQuantizer2(K, dim)
        # self.codebook_texture = VectorQuantizer2(K, dim//2)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {}

    def encode_shape(self, x, tex_in, bins=256):

        B, _, _ = x.shape

        z_e_x, centers = self.encoder_shape(x, tex_in) # B x T x C, B x T x 3

        centers_quantized = ((centers + 1) / 2 * (bins-1)).long()

        z_q_x_st, loss_vq, perplexity, encodings = self.codebook_shape(z_e_x)
        # print(z_q_x_st.shape, loss_vq.item(), perplexity.item(), encodings.shape)
        return z_e_x, z_q_x_st, centers_quantized, loss_vq, perplexity, encodings
        # return z_e_x, torch.cat((z_q_x_st, z_q_x_st_tex), dim=-1), centers_quantized, loss_vq+loss_vq_tex, [perplexity, perplexity_tex], encodings, encodings_tex

    def encode_texture(self, x, tex_in, bins=256):
        B, _, _ = x.shape

        feat = self.encoder_texture(tex_in)

        z_encode = feat.permute(0, 2, 3, 1)
        z_encode = self.pre_fc(z_encode)
        mu, logvar = torch.chunk(z_encode, 2, dim=3)
        std = torch.exp(0.5 * logvar)

        z_sample = mu + std *torch.randn(mu.shape).to(device=x.device)

        #Fixme: This should be revisited for different shape
        kld = 0
        var = torch.exp(logvar)
        kld = 0.5 * torch.sum(
            torch.pow(mu, 2) + var - 1.0 - logvar, dim=[1, 2, 3]
        )
        # import ipdb
        # ipdb.set_trace()
        kld = torch.sum(kld)/kld.shape[0]

        z_sample = self.post_fc(z_sample)

        return z_sample, kld



        # for k, v in feat.items():
        #     z_encode = self.pre_fc(v)
        #     mu, logvar = torch.chunk(z_encode)



    def forward(self, x, points, tex_in, tex_out):

        z_e_x, z_q_x_st, centers_quantized, loss_vq, perplexity, encodings = self.encode_shape(x, tex_in)
        z_sample, kld = self.encode_texture(x, tex_in)

        centers = centers_quantized.float() / 255.0 * 2 - 1

        z_q_x = z_q_x_st

        z_q_x_st = z_q_x_st
        B, N, C = z_q_x_st.shape

        logits, sigma = self.decoder_shape(z_q_x_st, centers, points)

        # logits, logits_tex, sigma, sigma_t = self.decoder(z_q_x_st, centers, points, tex_out)
        # These are used for cross-section information fusion.
        rgb = self.decoder_texuture(tex_out[..., :3], z_sample, centers, z_q_x_st)

        return logits, rgb,  z_e_x, z_q_x, sigma, loss_vq, 1e-5 * kld, perplexity



class AutoencoderKL(nn.Module):
    def __init__(self, N, dim=256, M=2048, latent_dim=256, Num_Cls=1, **kwargs):
        super().__init__()

        print(f'Autoencoder with KL-Divergence')

        # Comentts when opening pre/post operation
        # self.encoder = Encoder(N=N, dim=2*dim, M=M)
        self.encoder = Encoder_shape(N=N, dim=dim, M=M)



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
    model = Autoencoder_seperate(
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
    model = Autoencoder_seperate(
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
def vqvae_512_1024_2048_pureunet(pretrained=False, **kwargs):
    # import ipdb
    # ipdb.set_trace()
    model = Autoencoder_mixunet(
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
def vqvae_512_1024_2048_cross(pretrained=False, **kwargs):
    # import ipdb
    # ipdb.set_trace()
    model = Autoencoder_cross(
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