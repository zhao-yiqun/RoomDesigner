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
from torch_scatter import scatter_max

from einops import rearrange
import math

from uni_rep.camera.perspective_camera import PerspectiveCamera
from uni_rep.render.neural_render import NeuralRender

def _cfg(url='', **kwargs):
    return {
    }


def sdf_reg_loss_batch(sdf, all_edges):
    sdf_f1x6x2 = sdf[:, all_edges.reshape(-1)].reshape(sdf.shape[0], -1, 2)
    mask = torch.sign(sdf_f1x6x2[..., 0]) != torch.sign(sdf_f1x6x2[..., 1])
    sdf_f1x6x2 = sdf_f1x6x2[mask]
    sdf_diff = torch.nn.functional.binary_cross_entropy_with_logits(
        sdf_f1x6x2[..., 0], (sdf_f1x6x2[..., 1] > 0).float()) + \
               torch.nn.functional.binary_cross_entropy_with_logits(
                   sdf_f1x6x2[..., 1], (sdf_f1x6x2[..., 0] > 0).float())
    return sdf_diff


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



from dmtet import DMTetGeometry
from utils import create_my_world2cam_matrix, normalize_vecs
import ipdb
# dmtet = DMTetGeometry()


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

    def forward(self, x, z): # samples, and its corresponding lattents
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


class Implicitmlp(nn.Module):
    def __init__(self, query_channel=3, latent_channel=192, output_channel=3):
        super(Implicitmlp, self).__init__()
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
        self.l_out = weight_norm(nn.Linear(512, output_channel))
        # self.l4 = weight_norm(nn.Linear(512, 512 - query_channel - latent_channel - self.embedding_dim))
        # self.l5 = weight_norm(nn.Linear(512,. 512))
        # self.l6 = weight_norm(nn.Linear(512, 512))
        # self.l7 = weight_norm(nn.Linear(512, 512))
        # self.l_out = weight_norm(nn.Linear(512, output_channel))

    def forward(self, x, z):  # samples, and its corresponding lattents
        # x: B x N x 3
        # z: B x N x 192
        # input = torch.cat([x[:, :, None].expand(-1, -1, z.shape[1], -1), z[:, None].expand(-1, x.shape[1], -1, -1)], dim=-1)
        # print(x.shape, z.shape)

        pe = embed(x, self.basis)

        input = torch.cat([x, pe, z], dim=2)

        h = F.relu(self.l1(input))
        h = F.relu(self.l2(h))
        h = F.relu(self.l3(h))
        h = self.l_out(h)
        # h = F.relu(self.l4(h))
        # h = torch.cat((h, input), axis=2)
        # h = F.relu(self.l5(h))
        # h = F.relu(self.l6(h))
        # h = F.relu(self.l7(h))
        # h = self.l_out(h)
        return h




class Decoder(nn.Module):
    def __init__(self, latent_channel=192, grid_res = 90, deformation_multiplier = 2):
        super().__init__()

        self.sdf_fn = Implicitmlp(latent_channel=latent_channel, output_channel=1)
        self.def_fn = Implicitmlp(latent_channel=latent_channel, output_channel=3)
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
        self.grid_res = grid_res
        self.dmtet_scale = 1.9
        self.render_type= 'neural_render'
        self.deformation_multiplier = deformation_multiplier
        self.n_views = 8
        self.img_resolution = 1024
        self.device = 'cuda'


        # fovy = np.arctan(32 / 2 / 35) * 2
        # fovyangle = fovy / np.pi * 180.0
        #Note !!
        fovy = 0.6911112070083618
        fovyangle = fovy / np.pi * 180.0
        dmtet_camera = PerspectiveCamera(fovy=fovyangle, device=self.device)

        # Renderer we used.
        dmtet_renderer = NeuralRender(device=self.device, camera_model=dmtet_camera)
        self.dmtet = DMTetGeometry(self.grid_res, scale=self.dmtet_scale, renderer=dmtet_renderer, render_type=self.render_type, device=self.device)


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

    def get_geometry_prediction(self, latents, samples):
        # latents = torch.sum(weight[:, :, :, None] * latents[:, None, :, :], dim=2)  # B x N x 128
        # latents = new_latents
        # del new_latents
        # import ipdb
        # ipdb.set_trace()
        samples = samples.expand(latents.shape[0], -1, -1)

        pred_sdf = self.sdf_fn(samples, latents).squeeze(2)
        # pred_sdf, pred_def = pred_sdf[..., :1], pred_sdf[..., 1:]
        pred_def = self.def_fn(samples, latents).squeeze(2)

        # import ipdb
        # ipdb.set_trace()

        # Step 2: Normalize the deformation to avoid the flipped triangles.
        deformation = 1.0 / (self.grid_res * self.deformation_multiplier) * torch.tanh(pred_def)
        sdf_reg_loss = torch.zeros(pred_sdf.shape[0], device=pred_sdf.device, dtype=torch.float32)

        # Step 3: Fix some sdf if we observe empty shape (full positive or full negative)

        pos_shape = torch.sum((pred_sdf.squeeze(dim=-1) > 0).int(), dim=-1)
        neg_shape = torch.sum((pred_sdf.squeeze(dim=-1) < 0).int(), dim=-1)
        zero_surface = torch.bitwise_or(pos_shape == 0, neg_shape == 0)
        if torch.sum(zero_surface).item() > 0:
            update_sdf = torch.zeros_like(pred_sdf[0:1])
            max_sdf = pred_sdf.max()
            min_sdf = pred_sdf.min()
            update_sdf[:, self.dmtet.center_indices] += (1.0 - min_sdf)  # greater than zero
            update_sdf[:, self.dmtet.boundary_indices] += (-1 - max_sdf)  # smaller than zero
            new_sdf = torch.zeros_like(pred_sdf)
            for i_batch in range(zero_surface.shape[0]):
                if zero_surface[i_batch]:
                    new_sdf[i_batch:i_batch + 1] += update_sdf
            update_mask = (new_sdf == 0).float()
            # Regulraization here is used to push the sdf to be a different sign (make it not fully positive or fully negative)
            # sdf_reg_loss = torch.abs(pred_sdf).mean(dim=-1).mean(dim=-1)
            # sdf_reg_loss = sdf_reg_loss * zero_surface.float()
            pred_sdf = pred_sdf * update_mask + new_sdf * (1 - update_mask)

        # ipdb.set_trace()
        # Step 4: Here we remove the gradient for the bad sdf (full positive or full negative)

        final_sdf = []
        final_def = []
        all_detach = False
        if zero_surface.sum() == zero_surface.shape[0]:
            all_detach = True
        for i_batch in range(zero_surface.shape[0]):
            if zero_surface[i_batch]:
                final_sdf.append(pred_sdf[i_batch: i_batch + 1].detach())
                final_def.append(deformation[i_batch: i_batch + 1].detach())
            else:
                final_sdf.append(pred_sdf[i_batch: i_batch + 1])
                final_def.append(deformation[i_batch: i_batch + 1])
        sdf = torch.cat(final_sdf, dim=0)
        deformation = torch.cat(final_def, dim=0)
        # import ipdb
        # ipdb.set_trace()

        # ipdb.set_trace()

        v_deformed = self.dmtet.verts.unsqueeze(dim=0).expand(sdf.shape[0], -1, -1) + deformation

        tets = self.dmtet.indices
        n_batch = latents.shape[0]
        v_list = []
        f_list = []

        # Step 2: Using marching tet to obtain the mesh
        for i_batch in range(n_batch):
            verts, faces = self.dmtet.get_mesh(
                v_deformed[i_batch], sdf[i_batch].squeeze(dim=-1),
                with_uv=False, indices=tets)
            v_list.append(verts)
            f_list.append(faces)


        return v_list, f_list, sdf, all_detach

    def render_mesh(self, mesh_v, mesh_f, cam_mv):
        return_value_list = []
        # ipdb.set_trace()
        for i_mesh in range(len(mesh_v)):
            return_value = self.dmtet.render_mesh(
                mesh_v[i_mesh],
                mesh_f[i_mesh].int(),
                cam_mv[i_mesh],
                resolution=self.img_resolution,
                hierarchical_mask= False
            )
            return_value_list.append(return_value)
        return_keys = return_value_list[0].keys()
        return_value = dict()
        for k in return_keys:
            value = [v[k] for v in return_value_list]
            return_value[k] = value

        mask_list, hard_mask_list = torch.stack(return_value['mask'], dim=0), torch.stack(return_value['hard_mask'], dim=0)
        return mask_list, hard_mask_list, return_value


    def generate_camera(self, camera, radians, n_views):
        B = camera.shape[0]
        n = B * n_views
        theta = camera[:, 0].reshape(-1)
        phi = camera[:, 1].reshape(-1)

        compute_theta = -theta - 0.5 * math.pi
        output_points = torch.zeros((n, 3)).to(camera.device)
        output_points[:, 0] = radians * torch.sin(phi) * torch.cos(compute_theta)
        output_points[:, 2] = radians * torch.sin(phi) * torch.sin(compute_theta)
        output_points[:, 1] = radians * torch.cos(phi)
        rotation_angles = theta
        elevation_angles = phi
        forward_vector = normalize_vecs(output_points)

        world2cam_matrix = create_my_world2cam_matrix(forward_vector, output_points, camera.device)
        return world2cam_matrix, forward_vector, output_points, rotation_angles, elevation_angles



    def forward(self, latents, centers, cameras=None):
        # kernel average
        # samples: B x N x 3
        # latents: B x T x 320
        # centers: B x T x 3
        embeddings = embed(centers, self.basis)
        embeddings = self.embed(torch.cat([centers, embeddings], dim=2))
        latents = self.transformer(latents, embeddings)

        samples = self.dmtet.verts.unsqueeze(dim=0)
        # ipdb.set_trace()
        pdist = (samples[:,  :, None] - centers[:, None]).square().sum(dim=-1) # B x N x T
        sigma = torch.exp(self.log_sigma)
        # sort_dis, sort_id = pdist.sort(dim=-1)
        # weight = F.softmax(-sort_dis[..., :32]*sigma, dim=2)
        # #for the nearest 32 points, to compute the weight
        #
        # # weight = F.softmax(-pdist * sigma, dim=2)
        # # sort_weight, sort_id = weight.sort(descending=True, dim=-1)
        #
        #
        #
        # B, N = weight.shape[:2]
        # # next = []
        #
        # latents = torch.cat(
        #     [torch.sum(weight[i:i+1, ..., None]* latents[i, sort_id[i, :, :32].long()], dim=2) for i in range(B)]
        # )
        # for i in range(B):
        #     next.append(
        #         torch.sum(weight[i:i+1, ..., None]* latents[i, sort_id[i, :, :32].long()], dim=2)
        #     )
        # latents = torch.cat(next, dim=0)

        weight = F.softmax(-pdist * sigma, dim=2)
        B, N = weight.shape[:2]

        # ipdb.set_trace()
        # latents = torch.sum(weight[:, :, :, None] * latents[:, None, :, :], dim=2)  # B x N x 128
        latents = torch.cat(
            [torch.sum(weight[i:i + 1, :, :, None] * latents[i:i + 1, None, :, :], dim=2) for i in range(B)],
            dim=0)
        # P_num = weight.shape[1]
        # latents_pre = torch.cat(
        #     [torch.sum(weight[i:i + 1, :P_num//2, :, None] * latents[i:i + 1, None, :, :], dim=2) for i in range(B)], dim=0)
        # latents_aft = torch.cat(
        #     [torch.sum(weight[i:i + 1, P_num//2:, :, None] * latents[i:i + 1, None, :, :], dim=2) for i in range(B)], dim=0)
        # latents = torch.cat((latents_pre, latents_aft), dim=1)
        v_list, f_list, pred_sdf, all_detach = self.get_geometry_prediction(latents, samples)



        # points = torch.stack(p_list, dim=0)
        # run_n_view = self.n_views
        # with torch.no_grad():
        #     radian =  3.88
        #     world2cam_matrix, forward_vector, camera_origin, rotation_angle, elevation_angle = \
        #         self.generate_camera(cameras, radian, run_n_view)
        #     mv_batch = world2cam_matrix
        #     campos = camera_origin
        #     campos = campos.reshape(B, run_n_view, 3)
        #     mv_batch = mv_batch.reshape(B, run_n_view, 4, 4)
        #     gen_camera = (campos, mv_batch, radian, rotation_angle, elevation_angle)
        mv_batch = cameras
        # ipdb.set_trace()
        # import ipdb
        # ipdb.set_trace()
        antilias_mask, hard_mask, return_value = self.render_mesh(v_list, f_list, mv_batch.type_as(samples))# rendered results.
        # return_value also have a key 'tex_pos' which is the texture position
        # Antialias_mask is the antialiasing mask generating from the hard mask.
        # in get3d, author use antilias_mask as alpha map
        # hard mask are used to save the memory
        # How to split mask and depth map
        depth = torch.stack(return_value['depth'], dim=0)
        # ipdb.set_trace()


        return v_list, f_list, pred_sdf, sigma, depth, hard_mask, antilias_mask, all_detach


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


class Encoder(nn.Module):
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

    def forward(self, pc):
        # pc: B x N x D
        B, N, D = pc.shape
        assert N == self.M  # 2048 points

        flattened = pc.view(B * N, D)
        # point cloud for flatten.

        batch = torch.arange(B).to(pc.device)
        batch = torch.repeat_interleave(batch, N)

        pos = flattened

        idx = fps(pos, batch, ratio=self.ratio)  # 0.0625

        row, col = knn(pos, pos[idx], self.k, batch, batch[idx])
        edge_index = torch.stack([col, row], dim=0)

        x = self.conv(pos, pos[idx], edge_index, self.basis)
        pos, batch = pos[idx], batch[idx]

        x = x.view(B, -1, x.shape[-1])
        pos = pos.view(B, -1, 3)

        embeddings = embed(pos, self.basis)

        embeddings = self.embed(torch.cat([pos, embeddings], dim=2))

        out = self.transformer(x, embeddings)

        return out, pos


class Autoencoder(nn.Module):
    def __init__(self, N, K=512, dim=256, M=2048, latent_dim= 256, **kwargs):
        super().__init__()

        self.encoder = Encoder(N=N, dim=dim, M=M)

        self.pre_fc = nn.Linear(dim, latent_dim)
        self.post_fc = nn.Linear(latent_dim, dim)

        self.decoder = Decoder(latent_channel=dim, grid_res=kwargs['grid_res'])

        self.freeze = kwargs['freeze']
        #FixMe: This should be noted when revise different experiments.

        self.codebook = VectorQuantizer2(K, latent_dim)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {}

    def encode(self, x, bins=256):
        B, _, _ = x.shape

        z_e_x, centers = self.encoder(x)  # B x T x C, B x T x 3

        centers_quantized = ((centers + 1) / 2 * (bins - 1)).long()

        z_e_x = self.pre_fc(z_e_x)

        z_q_x_st, loss_vq, perplexity, encodings = self.codebook(z_e_x)

        z_q_x_st = self.post_fc(z_q_x_st)
        # print(z_q_x_st.shape, loss_vq.item(), perplexity.item(), encodings.shape)
        return z_e_x, z_q_x_st, centers_quantized, loss_vq, perplexity, encodings

    def forward(self, x, condinfo):
        if self.freeze:
            with torch.no_grad():
                z_e_x, z_q_x_st, centers_quantized, loss_vq, perplexity, encodings = self.encode(x)

                centers = centers_quantized.float() / 255.0 * 2 - 1

                z_q_x = z_q_x_st

                z_q_x_st = z_q_x_st
        else:
            z_e_x, z_q_x_st, centers_quantized, loss_vq, perplexity, encodings = self.encode(x)

            centers = centers_quantized.float() / 255.0 * 2 - 1

            z_q_x = z_q_x_st

            z_q_x_st = z_q_x_st


        B, N, C = z_q_x_st.shape
        # ipdb.set_trace()
        # import ipdb
        # ipdb.set_trace()

        v_list, f_list, pred_sdf, sigma, depth, hard_mask, antilias_mask, all_detach = self.decoder(z_q_x_st, centers, condinfo)

        sdf_reg_loss_entropy = sdf_reg_loss_batch(pred_sdf, self.decoder.dmtet.all_edges).mean() * 0.01

        # points, v_list, f_list, sdf_reg_loss, sigma = self.decoder(z_q_x_st, centers)

        return sdf_reg_loss_entropy, v_list, f_list, z_e_x, z_q_x, sigma, loss_vq, perplexity, depth, antilias_mask, all_detach


class AutoencoderKL(nn.Module):
    def __init__(self, N, dim=256, M=2048, latent_dim=256, Num_Cls=1, **kwargs):
        super().__init__()

        print(f'Autoencoder with KL-Divergence')

        # Comentts when opening pre/post operation
        # self.encoder = Encoder(N=N, dim=2*dim, M=M)
        self.encoder = Encoder(N=N, dim=dim, M=M)

        self.pre_fc = nn.Linear(dim, latent_dim * 2)
        self.post_fc = nn.Linear(latent_dim, dim)

        self.freeze = kwargs['freeze']


        self.decoder = Decoder(latent_channel=dim, grid_res=kwargs['grid_res'])

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

    def forward(self, x, condinfo):
        if self.freeze:
            with torch.no_grad():
                _, z, centers, kld, _, _ = self.encode(x, return_kl=True)
        else:
            _, z, centers, kld, _, _ = self.encode(x, return_kl=True)
        v_list, f_list, pred_sdf, sigma, depth, hard_mask, antilias_mask, all_detach  = self.decoder(z, centers, condinfo)

        # import ipdb
        # ipdb.set_trace()

        sdf_reg_loss_entropy = sdf_reg_loss_batch(pred_sdf, self.decoder.dmtet.all_edges).mean() * 0.01

        # return sdf_reg_loss_entropy, v_list, f_list, 0, 0, sigma, 1e-5 * kld, 0, depth, antilias_mask, all_detach
        return sdf_reg_loss_entropy, v_list, f_list, 0, 0, sigma, 1e-7 * kld, 0, depth, antilias_mask, all_detach

        # points, v_list, f_list, sdf_reg_loss, sigma = self.decoder(z_q_x_st, centers)

        # return logits, 0, 0, sigma, 1e-5 * kld, 0


# @register_model
# def vqvae_512_1024_2048_dmtet(pretrained=False, **kwargs):
#     model = Autoencoder(
#         N=64,
#         K=1024,
#         M=2048,
#         **kwargs)
#     model.default_cfg = _cfg()
#     if pretrained:
#         checkpoint = torch.load(
#             kwargs["init_ckpt"], map_location="cpu"
#         )
#         model.load_state_dict(checkpoint["model"])
#     return model


@register_model
def vqvae_64_1024_2048_dmtet(pretrained=False, **kwargs):
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
def vqvae_128_1024_2048_dmtet(pretrained=False, **kwargs):
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
def vqvae_256_1024_2048_dmtet(pretrained=False, **kwargs):
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
def vqvae_512_512_2048_dmtet(pretrained=False, **kwargs):
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
def vqvae_512_1024_2048_dmtet_64(pretrained=False, **kwargs):
    # import ipdb
    # ipdb.set_trace()
    kwargs['grid_res'] = 64
    model = Autoencoder(
        N=512,
        K=1024,
        dim=32,
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
def vqvae_512_1024_2048_dmtet_90(pretrained=False, **kwargs):

    kwargs['grid_res'] = 90
    model = Autoencoder(
        N=512,
        K=1024,
        dim=32,
        M=2048,
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model


def vqvae_256_1024_2048_dmtet(pretrained=False, **kwargs):
    # import ipdb
    # ipdb.set_trace()
    model = Autoencoder(
        N=256,
        K=1024,
        dim=32,
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
def vqvae_1024_1024_4096_dmtet(pretrained=False, **kwargs):
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
def vaekl_512_256_2048_dmtet(pretrained=False, **kwargs):
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
def vaekl_256_128_2048_dmtet(pretrained=False, **kwargs):
    model = AutoencoderKL(
        N=256,
        dim=128,
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
def vaekl_512_64_2048_dmtet(pretrained=False, **kwargs):
    model = AutoencoderKL(
        N=512,
        dim=64,
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
def vaekl_256_32_2048_dmtet_64(pretrained=False, **kwargs):
    kwargs['grid_res'] = 64
    model = AutoencoderKL(
        N=256,
        dim=32,
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
def vaekl_512_32_2048_dmtet_90(pretrained=False, **kwargs):
    kwargs['grid_res'] = 90
    model = AutoencoderKL(
        N=512,
        dim=32,
        M=2048,
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model

# @register_model
# def vaekl_512_32_2048_dmtet(pretrained=False, **kwargs):
#     model = AutoencoderKL(
#         N=512,
#         dim=32,
#         M=2048,
#         **kwargs)
#     model.default_cfg = _cfg()
#     if pretrained:
#         checkpoint = torch.load(
#             kwargs["init_ckpt"], map_location="cpu"
#         )
#         model.load_state_dict(checkpoint["model"])
#     return model

def main():
    model = vqvae_512_1024_2048()
    surface = torch.randn(1, 2048, 3)
    points = torch.randn(1, 2048, 3)
    points, sdf_reg_loss, v_list, f_list, z_e_x, z_q_x, sigma, loss_commit, perplexity = model(surface, points)
    import ipdb
    ipdb.set_trace()




if __name__ == '__main__':
    main()

