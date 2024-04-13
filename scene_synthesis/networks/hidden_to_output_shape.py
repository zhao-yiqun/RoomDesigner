#
# Copyright (C) 2021 NVIDIA Corporation.  All rights reserved.
# Licensed under the NVIDIA Source Code License.
# See LICENSE at https://github.com/nv-tlabs/ATISS.
# Authors: Despoina Paschalidou, Amlan Kar, Maria Shugrina, Karsten Kreis,
#          Andreas Geiger, Sanja Fidler
#

import torch
import torch.nn as nn

from .bbox_output_shape import AutoregressiveBBoxOutputShape
from .base import FixedPositionalEncoding, sample_from_dmll
import torch.nn.functional as F
from .gpt import GPT
# from Shape_encoder import Shape_encoder_512_1024_24_K1024 as shape_encoder

class Hidden2OutputShape(nn.Module):
    def __init__(self, hidden_size, n_classes, with_extra_fc=False):
        super().__init__()
        self.with_extra_fc = with_extra_fc
        self.n_classes = n_classes
        self.hidden_size = hidden_size

        mlp_layers = [
            nn.Linear(hidden_size, 2*hidden_size),
            nn.ReLU(),
            nn.Linear(2*hidden_size, hidden_size),
            nn.ReLU()
        ]
        if self.with_extra_fc:
            self.hidden2output = nn.Sequential(*mlp_layers)

    def apply_linear_layers(self, x):
        if self.with_extra_fc:
            x = self.hidden2output(x)

        class_labels = self.class_layer(x)
        translations = (
            self.centroid_layer_x(x),
            self.centroid_layer_y(x),
            self.centroid_layer_z(x)
        )
        sizes = (
            self.size_layer_x(x),
            self.size_layer_y(x),
            self.size_layer_z(x)
        )
        angles = self.angle_layer(x)
        return class_labels, translations, sizes, angles

    def forward(self, x, sample_params=None):
        raise NotImplementedError()


class AutoregressiveDMLLShape(Hidden2OutputShape):
    def __init__(
        self,
        hidden_size,
        n_classes,
        n_mixtures,
        bbox_output,
        shape_embed,
        stage,
        with_extra_fc=False
    ):
        super().__init__(hidden_size, n_classes, with_extra_fc)
        self.shape_embed = shape_embed
        self.stage = stage
        if not isinstance(n_mixtures, list):
            n_mixtures = [n_mixtures]*7

        self.class_layer = nn.Linear(hidden_size, n_classes)

        self.fc_class_labels = nn.Linear(n_classes, 64)
        # Positional embedding for the target translation
        self.pe_trans_x = FixedPositionalEncoding(proj_dims=64)
        self.pe_trans_y = FixedPositionalEncoding(proj_dims=64)
        self.pe_trans_z = FixedPositionalEncoding(proj_dims=64)
        # Positional embedding for the target angle
        self.pe_angle_z = FixedPositionalEncoding(proj_dims=64)


        self.pe_size_x = FixedPositionalEncoding(proj_dims=64)
        self.pe_size_y = FixedPositionalEncoding(proj_dims=64)
        self.pe_size_z = FixedPositionalEncoding(proj_dims=64)

        self.pe_shape_x = FixedPositionalEncoding(proj_dims=64)
        self.pe_shape_y = FixedPositionalEncoding(proj_dims=64)
        self.pe_shape_z = FixedPositionalEncoding(proj_dims=64)

        # self.nxt_hidden_size = 512

        c_hidden_size = hidden_size + 64
        self.centroid_layer_x = AutoregressiveDMLLShape._mlp(
            c_hidden_size, n_mixtures[0]*3
        )
        self.centroid_layer_y = AutoregressiveDMLLShape._mlp(
            c_hidden_size, n_mixtures[1]*3
        )
        self.centroid_layer_z = AutoregressiveDMLLShape._mlp(
            c_hidden_size, n_mixtures[2]*3
        )
        c_hidden_size = c_hidden_size + 64*3
        self.angle_layer = AutoregressiveDMLLShape._mlp(
            c_hidden_size, n_mixtures[6]*3
        )
        c_hidden_size = c_hidden_size + 64
        self.size_layer_x = AutoregressiveDMLLShape._mlp(
            c_hidden_size, n_mixtures[3]*3
        )
        self.size_layer_y = AutoregressiveDMLLShape._mlp(
            c_hidden_size, n_mixtures[4]*3
        )
        self.size_layer_z = AutoregressiveDMLLShape._mlp(
            c_hidden_size, n_mixtures[5]*3
        )
        c_hidden_size = c_hidden_size + 64*3

        # if self.stage =="second":
            # Solution #4：Make the shape decoder using 3DILG
        hidden_channel = 256
        self.pos_emb = nn.Parameter(nn.Embedding(self.shape_embed, hidden_channel).weight[None])
        self.shape_encoder = nn.Linear(c_hidden_size, hidden_channel)

        self.pos_x = nn.Embedding(256, hidden_channel)
        self.pos_y = nn.Embedding(256, hidden_channel)
        self.pos_z = nn.Embedding(256, hidden_channel)
        # self.pos_shape = nn.Embedding(1024, hidden_channel)
        self.transformer = GPT(vocab_size=512, block_size=self.shape_embed, n_layer=12, n_head=4, n_embd=256,
                               embd_pdrop=0.1, resid_pdrop=0.1, attn_pdrop=0.1)

        self.shape_layer_x = AutoregressiveDMLLShape._mlp(
            hidden_channel, 30
        )
        self.shape_layer_y = AutoregressiveDMLLShape._mlp(
            hidden_channel, 30
        )
        self.shape_layer_z = AutoregressiveDMLLShape._mlp(
            hidden_channel, 30
        )
        self.shape_layer_latent = AutoregressiveDMLLShape._mlp(
            hidden_channel, 1024
        )
        self.ln_x = nn.LayerNorm(256)
        self.ln_y = nn.LayerNorm(256)
        self.ln_z = nn.LayerNorm(256)
        self.ln_latent = nn.LayerNorm(256)


        # Solution #5: DMLL loss for decoder

        # hidden_channel = 256
        # self.pos_emb = nn.Parameter(nn.Embedding(512, hidden_channel).weight[None])
        # self.shape_encoder = nn.Linear(c_hidden_size, hidden_channel)
        # self.shape_layer_x = AutoregressiveDMLLShape._mlp1(
        #     hidden_channel, 30
        # )
        # self.shape_layer_y = AutoregressiveDMLLShape._mlp1(
        #     hidden_channel, 30
        # )
        # self.shape_layer_z = AutoregressiveDMLLShape._mlp1(
        #     hidden_channel, 30
        # )
        # self.shape_layer_latent = AutoregressiveDMLLShape._mlp(
        #     hidden_channel, 1024
        # )


        # self.shape_encoder = shape_encoder()
        # ninp = 1024
        # coord_vocab_size = 256
        # latent_vocab_size = 1024
        # self.pos_emb = nn.Parameter(nn.Embedding(512, ninp).weight[None])
        # self.x_tok_emb = nn.Embedding(coord_vocab_size, ninp)
        # self.y_tok_emb = nn.Embedding(coord_vocab_size, ninp)
        # self.z_tok_emb = nn.Embedding(coord_vocab_size, ninp)
        # self.latent_tok_emb = nn.Embedding(latent_vocab_size, ninp)
        #
        # self.shape_encoder = nn.Linear(c_hidden_size, 1024)
        # self.shape_layer_x = AutoregressiveDMLLShape._mlp(
        #     latent_vocab_size, ninp
        # )
        # self.shape_layer_y = AutoregressiveDMLLShape._mlp(
        #     latent_vocab_size, ninp
        # )
        # self.shape_layer_z = AutoregressiveDMLLShape._mlp(
        #     latent_vocab_size, ninp
        # )
        # self.shape_layer_latent = AutoregressiveDMLLShape._mlp(
        #     latent_vocab_size, ninp
        # )
        # self.x_head = nn.Linear(ninp, coord_vocab_size, bias=False)


        # Solution #3: make the decoder auto-regressively
        # c_hidden_size = c_hidden_size + 64 * 3
        # self.shape_layer_x = AutoregressiveDMLLShape._mlp(
        #         c_hidden_size, 512
        # )
        # c_hidden_size = c_hidden_size + 64
        # self.shape_layer_y = AutoregressiveDMLLShape._mlp(
        #     c_hidden_size, 1
        # )
        # c_hidden_size = c_hidden_size + 64
        # self.shape_layer_z = AutoregressiveDMLLShape._mlp(
        #     c_hidden_size, 1
        # )
        # c_hidden_size = c_hidden_size + 64
        # self.shape_feature = AutoregressiveDMLLShape._mlp(
        #     c_hidden_size, 1024
        # )


        # Solution #2: previous are too heavy, I need to express 512, feature for the whole coords and featue predict.
        # But this version do not support the autoregressive on point-level

        # c_hidden_size = c_hidden_size + 64 * 3
        # self.shape_layer = AutoregressiveDMLLShape._mlp(
        #     c_hidden_size, 2048
        # )
        #
        # self.shape_layer_x = AutoregressiveDMLLShape._mlp(
        #     2048, 512
        # )
        # self.shape_layer_y = AutoregressiveDMLLShape._mlp(
        #     2048 , 512
        # )
        # self.shape_layer_z = AutoregressiveDMLLShape._mlp(
        #     2048 , 512
        # )
        # self.shape_feature = AutoregressiveDMLLShape._mlp(
        #     2048 + 3, 1024
        # )


        # c_hidden_size = c_hidden_size + 64*3 + 512*4
        # self.point_layer_x = AutoregressiveDMLLShape._mlp(
        #     c_hidden_size, 512
        # )
        #
        # c_hidden_size = c_hidden_size + 512
        #
        # self.point_layer_y = AutoregressiveDMLLShape._mlp(
        #     c_hidden_size, 512
        # )
        # c_hidden_size = c_hidden_size + 512
        #
        # self.point_layer_z = AutoregressiveDMLLShape._mlp(
        #     c_hidden_size, 512
        # )
        # c_hidden_size = c_hidden_size + 512
        # self.feature_layer = AutoregressiveDMLLShape._mlp(
        #     c_hidden_size, 512
        # )
        #
        # self.x_logits = nn.Sequential(*[
        #     nn.Linear(1, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, 256)
        #     ])
        # self.y_logits = nn.Sequential(*[
        #     nn.Linear(1, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, 256)
        #     ])
        # self.z_logits = nn.Sequential(*[
        #     nn.Linear(1, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, 256)
        #     ])
        # self.feature_logits = nn.Sequential(*[
        #     nn.Linear(1, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 1024)
        #     ])


        #Solution #1: Previous is a solution former, not a good choice currently, therefore, the next.

        # c_hidden_size = c_hidden_size + 64*3 + 512*4
        #
        # self.feature_aggregate = self._mlp(c_hidden_size, c_hidden_size * 2)
        #
        # self.point_layer_x = self._mlp(
        #     c_hidden_size // 512 *2, 256
        # )
        #
        # self.point_layer_y = self._mlp(
        #     c_hidden_size // 512 *2, 256
        # )
        #
        # self.point_layer_z =  self._mlp(
        #     c_hidden_size // 512 *2, 256
        # )
        #
        # self.feature_layer =  self._mlp(
        #     c_hidden_size // 512 *2, 1024
        # )


        self.bbox_output = bbox_output

    @staticmethod
    def _mlp1(hidden_size, output_size):
        mlp_layers = [
            nn.Linear(hidden_size, 2 * hidden_size),
            nn.ReLU(),
            nn.Linear(2 * hidden_size,  4 * hidden_size),
            nn.ReLU(),
            nn.Linear(4 * hidden_size, 4 * output_size),
            nn.ReLU(),
            # nn.Linear(2 * hidden_size, 4 * output_size),
            # nn.ReLU(),
            nn.Linear(4 * output_size , output_size)
        ]

        return nn.Sequential(*mlp_layers)
    @staticmethod
    def _mlp(hidden_size, output_size):
        mlp_layers = [
            nn.Linear(hidden_size, 2*hidden_size),
            nn.ReLU(),
            nn.Linear(2*hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        ]
        return nn.Sequential(*mlp_layers)


    @staticmethod
    def _extract_properties_from_target(sample_params):
        class_labels = sample_params["class_labels_tr"].float()
        translations = sample_params["translations_tr"].float()
        sizes = sample_params["sizes_tr"].float()
        angles = sample_params["angles_tr"].float()
        if "shapes_tr" in sample_params.keys():
            shapes = sample_params["shapes_tr"]
            return class_labels, translations, sizes, angles, shapes
        else:
            return class_labels, translations, sizes, angles

    @staticmethod
    def get_dmll_params(pred):
        assert len(pred.shape) == 2

        N = pred.size(0)
        nr_mix = pred.size(1) // 3

        probs = torch.softmax(pred[:, :nr_mix], dim=-1)
        means = pred[:, nr_mix:2 * nr_mix]
        scales = torch.nn.functional.elu(pred[:, 2*nr_mix:3*nr_mix]) + 1.0001

        return probs, means, scales

    def get_translations_dmll_params(self, x, class_labels):
        # Extract the sizes in local variables for convenience
        B, L, _ = class_labels.shape

        c = self.fc_class_labels(class_labels)
        cf = torch.cat([x, c], dim=-1)
        translations_x = self.centroid_layer_x(cf).reshape(B*L, -1)
        translations_y = self.centroid_layer_y(cf).reshape(B*L, -1)
        translations_z = self.centroid_layer_z(cf).reshape(B*L, -1)

        dmll_params = {}
        p = AutoregressiveDMLLShape.get_dmll_params(translations_x)
        dmll_params["translations_x_probs"] = p[0]
        dmll_params["translations_x_means"] = p[1]
        dmll_params["translations_x_scales"] = p[2]

        p = AutoregressiveDMLLShape.get_dmll_params(translations_y)
        dmll_params["translations_y_probs"] = p[0]
        dmll_params["translations_y_means"] = p[1]
        dmll_params["translations_y_scales"] = p[2]

        p = AutoregressiveDMLLShape.get_dmll_params(translations_z)
        dmll_params["translations_z_probs"] = p[0]
        dmll_params["translations_z_means"] = p[1]
        dmll_params["translations_z_scales"] = p[2]

        return dmll_params

    def sample_class_labels(self, x):
        class_labels = self.class_layer(x)

        # Extract the sizes in local variables for convenience
        B, L, _ = class_labels.shape
        C = self.n_classes

        # Sample the class
        class_probs = torch.softmax(class_labels, dim=-1).view(B*L, C)
        sampled_classes = torch.multinomial(class_probs, 1).view(B, L)
        return torch.eye(C, device=x.device)[sampled_classes]

    def sample_translations(self, x, class_labels):
        # Extract the sizes in local variables for convenience
        B, L, _ = class_labels.shape

        c = self.fc_class_labels(class_labels)
        cf = torch.cat([x, c], dim=-1)
        translations_x = self.centroid_layer_x(cf)
        translations_y = self.centroid_layer_y(cf)
        translations_z = self.centroid_layer_z(cf)

        t_x = sample_from_dmll(translations_x.reshape(B*L, -1))
        t_y = sample_from_dmll(translations_y.reshape(B*L, -1))
        t_z = sample_from_dmll(translations_z.reshape(B*L, -1))
        return torch.cat([t_x, t_y, t_z], dim=-1).view(B, L, 3)

    def sample_angles(self, x, class_labels, translations):
        # Extract the sizes in local variables for convenience
        B, L, _ = class_labels.shape

        c = self.fc_class_labels(class_labels)
        cf = torch.cat([x, c], dim=-1)
        tx = self.pe_trans_x(translations[:, :, 0:1])
        ty = self.pe_trans_y(translations[:, :, 1:2])
        tz = self.pe_trans_z(translations[:, :, 2:3])
        tf = torch.cat([cf, tx, ty, tz], dim=-1)
        angles = self.angle_layer(tf)
        return sample_from_dmll(angles.reshape(B*L, -1)).view(B, L, 1)

    def sample_sizes(self, x, class_labels, translations, angles):
        # Extract the sizes in local variables for convenience
        B, L, _ = class_labels.shape

        c = self.fc_class_labels(class_labels)
        cf = torch.cat([x, c], dim=-1)
        tx = self.pe_trans_x(translations[:, :, 0:1])
        ty = self.pe_trans_y(translations[:, :, 1:2])
        tz = self.pe_trans_z(translations[:, :, 2:3])
        tf = torch.cat([cf, tx, ty, tz], dim=-1)
        a = self.pe_angle_z(angles)
        sf = torch.cat([tf, a], dim=-1)

        sizes_x = self.size_layer_x(sf)
        sizes_y = self.size_layer_y(sf)
        sizes_z = self.size_layer_z(sf)

        s_x = sample_from_dmll(sizes_x.reshape(B*L, -1))
        s_y = sample_from_dmll(sizes_y.reshape(B*L, -1))
        s_z = sample_from_dmll(sizes_z.reshape(B*L, -1))
        return torch.cat([s_x, s_y, s_z], dim=-1).view(B, L, 3)

    # def sample_shape(self, x, class_labels, translations, angles, sizes): # This is the version for transformer as feature

    def sample_shape(self, x, class_labels, translations, angles, sizes): # This is the version 3
        B, L, _ = class_labels.shape
        c = self.fc_class_labels(class_labels)
        cf = torch.cat([x, c], dim=-1)
        tx = self.pe_trans_x(translations[:, :, 0:1])
        ty = self.pe_trans_y(translations[:, :, 1:2])
        tz = self.pe_trans_z(translations[:, :, 2:3])
        tf = torch.cat([cf, tx, ty, tz], dim=-1)
        a = self.pe_angle_z(angles)
        sf = torch.cat([tf, a], dim=-1)
        sx = self.pe_size_x(sizes[:, :, 0:1])
        sy = self.pe_size_y(sizes[:, :, 1:2])
        sz = self.pe_size_z(sizes[:, :, 2:3])


        points = torch.cat([sf, sx, sy, sz], dim=-1)
        points = self.shape_encoder(points)
        # latent_embd = self.pe_latent(shapes[..., 3].long())
        points = points.unsqueeze(2).repeat(1, 1, self.shape_embed, 1)
        # self.pe_x()
        points = points + self.pos_emb  # latent_embd
        points = points.squeeze(1)
        # import ipdb
        # ipdb.set_trace()
        for block in self.transformer.blocks[:6]:
            points = block(points)
        shape_x = self.shape_layer_x(self.ln_x(points))
        point_x = sample_from_dmll(shape_x.reshape(B * L * self.shape_embed, -1)).reshape(B, L, self.shape_embed, -1)
        x_embd = self.pe_x(point_x[..., 0])
        points = points.unsqueeze(1) + x_embd
        points = points.squeeze(1)
        # ipdb.set_trace()
        for block in self.transformer.blocks[6:8]:
            points = block(points)
        shape_y = self.shape_layer_y(self.ln_y(points))
        point_y = sample_from_dmll(shape_y.reshape(B*L*self.shape_embed, -1)).reshape(B, L, self.shape_embed, -1)
        y_embd = self.pe_y(point_y[..., 0])
        points = points.unsqueeze(1) + y_embd
        points = points.squeeze(1)
        # ipdb.set_trace()
        for block in self.transformer.blocks[8:10]:
            points = block(points)
        shape_z = self.shape_layer_z(self.ln_z(points))
        point_z = sample_from_dmll(shape_z.reshape(B * L * self.shape_embed, -1)).reshape(B, L, self.shape_embed, -1)
        z_embd = self.pe_z(point_z[..., 0])
        points = points.unsqueeze(1) + z_embd
        points = points.squeeze(1)
        # ipdb.set_trace()
        for block in self.transformer.blocks[10:12]:
            points = block(points)
        shape_latents = self.shape_layer_latent(self.ln_latent(points)).unsqueeze(1)

        # import ipdb
        # ipdb.set_trace()
        shapes = (
            point_x,
            point_y,
            point_z,
            F.log_softmax(shape_latents, dim=-1)
        )

        return shapes

    def pred_class_probs(self, x):
        class_labels = self.class_layer(x)

        # Extract the sizes in local variables for convenience
        b, l, _ = class_labels.shape
        c = self.n_classes

        # Sample the class
        class_probs = torch.softmax(class_labels, dim=-1).view(b*l, c)

        return class_probs

    def pred_dmll_params_translation(self, x, class_labels):
        def dmll_params_from_pred(pred):
            assert len(pred.shape) == 2

            N = pred.size(0)
            nr_mix = pred.size(1) // 3

            probs = torch.softmax(pred[:, :nr_mix], dim=-1)
            means = pred[:, nr_mix:2 * nr_mix]
            scales = torch.nn.functional.elu(pred[:, 2*nr_mix:3*nr_mix])
            scales = scales + 1.0001

            return probs, means, scales

        # Extract the sizes in local variables for convenience
        B, L, _ = class_labels.shape

        c = self.fc_class_labels(class_labels)
        cf = torch.cat([x, c], dim=-1)
        t_x = self.centroid_layer_x(cf).reshape(B*L, -1)
        t_y = self.centroid_layer_y(cf).reshape(B*L, -1)
        t_z = self.centroid_layer_z(cf).reshape(B*L, -1)

        return dmll_params_from_pred(t_x), dmll_params_from_pred(t_y),\
            dmll_params_from_pred(t_z)

    def pe_x(self, shape):
        shape = ((shape + 1) / 2 * 255).long()
        shape = self.pos_x(shape)
        return shape

    def pe_y(self, shape):
        shape = ((shape + 1) / 2 * 255).long()
        shape = self.pos_y(shape)
        return shape

    def pe_z(self, shape):
        shape = ((shape + 1) / 2 * 255).long()
        shape = self.pos_z(shape)
        return shape

    def pe_latent(self, shape):
        # shape = ((shape + 1) / 2 * 255).long()
        shape = self.pos_shape(shape)
        return shape

    def forward(self, x, sample_params):
        if self.with_extra_fc:
            x = self.hidden2output(x)

        #How should this be integrated for decoder

        # Extract the target properties from sample_params and embed them into
        # a higher dimensional space.

        target_properties = \
            AutoregressiveDMLLShape._extract_properties_from_target(
                sample_params
            )

        class_labels = target_properties[0]
        translations = target_properties[1]
        angles = target_properties[3]
        sizes = target_properties[2]
        # if self.stage == "second":
        shapes = target_properties[4]


        c = self.fc_class_labels(class_labels)

        tx = self.pe_trans_x(translations[:, :, 0:1])
        ty = self.pe_trans_y(translations[:, :, 1:2])
        tz = self.pe_trans_z(translations[:, :, 2:3])

        a = self.pe_angle_z(angles)
        # ipdb.set_trace()

        sx = self.pe_size_x(sizes[:, :, 0:1])
        sy = self.pe_size_y(sizes[:, :, 1:2])
        sz = self.pe_size_z(sizes[:, :, 2:3])

        # Decoding the latent code from here.
        # The first 512 feature for previous R,T,S,a,c etc.
        class_labels = self.class_layer(x)

        cf = torch.cat([x, c], dim=-1)
        # pointx = c
        # ipdb.set_trace()

        translations = (
            self.centroid_layer_x(cf),
            self.centroid_layer_y(cf),
            self.centroid_layer_z(cf)
        )
        tf = torch.cat([cf, tx, ty, tz], dim=-1)
        angles = self.angle_layer(tf)

        sf = torch.cat([tf, a], dim=-1)
        sizes = (
            self.size_layer_x(sf),
            self.size_layer_y(sf),
            self.size_layer_z(sf)
        )

        # if self.stage == "second":
            # Solution #5 for transformer decoder
        points = torch.cat([sf, sx, sy, sz], dim=-1)
        points = self.shape_encoder(points)
        x_embd = self.pe_x(shapes[..., 0])
        y_embd = self.pe_y(shapes[..., 1])
        z_embd = self.pe_z(shapes[..., 2])
        # latent_embd = self.pe_latent(shapes[..., 3].long())
        points = points.unsqueeze(2).repeat(1, 1, self.shape_embed, 1)

        # This is one way, you can also refer to another one with k-1 to predict k.
        # self.pe_x()

        points = points + self.pos_emb  # latent_embd
        points = points.squeeze(1)
        for block in self.transformer.blocks[:6]:
            points = block(points)
        shape_x = self.shape_layer_x(self.ln_x(points))
        points = points.unsqueeze(1) + x_embd
        points = points.squeeze(1)
        for block in self.transformer.blocks[6:8]:
            points = block(points)
        shape_y = self.shape_layer_y(self.ln_y(points))
        points = points.unsqueeze(1) + y_embd
        points = points.squeeze(1)
        for block in self.transformer.blocks[8:10]:
            points = block(points)
        shape_z = self.shape_layer_z(self.ln_z(points))
        points = points.unsqueeze(1) + z_embd
        points = points.squeeze(1)
        for block in self.transformer.blocks[10:12]:
            points = block(points)
        shape_latents = self.shape_layer_latent(self.ln_latent(points))

        shapes = (
            shape_x,
            shape_y,
            shape_z,
            F.log_softmax(shape_latents, dim=-1)
        )
        return self.bbox_output(sizes, translations, angles, class_labels, shapes)
        # else:
        #     return self.bbox_output(sizes, translations, angles, class_labels)

        # import ipdb
        # ipdb.set_trace()


        #Solution #4 using dmll for loss
        # points = self.shape_encoder(points)
        # points = points.unsqueeze(2).repeat(1, 1, 512, 1)
        # points = points + self.pos_emb
        # shape_x = self.shape_layer_x(points)
        # points = points + self.pe_shape_x(shapes[..., 0:1])
        # shape_y = self.shape_layer_y(points)
        # points = points + self.pe_shape_y(shapes[..., 1:2])
        # shape_z = self.shape_layer_z(points)
        # points = points + self.pe_shape_z(shapes[..., 2:3])
        # shape_latents = self.shape_layer_latent(points)
        #
        # shapes = (
        #     shape_x,
        #     shape_y,
        #     shape_z,
        #     F.log_softmax(shape_latents, dim=-1)
        # )



        # Solution #3

        # shape_x = self.shape_layer_x(points)
        # spx = self.pe_shape_x(shape_x[..., None])
        # points = torch.cat([points.unsqueeze(2).repeat(1, 1, 512, 1), spx], dim=-1)
        # shape_y = self.shape_layer_y(points)
        # points = torch.cat([points, self.pe_shape_y(shape_y)], dim=-1)
        # shape_z = self.shape_layer_z(points)
        # points = torch.cat([points, self.pe_shape_z(shape_z)], dim=-1)
        # shape_feature = self.shape_feature(points)
        # shapes = (
        #     shape_x[..., None],
        #     shape_y,
        #     shape_z,
        #     F.log_softmax(shape_feature, dim=-1) #shape_feature
        # )
        #Solution #2

        # shape_x = self.shape_layer_x(points)
        # points = torch.cat([points, self.pe_shape_x(shape_x)], dim = -1)
        #
        # # shape_x = self.shape_layer_x(points)
        # shape_y = self.shape_layer_y(points)
        # shape_z = self.shape_layer_z(points)
        # # import ipdb
        # # ipdb.set_trace()
        #
        # shape_feature = torch.cat([
        #     shape_x[..., None],
        #     shape_y[..., None],
        #     shape_z[..., None],
        #     points[:, :, None].repeat(1, 1, 512, 1),
        # ], dim=-1)
        # shape_feature = self.shape_feature(shape_feature)


        # pointx = self.feature_aggregate(pointx).reshape(*pointx.shape[:2], self.nxt_hidden_size,  -1)

        #shape how to embed
        # shapes = target_properties[4]
        # pred_x = self.point_layer_x(pointx)
        # pointx = torch.cat([pointx, pred_x], dim=-1)
        # pred_y = self.point_layer_y(pointx)
        # pointx = torch.cat([pointx, pred_y], dim=-1)
        # pred_z = self.point_layer_z(pointx)
        # pointx = torch.cat([pointx, pred_z], dim=-1)
        # pred_feature = self.feature_layer(pointx)
        # shapes = (
        #     F.log_softmax(self.x_logits(pred_x[..., None]), dim=-1),
        #     self.y_logits(pred_y[..., None]),
        #     self.z_logits(pred_z[..., None]),
        #     self.feature_logits(pred_feature[..., None])
        # )
        #
        # previous is a failure version, not well currently.

        # pred_x = F.log_softmax(self.point_layer_x(pointx), dim=-1)
        # pred_y = F.log_softmax(self.point_layer_y(pointx), dim=-1)
        # pred_z = F.log_softmax(self.point_layer_z(pointx), dim=-1)
        # pred_feature = F.log_softmax(self.feature_layer(pointx), dim=-1)
        # shapes = (
        #     pred_x,
        #     pred_y,
        #     pred_z,
        #     pred_feature
        # )

        #previous also failed with so much feature embedding






def get_bbox_outputShape(bbox_type):
    return {
        "autoregressive_mlc_shape": AutoregressiveBBoxOutputShape
    }[bbox_type]
