#
# Copyright (C) 2021 NVIDIA Corporation.  All rights reserved.
# Licensed under the NVIDIA Source Code License.
# See LICENSE at https://github.com/nv-tlabs/ATISS.
# Authors: Despoina Paschalidou, Amlan Kar, Maria Shugrina, Karsten Kreis,
#          Andreas Geiger, Sanja Fidler
#

import torch

from ..losses import cross_entropy_loss, dmll
from ..stats_logger import StatsLogger
from torch import nn


#FIXME: this should be revisited
class BBoxOutputShape(object):
    def __init__(self, sizes, translations, angles, class_labels, shape):
        self.sizes = sizes
        self.translations = translations
        self.angles = angles
        self.class_labels = class_labels
        self.shape = shape

    def __len__(self):
        return len(self.members)

    @property
    def members(self):
        return (self.sizes, self.translations, self.angles, self.class_labels)

    @property
    def n_classes(self):
        return self.class_labels.shape[-1]

    @property
    def device(self):
        return self.class_labels.device

    @staticmethod
    def extract_bbox_params_from_tensor(t):
        if isinstance(t, dict):
            class_labels = t["class_labels_tr"]
            translations = t["translations_tr"]
            sizes = t["sizes_tr"]
            angles = t["angles_tr"]
            shape = t["shapes_tr"]
        else:
            assert len(t.shape) == 3
            class_labels = t[:, :, :-7]
            translations = t[:, :, -7:-4]
            sizes = t[:, :, -4:-1]
            angles = t[:, :, -1:]

        return class_labels, translations, sizes, angles, shape

    @property
    def feature_dims(self):
        raise NotImplementedError()

    def get_losses(self, X_target):
        raise NotImplementedError()

    def reconstruction_loss(self, sample_params):
        raise NotImplementedError()

#FIXME: revised for shape feature
class AutoregressiveBBoxOutputShape(BBoxOutputShape):
    def __init__(self, sizes, translations, angles, class_labels, shapes):
        self.sizes_x, self.sizes_y, self.sizes_z = sizes
        self.translations_x, self.translations_y, self.translations_z = \
            translations
        self.class_labels = class_labels
        self.angles = angles
        self.shape_x, self.shape_y, self.shape_z, self.shape_logits = \
            (u.squeeze(1) for u in shapes)
        # self.coords_loss = torch.nn.MSELoss()
        self.shape_loss = torch.nn.NLLLoss()
        # self.shapes = shapes
        # this should be divided into a points and corresponding features

    @property
    def members(self):
        return (
            self.sizes_x, self.sizes_y, self.sizes_z,
            self.translations_x, self.translations_y, self.translations_z,
            self.angles, self.class_labels, self.shapes
        )

    @property
    def feature_dims(self):
        return self.n_classes + 3 + 3 + 1

    def _targets_from_tensor(self, X_target):
        # Make sure that everything has the correct shape
        # Extract the bbox_params for the target tensor
        target_bbox_params = self.extract_bbox_params_from_tensor(X_target)
        target = {}
        target["labels"] = target_bbox_params[0]
        target["translations_x"] = target_bbox_params[1][:, :, 0:1]
        target["translations_y"] = target_bbox_params[1][:, :, 1:2]
        target["translations_z"] = target_bbox_params[1][:, :, 2:3]
        target["sizes_x"] = target_bbox_params[2][:, :, 0:1]
        target["sizes_y"] = target_bbox_params[2][:, :, 1:2]
        target["sizes_z"] = target_bbox_params[2][:, :, 2:3]
        target["angles"] = target_bbox_params[3]

        # FIXME when use other code for correction

        # if target_bbox_params[4][..., 0].squeeze(1).max()>2:
        # target["shapes_x"] = ((target_bbox_params[4][..., 0].squeeze(1) + 1) / 2 * 255).long()
        # target["shapes_y"] = ((target_bbox_params[4][..., 1].squeeze(1) + 1) / 2 * 255).long()
        # target["shapes_z"] = ((target_bbox_params[4][..., 2].squeeze(1) + 1) / 2 * 255).long()
        # else:
        target["shapes_x"] = target_bbox_params[4][..., 0].squeeze(1)
        target["shapes_y"] = target_bbox_params[4][..., 1].squeeze(1)
        target["shapes_z"] = target_bbox_params[4][..., 2].squeeze(1)


        target["shapes_logits"] = target_bbox_params[4][..., 3].squeeze(1).long()

        return target

    def get_losses(self, X_target):
        target = self._targets_from_tensor(X_target)

        assert torch.sum(target["labels"][..., -2]).item() == 0

        # For the class labels compute the cross entropy loss between the
        # target and the predicted labels
        label_loss = cross_entropy_loss(self.class_labels, target["labels"])

        # For the translations, sizes and angles compute the discretized
        # logistic mixture likelihood as described in
        # PIXELCNN++: Improving the PixelCNN with Discretized Logistic Mixture Likelihood and
        # Other Modifications, by Salimans et al.
        translation_loss = dmll(self.translations_x, target["translations_x"])
        translation_loss += dmll(self.translations_y, target["translations_y"])
        translation_loss += dmll(self.translations_z, target["translations_z"])
        # import ipdb
        # ipdb.set_trace()
        size_loss = dmll(self.sizes_x, target["sizes_x"])
        size_loss += dmll(self.sizes_y, target["sizes_y"])
        size_loss += dmll(self.sizes_z, target["sizes_z"])
        angle_loss = dmll(self.angles, target["angles"])
        # import ipdb
        # ipdb.set_trace()
        # coord_loss = self.shape_loss()
        coord_loss = dmll(self.shape_x, target["shapes_x"][..., None])
        coord_loss += dmll(self.shape_y, target["shapes_y"][..., None])
        coord_loss += dmll(self.shape_z, target["shapes_z"][..., None])
        # previous version
        #FIXME when use other code for correction
        # coord_loss = self.shape_loss(self.shape_x.transpose(1, 2), target["shapes_x"])
        # coord_loss += self.shape_loss(self.shape_y.transpose(1, 2), target["shapes_y"])
        # coord_loss += self.shape_loss(self.shape_z.transpose(1, 2), target["shapes_z"])
        shape_loss = self.shape_loss(self.shape_logits.transpose(1, 2), target["shapes_logits"]) * 5

        return label_loss, translation_loss, size_loss, angle_loss, coord_loss, shape_loss

    def reconstruction_loss(self, X_target, lengths):
        # Compute the losses

        label_loss, translation_loss, size_loss, angle_loss, coord_loss, shape_loss = \
            self.get_losses(X_target)

        label_loss = label_loss.mean()
        translation_loss = translation_loss.mean()
        size_loss = size_loss.mean()
        angle_loss = angle_loss.mean()
        coord_loss = coord_loss.mean()
        shape_loss = shape_loss.mean()

        StatsLogger.instance()["losses.size"].value = size_loss.item()
        StatsLogger.instance()["losses.translation"].value = \
            translation_loss.item()
        StatsLogger.instance()["losses.angle"].value = angle_loss.item()
        StatsLogger.instance()["losses.label"].value = label_loss.item()
        StatsLogger.instance()["losses.coords"].value = coord_loss.item()
        StatsLogger.instance()["losses.shape"].value = shape_loss.item()

        # return translation_loss + size_loss + angle_loss
        return label_loss + translation_loss + size_loss + angle_loss + coord_loss + shape_loss
