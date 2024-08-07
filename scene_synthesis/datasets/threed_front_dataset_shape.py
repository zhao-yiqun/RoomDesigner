#
# Copyright (C) 2021 NVIDIA Corporation.  All rights reserved.
# Licensed under the NVIDIA Source Code License.
# See LICENSE at https://github.com/nv-tlabs/ATISS.
# Authors: Despoina Paschalidou, Amlan Kar, Maria Shugrina, Karsten Kreis,
#          Andreas Geiger, Sanja Fidler
#

import numpy as np

from functools import lru_cache
from scipy.ndimage import rotate

import torch
from torch.utils.data import Dataset


class DatasetDecoratorBaseShape(Dataset):
    """A base class that helps us implement decorators for ThreeDFront-like
    datasets."""
    def __init__(self, dataset):
        self._dataset = dataset

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        return self._dataset[idx]

    @property
    def bounds(self):
        return self._dataset.bounds

    @property
    def n_classes(self):
        return self._dataset.n_classes

    @property
    def class_labels(self):
        return self._dataset.class_labels

    @property
    def shape_feature(self):
        return self._dataset.shape

    @property
    def class_frequencies(self):
        return self._dataset.class_frequencies

    @property
    def n_object_types(self):
        return self._dataset.n_object_types

    @property
    def object_types(self):
        return self._dataset.object_types

    @property
    def feature_size(self):
        return self.bbox_dims + self.n_classes

    @property
    def bbox_dims(self):
        raise NotImplementedError()

    def post_process(self, s):
        return self._dataset.post_process(s)


class BoxOrderedDatasetshape(DatasetDecoratorBaseShape):
    def __init__(self, dataset, box_ordering=None):
        super().__init__(dataset) # This dataset is from CachedShape dataset
        self.box_ordering = box_ordering

    @lru_cache(maxsize=16)
    def _get_boxes(self, scene_idx):
        scene = self._dataset[scene_idx]
        if self.box_ordering is None:
            return scene.bboxes
        elif self.box_ordering == "class_frequencies":
            return scene.ordered_bboxes_with_class_frequencies(
                self.class_frequencies
            )
        else:
            raise NotImplementedError()


class DataEncodershape(BoxOrderedDatasetshape):
    """DataEncoder is a wrapper for all datasets we have
    """
    @property
    def property_type(self):
        raise NotImplementedError()


class RoomLayoutEncodershape(DataEncodershape):
    @property
    def property_type(self):
        return "room_layout"

    def __getitem__(self, idx):
        """Implement the encoding for the room layout as images."""
        img = self._dataset[idx].room_mask[:, :, 0:1]
        return np.transpose(img, (2, 0, 1))

    @property
    def bbox_dims(self):
        return 0


class ClassLabelsEncodershape(DataEncodershape):
    """Implement the encoding for the class labels."""
    @property
    def property_type(self):
        return "class_labels"

    def __getitem__(self, idx):
        # Make a local copy of the class labels
        classes = self.class_labels

        # Get the scene
        boxes = self._get_boxes(idx)
        L = len(boxes)  # sequence length
        C = len(classes)  # number of classes
        class_labels = np.zeros((L, C), dtype=np.float32)
        for i, bs in enumerate(boxes):
            class_labels[i] = bs.one_hot_label(classes)
        return class_labels

    @property
    def bbox_dims(self):
        return 0


class TranslationEncodershape(DataEncodershape):
    @property
    def property_type(self):
        return "translations"

    def __getitem__(self, idx):
        # Get the scene
        scene = self._dataset[idx]
        boxes = self._get_boxes(idx)
        L = len(boxes)  # sequence length
        translations = np.zeros((L, 3), dtype=np.float32)
        for i, bs in enumerate(boxes):
            translations[i] = bs.centroid(-scene.centroid)
        return translations

    @property
    def bbox_dims(self):
        return 3


class SizeEncodershape(DataEncodershape):
    @property
    def property_type(self):
        return "sizes"

    def __getitem__(self, idx):
        # Get the scene
        boxes = self._get_boxes(idx)
        L = len(boxes)  # sequence length
        sizes = np.zeros((L, 3), dtype=np.float32)
        for i, bs in enumerate(boxes):
            sizes[i] = bs.size
        return sizes

    @property
    def bbox_dims(self):
        return 3


class AngleEncodershape(DataEncodershape):
    @property
    def property_type(self):
        return "angles"

    def __getitem__(self, idx):
        # Get the scene
        boxes = self._get_boxes(idx)
        # Get the rotation matrix for the current scene
        L = len(boxes)  # sequence length
        angles = np.zeros((L, 1), dtype=np.float32)
        for i, bs in enumerate(boxes):
            angles[i] = bs.z_angle
        return angles

    @property
    def bbox_dims(self):
        return 1

# Shape Encoder should be refined
class ShapeEncodershape(DataEncodershape):
    @property
    def property_type(self):
        return "shape"

    def __getitem__(self, idx):
        # Get the scene shape
        boxes = self._get_boxes(idx)
        L = len(boxes)  # sequence length
        sizes = np.zeros((L, 3), dtype=np.float32)
        for i, bs in enumerate(boxes):
            sizes[i] = bs.size
        return sizes




class DatasetCollectionshape(DataEncodershape):
    def __init__(self, *datasets):
        super().__init__(datasets[0])
        self._datasets = datasets

    @property
    def bbox_dims(self):
        return sum(d.bbox_dims for d in self._datasets)

    def __getitem__(self, idx):
        # import ipdb
        # ipdb.set_trace()
        sample_params = {}
        for di in self._datasets:
            sample_params[di.property_type] = di[idx]
        return sample_params

    @staticmethod
    def collate_fn(samples):
        # We assume that all samples have the same set of keys
        key_set = set(samples[0].keys()) - set(["length"])

        # Compute the max length of the sequences in the batch
        max_length = max(sample["length"] for sample in samples)

        # Assume that all inputs that are 3D or 1D do not need padding.
        # Otherwise, pad the first dimension.
        padding_keys = set(k for k in key_set if len(samples[0][k].shape) == 2)- set(["shapes_tr"])
        padding_keys.add("shapes")
        sample_params = {}
        import ipdb
        # ipdb.set_trace()
        sample_params.update({
            k: np.stack([sample[k] for sample in samples], axis=0)
            for k in (key_set-padding_keys)
        })
        # ipdb.set_trace()
        # padding_keys = padding_keys- set(["shapes"])
        sample_params.update({
            k: np.stack([
                np.vstack([
                    sample[k],
                    np.zeros((max_length-len(sample[k]), *sample[k].shape[1:]))
                ]) for sample in samples
            ], axis=0)
            for k in padding_keys
        })
        # ipdb.set_trace()
        # padding_keys.add("shapes")
        sample_params["lengths"] = np.array([
            sample["length"] for sample in samples
        ])

        # Make torch tensors from the numpy tensors
        torch_sample = {
            k: torch.from_numpy(sample_params[k]).float()
            for k in sample_params
        }

        torch_sample.update({
            k: torch_sample[k][:, None]
            for k in torch_sample.keys()
            if "_tr" in k
        })

        return torch_sample


class CachedDatasetCollectionshape(DatasetCollectionshape):
    def __init__(self, dataset):
        super().__init__(dataset)
        self._dataset = dataset

    def __getitem__(self, idx):
        return self._dataset.get_room_params(idx)

    @property
    def bbox_dims(self):
        return self._dataset.bbox_dims


class RotationAugmentation(DatasetDecoratorBaseShape):
    def __init__(self, dataset, min_rad=0.174533, max_rad=5.06145):
        super().__init__(dataset)
        self._min_rad = min_rad
        self._max_rad = max_rad

    @staticmethod
    def rotation_matrix_around_y(theta):
        R = np.zeros((3, 3))
        R[0, 0] = np.cos(theta)
        R[0, 2] = -np.sin(theta)
        R[2, 0] = np.sin(theta)
        R[2, 2] = np.cos(theta)
        R[1, 1] = 1.
        return R

    @property
    def rot_angle(self):
        if np.random.rand() < 0.5:
            return np.random.uniform(self._min_rad, self._max_rad)
        else:
            return 0.0

    def __getitem__(self, idx):
        # Get the rotation matrix for the current scene
        rot_angle = self.rot_angle
        R = RotationAugmentation.rotation_matrix_around_y(rot_angle)

        sample_params = self._dataset[idx]
        for k, v in sample_params.items():
            if k == "translations":
                sample_params[k] = v.dot(R)
            elif k == "angles":
                angle_min, angle_max = self.bounds["angles"]
                sample_params[k] = \
                    (v + rot_angle - angle_min) % (2 * np.pi) + angle_min
            elif k == "room_layout":
                # Fix the ordering of the channels because it was previously
                # changed
                img = np.transpose(v, (1, 2, 0))
                sample_params[k] = np.transpose(rotate(
                    img, rot_angle * 180 / np.pi, reshape=False
                ), (2, 0, 1))
        return sample_params

# This is for scale the input to the normalized dim,
# with both scaled to [-1, 1]
class Scale(DatasetDecoratorBaseShape):
    @staticmethod
    def scale(x, minimum, maximum):
        X = x.astype(np.float32)
        X = np.clip(X, minimum, maximum)
        X = ((X - minimum) / (maximum - minimum))
        X = 2 * X - 1
        return X

    @staticmethod
    def descale(x, minimum, maximum):
        x = (x + 1) / 2
        x = x * (maximum - minimum) + minimum
        return x

    def __getitem__(self, idx):
        bounds = self.bounds
        sample_params = self._dataset[idx]
        for k, v in sample_params.items():
            if k in bounds:
                sample_params[k] = Scale.scale(
                    v, bounds[k][0], bounds[k][1]
                )
        return sample_params

    def post_process(self, s):
        bounds = self.bounds
        sample_params = {}
        for k, v in s.items():
            if k == "room_layout" or k == "class_labels":
                sample_params[k] = v
            elif k == "shapes" or k == "texture":
                sample_params[k] = v
            else:
                # import ipdb
                # ipdb.set_trace()
                sample_params[k] = Scale.descale(
                    v, bounds[k][0], bounds[k][1]
                )
        return super().post_process(sample_params)

    @property
    def bbox_dims(self):
        return 3 + 3 + 1


class Jitter(DatasetDecoratorBaseShape):
    def __getitem__(self, idx):
        sample_params = self._dataset[idx]
        for k, v in sample_params.items():
            if k == "room_layout" or k == "class_labels":
                sample_params[k] = v
            else:
                sample_params[k] = v + np.random.normal(0, 0.01)
        return sample_params


class Permutation(DatasetDecoratorBaseShape):
    def __init__(self, dataset, permutation_keys, permutation_axis=0):
        super().__init__(dataset)
        self._permutation_keys = permutation_keys
        self._permutation_axis = permutation_axis
        # import ipdb
        # ipdb.set_trace()

    def __getitem__(self, idx):
        sample_params = self._dataset[idx]

        shapes = sample_params["class_labels"].shape
        ordering = np.random.permutation(shapes[self._permutation_axis])

        for k in self._permutation_keys:
            sample_params[k] = sample_params[k][ordering]
        return sample_params


class OrderedDatasetshape(DatasetDecoratorBaseShape): #Note: final encoded function
    def __init__(self, dataset, ordered_keys, box_ordering=None):
        super().__init__(dataset)
        self._ordered_keys = ordered_keys
        self._box_ordering = box_ordering

    def __getitem__(self, idx):
        if self._box_ordering is None:
            return self._dataset[idx]

        if self._box_ordering != "class_frequencies":
            raise NotImplementedError()
        sample = self._dataset[idx]
        order = self._get_class_frequency_order(sample)
        for k in self._ordered_keys:
            sample[k] = sample[k][order]
        return sample

    def _get_class_frequency_order(self, sample):
        t = sample["translations"]
        c = sample["class_labels"].argmax(-1)
        class_frequencies = self.class_frequencies
        class_labels = self.class_labels
        f = np.array([
            [class_frequencies[class_labels[ci]]]
            for ci in c
        ])

        return np.lexsort(np.hstack([t, f]).T)[::-1]


class Autoregressive(DatasetDecoratorBaseShape):
    def __getitem__(self, idx):
        sample_params = self._dataset[idx]
        sample_params_target = {}
        # Compute the target from the input
        for k, v in sample_params.items():
            if k == "room_layout" or k == "length":
                pass
            elif k == "class_labels":
                class_labels = np.copy(v)
                L, C = class_labels.shape
                # Add the end label the end of each sequence
                end_label = np.eye(C)[-1]
                sample_params_target[k+"_tr"] = np.vstack([
                    class_labels, end_label
                ])
            elif k == "shapes":
                p = np.copy(v)
                _, N, C = p.shape # p may be (B, 2048, 3) with 3 dimentions
                # import ipdb
                # ipdb.set_trace()
                sample_params_target[k+"_tr"] = np.vstack(
                    [p, np.zeros((1, N, C))]
                )
            else:
                p = np.copy(v)
                # Set the attributes to for the end symbol
                _, C = p.shape
                sample_params_target[k+"_tr"] = np.vstack([p, np.zeros(C)])


        sample_params.update(sample_params_target)

        # Add the number of bounding boxes in the scene
        sample_params["length"] = sample_params["class_labels"].shape[0]

        return sample_params

    def collate_fn(self, samples):
        return DatasetCollectionshape.collate_fn(samples)

    @property
    def bbox_dims(self):
        return 7


class AutoregressiveWOCM(Autoregressive):
    def __getitem__(self, idx):
        sample_params = super().__getitem__(idx)
        # Split the boxes and generate input sequences and target boxes
        L, C = sample_params["class_labels"].shape
        n_boxes = np.random.randint(0, L+1)

        for k, v in sample_params.items():
            if k == "room_layout" or k == "length":
                pass
            else:
                if "_tr" in k:
                    sample_params[k] = v[n_boxes]
                else:
                    sample_params[k] = v[:n_boxes]
        sample_params["length"] = n_boxes

        return sample_params

class Autoregressive_sceneformer(Autoregressive):
    def __getitem__(self, idx):
        sample_params = super().__getitem__(idx)
        # Split the boxes and generate input sequences and target boxes
        L, C = sample_params["class_labels"].shape
        n_boxes = np.random.randint(0, L+1)

        for k, v in sample_params.items():
            if k == "room_layout" or k == "length":
                pass
            else:
                if "_tr" in k:
                    sample_params[k] = v[n_boxes]
                else:
                    sample_params[k] = v[:n_boxes]
        sample_params["length"] = n_boxes

        return sample_params


# collect dataset from here
def dataset_encoding_factory_shape(
    name,
    dataset,
    augmentations=None,
    box_ordering=None
):
    # NOTE: The ordering might change after augmentations so really it should
    #       be done after the augmentations. For class frequencies it is fine
    #       though.
    if "cached" in name:
        if "shape" in name:
            dataset_collection = OrderedDatasetshape(
                CachedDatasetCollectionshape(dataset),
                ["class_labels", "translations", "sizes", "angles", "shapes"],
                box_ordering=box_ordering
            )
        else:
            raise NotImplementedError()
    else:
        box_ordered_dataset = BoxOrderedDatasetshape(
            dataset,
            box_ordering
        )
        room_layout = RoomLayoutEncodershape(box_ordered_dataset)
        class_labels = ClassLabelsEncodershape(box_ordered_dataset)
        translations = TranslationEncodershape(box_ordered_dataset)
        sizes = SizeEncodershape(box_ordered_dataset)
        angles = AngleEncodershape(box_ordered_dataset)

        dataset_collection = DatasetCollectionshape(
            room_layout,
            class_labels,
            translations,
            sizes,
            angles
        )

    if name == "basic":
        return DatasetCollectionshape(
            class_labels,
            translations,
            sizes,
            angles
        )


    if isinstance(augmentations, list):# this should be revised for a plausible layout
        for aug_type in augmentations:
            if aug_type == "rotations":
                print("Applying rotation augmentations")
                dataset_collection = RotationAugmentation(dataset_collection)
            elif aug_type == "jitter":
                print("Applying jittering augmentations")
                dataset_collection = Jitter(dataset_collection)

    # Scale the input
    dataset_collection = Scale(dataset_collection)
    if "eval" in name:
        return dataset_collection
    elif "wocm_no_prm" in name:
        return AutoregressiveWOCM(dataset_collection)
    elif "wocm" in name:
        dataset_collection = Permutation(
            dataset_collection,
            ["class_labels", "translations", "sizes", "angles", "shapes"]
        )
        return AutoregressiveWOCM(dataset_collection)
    else:
        raise NotImplementedError()
