# 
# Copyright (C) 2021 NVIDIA Corporation.  All rights reserved.
# Licensed under the NVIDIA Source Code License.
# See LICENSE at https://github.com/nv-tlabs/ATISS.
# Authors: Despoina Paschalidou, Amlan Kar, Maria Shugrina, Karsten Kreis,
#          Andreas Geiger, Sanja Fidler
# 

from functools import partial

import ipdb
import torch
try:
    from radam import RAdam
except ImportError:
    pass

from .autoregressive_transformer import AutoregressiveTransformer, \
    AutoregressiveTransformerPE, \
    train_on_batch as train_on_batch_simple_autoregressive, \
    validate_on_batch as validate_on_batch_simple_autoregressive

from .autoregressive_transformer_shape import AutoregressiveTransformerShape,\
    AutoregressiveTransformerPEShape,\
    train_on_batch as train_on_batch_shape_autoregressive,\
    validate_on_batch as validate_on_batch_shape_autoregressive

from .autoregressive_transformer_open import AutoregressiveTransformerOpen,\
    train_on_batch as train_on_batch_open_autoregressive,\
    validate_on_batch as validate_on_batch_open_autoregressive

from .autoregressive_transformer_all import AutoregressiveTransformerAll,\
    AutoregressiveTransformerPEAll,\
    train_on_batch_all as train_on_batch_all_autoregressive,\
    validate_on_batch_all as validate_on_batch_all_autoregressive

from .hidden_to_output import AutoregressiveDMLL, get_bbox_output
# from .hidden_to_output_shape import AutoregressiveDMLLShape, get_bbox_outputShape
from .hidden_to_output_open import AutoregressiveDMLLOpen, get_bbox_outputOpen
from .hidden_to_output_shapes import AutoregressiveDMLLShape, get_bbox_outputShape
from .hidden_to_output_all import AutoregressiveDMLLAll, get_bbox_outputAll
from .feature_extractors import get_feature_extractor


def hidden2output_layer(config, n_classes):
    config_n = config["network"]
    hidden2output_layer = config_n.get("hidden2output_layer")
    # ipdb.set_trace()

    if hidden2output_layer == "autoregressive_mlc":
        return AutoregressiveDMLL(
            config_n.get("hidden_dims", 768),
            n_classes,
            config_n.get("n_mixtures", 4),
            get_bbox_output(config_n.get("bbox_output", "autoregressive_mlc")),
            config_n.get("with_extra_fc", False),
        )
    elif hidden2output_layer =="autoregressive_mlc_shape":

        if config_n.get("stage", "second") == "first":
            return AutoregressiveDMLLShape(
                config_n.get("hidden_dims", 768),
                n_classes,
                config_n.get("n_mixtures", 4),
                get_bbox_output("autoregressive_mlc"),
                config_n.get("shape_dimensions", 512),
                config_n.get("with_extra_fc", False),
            )
        else:
            return AutoregressiveDMLLShape(
                config_n.get("hidden_dims", 768),
                n_classes,
                config_n.get("n_mixtures", 4),
                get_bbox_outputShape(config_n.get("bbox_output", "autoregressive_mlc_shape")),
                config_n.get("shape_dimensions", 512),
                #FIXME
                # config_n.get("stage", "second"),
                config_n.get("with_extra_fc", False),
            )
    elif hidden2output_layer == "autoregressive_mlc_open":
        return AutoregressiveDMLLOpen(
            config_n.get("hidden_dims", 768),
            n_classes,
            config_n.get("n_mixtures", 4),
            get_bbox_outputOpen(config_n.get("bbox_output", "autoregressive_mlc_open")),
            config_n.get("with_extra_fc", False),
        )
    elif hidden2output_layer == "autoregressive_mlc_all":
        if config_n.get("stage", "first") == "first":
            return AutoregressiveDMLLAll(
                config_n.get("hidden_dims", 768),
                n_classes,
                config_n.get("n_mixtures", 4),
                get_bbox_output("autoregressive_mlc"),
                config_n.get("stage", "first"),
                config_n.get("with_extra_fc", False),
            )
        else:
            return AutoregressiveDMLLAll(
                config_n.get("hidden_dims", 768),
                n_classes,
                config_n.get("n_mixtures", 4),
                get_bbox_outputAll(config_n.get("bbox_output", "autoregressive_mlc_all")),
                config_n.get("stage", "first"),
                config_n.get("with_extra_fc", False),
            )
    else:
        raise NotImplementedError()


def optimizer_factory(config, parameters):
    """Based on the provided config create the suitable optimizer."""
    optimizer = config.get("optimizer", "Adam")
    lr = config.get("lr", 1e-3)
    momentum = config.get("momentum", 0.9)
    # weight_decay = config.get("weight_decay", 0.0)
    # Weight decay was set to 0.0 in the paper's experiments. We note that
    # increasing the weight_decay deteriorates performance.
    weight_decay = 0.0

    if optimizer == "SGD":
        return torch.optim.SGD(
            parameters, lr=lr, momentum=momentum, weight_decay=weight_decay
        )
    elif optimizer == "Adam":
        return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
    elif optimizer == "RAdam":
        return RAdam(parameters, lr=lr, weight_decay=weight_decay)
    else:
        raise NotImplementedError()

def optimizer_factory_stage2(config, network):
    """Based on the provided config create the suitable optimizer."""
    optimizer = config.get("optimizer", "Adam")
    lr = config.get("lr", 1e-3)
    momentum = config.get("momentum", 0.9)

    # weight_decay = config.get("weight_decay", 0.0)
    # Weight decay was set to 0.0 in the paper's experiments. We note that
    # increasing the weight_decay deteriorates performance.
    weight_decay = 0.0

    if optimizer == "SGD":
        return torch.optim.SGD(
            network, lr=lr, momentum=momentum, weight_decay=weight_decay
        )
    elif optimizer == "Adam":
        texture_group = []
        shape_group = []
        dof_group = []
        other_group = []
        for k, v in network.named_parameters():
            if "hidden2output.texture" in k:
                texture_group += [v]
            elif "hidden2output.transformer" in k  or "hidden2output.pos" in k  :
                shape_group += [v]
            # elif "shape2hidden" in k or "shape_decode" in k or "texture2hidden" in k:
            #     dof_group += [v]
            #     print(k)
            else:
                other_group += [v]
        # print(dof_group)
        # train from scratch loss
        return torch.optim.Adam([
            {'params': shape_group, 'lr': lr},
            {'params': texture_group, 'lr': lr * 0.1},
            {'params': other_group, 'lr': lr }
        ], )
        # fin-tuned loss
        # return torch.optim.Adam([
        #         {'params': shape_group, 'lr': lr },
        #         {'params': texture_group, 'lr': lr*0.1},
        #         {'params': other_group, 'lr': lr*0.2 }
        #     ], )
        #return torch.optim.Adam(network, lr=lr, weight_decay=weight_decay)
    elif optimizer == "RAdam":
        return RAdam(network, lr=lr, weight_decay=weight_decay)
    else:
        raise NotImplementedError()
# optimizer = optimizer_factory(config["training"], network.parameters())

def build_network(
    input_dims,
    n_classes,
    config,
    weight_file=None,
    device="cpu"):
    network_type = config["network"]["type"]

    if network_type == "autoregressive_transformer":
        train_on_batch = train_on_batch_simple_autoregressive
        validate_on_batch = validate_on_batch_simple_autoregressive
        network = AutoregressiveTransformer(
            input_dims,
            hidden2output_layer(config, n_classes),
            get_feature_extractor(
                config["feature_extractor"].get("name", "resnet18"),
                freeze_bn=config["feature_extractor"].get("freeze_bn", True),
                input_channels=config["feature_extractor"].get("input_channels", 1),
                feature_size=config["feature_extractor"].get("feature_size", 256),
            ),
            config["network"]
        )
    elif network_type == "autoregressive_transformer_pe":
        train_on_batch = train_on_batch_simple_autoregressive
        validate_on_batch = validate_on_batch_simple_autoregressive
        network = AutoregressiveTransformerPE(
            input_dims,
            hidden2output_layer(config, n_classes),
            get_feature_extractor(
                config["feature_extractor"].get("name", "resnet18"),
                freeze_bn=config["feature_extractor"].get("freeze_bn", True),
                input_channels=config["feature_extractor"].get("input_channels", 1),
                feature_size=config["feature_extractor"].get("feature_size", 256),
            ),
            config["network"]
        )
    elif network_type == "autoregressive_transformer_shape":
        train_on_batch = train_on_batch_shape_autoregressive
        validate_on_batch = validate_on_batch_shape_autoregressive
        network = AutoregressiveTransformerShape(
            input_dims,
            hidden2output_layer(config, n_classes),
            get_feature_extractor(
                config["feature_extractor"].get("name", "resnet18"),
                freeze_bn=config["feature_extractor"].get("freeze_bn", True),
                input_channels=config["feature_extractor"].get("input_channels", 1),
                feature_size=config["feature_extractor"].get("feature_size", 256),
            ),
            config["network"],
            stage=config["network"].get("stage", "first")
        )
    elif network_type == "autoregressive_transformer_open":
        train_on_batch = train_on_batch_open_autoregressive
        validate_on_batch = validate_on_batch_open_autoregressive
        network = AutoregressiveTransformerOpen(
            input_dims,
            hidden2output_layer(config, n_classes),
            get_feature_extractor(
                config["feature_extractor"].get("name", "resnet18"),
                freeze_bn=config["feature_extractor"].get("freeze_bn", True),
                input_channels=config["feature_extractor"].get("input_channels", 1),
                feature_size=config["feature_extractor"].get("feature_size", 256),
            ),
            config["network"]
        )
    elif network_type == "autoregressive_transformer_all":
        train_on_batch = train_on_batch_all_autoregressive
        validate_on_batch = validate_on_batch_all_autoregressive
        network = AutoregressiveTransformerAll(
            input_dims,
            hidden2output_layer(config, n_classes),
            get_feature_extractor(
                config["feature_extractor"].get("name", "resnet18"),
                freeze_bn=config["feature_extractor"].get("freeze_bn", True),
                input_channels=config["feature_extractor"].get("input_channels", 1),
                feature_size=config["feature_extractor"].get("feature_size", 256),
            ),
            config["network"],
            stage=config["network"].get("stage", "first")
        )
    else:
        raise NotImplementedError()


    # Check whether there is a weight file provided to continue training from
    if weight_file is not None:
        print("Loading weight file from {}".format(weight_file))
        # network.load_state_dict(
        #     torch.load(weight_file, map_location=device)
        # )
        # This is because train_ddp do not saved as model.state_dict()
        # import ipdb
        # ipdb.set_trace()
        model_dict = network.state_dict()
        network.load_state_dict({k.replace('module.', ''): v for k, v in
                               torch.load(weight_file, map_location=device).items() if k.replace('module.', '') in model_dict}, strict=False)
    network.to(device)
    return network, train_on_batch, validate_on_batch
