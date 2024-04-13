#
# Copyright (C) 2021 NVIDIA Corporation.  All rights reserved.
# Licensed under the NVIDIA Source Code License.
# See LICENSE at https://github.com/nv-tlabs/ATISS.
# Authors: Despoina Paschalidou, Amlan Kar, Maria Shugrina, Karsten Kreis,
#          Andreas Geiger, Sanja Fidler
#

"""Script used to train a ATISS."""
import argparse
import logging
import os
import sys

import shapely

os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"
import numpy as np
from timm.models import create_model
import torch
from torch.utils.data import DataLoader

from training_utils import id_generator, save_experiment_params, load_config

from scene_synthesis.datasets import get_encoded_dataset, filter_function
from scene_synthesis.networks import build_network, optimizer_factory
from scene_synthesis.stats_logger import StatsLogger, WandB
from scene_synthesis.networks.vqvae_model import vqvae_512_1024_2048 as vqvae_model
# from scene_synthesis.networks.vqvae_model import vqvae_256_1024_2048 as vqvae_model
# from scene_synthesis.networks.vqvae_model import vqvae_128_1024_2048 as vqvae_model
from scene_synthesis.networks.vqvae_model_texture import vqvae_512_1024_2048_tex as vqvae_model_tex, vqvae_512_1024_2048_cross
# import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import utils_dist

def yield_forever(iterator):
    while True:
        for x in iterator:
            yield x


def load_checkpoints(model, optimizer, experiment_directory, args, device):
    model_files = [
        f for f in os.listdir(experiment_directory)
        if f.startswith("model_")
    ]
    if len(model_files) == 0:
        return
    ids = [int(f[6:]) for f in model_files]
    max_id = max(ids)
    model_path = os.path.join(
        experiment_directory, "model_best"
    )#.format(max_id)
    opt_path = os.path.join(
        experiment_directory, "opt_best"
    )#.format(max_id)
    if not (os.path.exists(model_path) and os.path.exists(opt_path)):
        return

    print("Loading model checkpoint from {}".format(model_path))
    model.load_state_dict(torch.load(model_path, map_location=device))
    print("Loading optimizer checkpoint from {}".format(opt_path))
    optimizer.load_state_dict(
        torch.load(opt_path, map_location=device)
    )
    args.continue_from_epoch = max_id+1


def save_checkpoints(epoch, model, optimizer, experiment_directory):
    torch.save(
        model.state_dict(),
        os.path.join(experiment_directory, "model_best")
        # os.path.join(experiment_directory, "model_{:05d}").format(epoch)
    )
    torch.save(
        optimizer.state_dict(),
        os.path.join(experiment_directory, "opt_best")
        # os.path.join(experiment_directory, "opt_{:05d}").format(epoch)
    )


# def sort(centers_quantized, encodings):
#     ind3 = torch.argsort(centers_quantized[:, :, 2], dim=1)
#     centers_quantized = torch.gather(centers_quantized, 1, ind3[:, :, None].expand(-1, -1, centers_quantized.shape[-1]))
#     encodings = torch.gather(encodings, 1, ind3)
#
#     _, ind2 = torch.sort(centers_quantized[:, :, 1], dim=1, stable=True)
#     centers_quantized = torch.gather(centers_quantized, 1, ind2[:, :, None].expand(-1, -1, centers_quantized.shape[-1]))
#     encodings = torch.gather(encodings, 1, ind2)
#
#     _, ind1 = torch.sort(centers_quantized[:, :, 0], dim=1, stable=True)
#     centers_quantized = torch.gather(centers_quantized, 1, ind1[:, :, None].expand(-1, -1, centers_quantized.shape[-1]))
#     encodings = torch.gather(encodings, 1, ind1)
#     return centers_quantized, encodings

def sort(centers_quantized, encodings, centers):
    ind3 = torch.argsort(centers_quantized[:, :, 2], dim=1)
    centers_quantized = torch.gather(centers_quantized, 1, ind3[:, :, None].expand(-1, -1, centers_quantized.shape[-1]))
    centers = torch.gather(centers, 1, ind3[:, :, None].expand(-1, -1, centers_quantized.shape[-1]))
    encodings = torch.gather(encodings, 1, ind3)

    _, ind2 = torch.sort(centers_quantized[:, :, 1], dim=1, stable=True)
    centers_quantized = torch.gather(centers_quantized, 1, ind2[:, :, None].expand(-1, -1, centers_quantized.shape[-1]))
    centers = torch.gather(centers, 1, ind2[:, :, None].expand(-1, -1, centers_quantized.shape[-1]))
    encodings = torch.gather(encodings, 1, ind2)

    _, ind1 = torch.sort(centers_quantized[:, :, 0], dim=1, stable=True)
    centers_quantized = torch.gather(centers_quantized, 1, ind1[:, :, None].expand(-1, -1, centers_quantized.shape[-1]))
    centers = torch.gather(centers, 1, ind1[:, :, None].expand(-1, -1, centers_quantized.shape[-1]))
    encodings = torch.gather(encodings, 1, ind1)
    return centers_quantized, encodings, centers

import math
import trimesh
import mcubes

def main(argv):
    parser = argparse.ArgumentParser(
        description="Train a generative model on bounding boxes"
    )

    parser.add_argument(
        "config_file",
        help="Path to the file that contains the experiment configuration"
    )
    parser.add_argument(
        "output_directory",
        help="Path to the output directory"
    )
    parser.add_argument(
        "--weight_file",
        default=None,
        help=("The path to a previously trained model to continue"
              " the training from")
    )
    parser.add_argument(
        "--continue_from_epoch",
        default=0,
        type=int,
        help="Continue training from epoch (default=0)"
    )
    parser.add_argument(
        "--n_processes",
        type=int,
        default=4,
        help="The number of processed spawned by the batch provider"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=27,
        help="Seed for the PRNG"
    )
    parser.add_argument(
        "--experiment_tag",
        default=None,
        help="Tag that refers to the current experiment"
    )
    parser.add_argument(
        "--with_wandb_logger",
        action="store_true",
        help="Use wandB for logging the training progress"
    )
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')

    args = parser.parse_args(argv)

    utils_dist.init_distributed_mode(args)
    device = torch.device(args.device)

    # Disable trimesh's logger
    logging.getLogger("trimesh").setLevel(logging.ERROR)

    local_rank = args.local_rank
    torch.cuda.set_device(local_rank)
    # Set the random seed
    np.random.seed(args.seed)
    torch.manual_seed(np.random.randint(np.iinfo(np.int32).max))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(np.random.randint(np.iinfo(np.int32).max))


    # if torch.cuda.is_available():
    #     device = torch.device("cuda:0")
    # else:
    #     device = torch.device("cpu")
    # print("Running code on", device)

    # Check if output directory exists and if it doesn't create it
    if not os.path.exists(args.output_directory):
        os.makedirs(args.output_directory)

    # Create an experiment directory using the experiment_tag
    if args.experiment_tag is None:
        experiment_tag = id_generator(9)
    else:
        experiment_tag = args.experiment_tag

    experiment_directory = os.path.join(
        args.output_directory,
        experiment_tag
    )
    # if dist.get_rank() == 0:
    if not os.path.exists(experiment_directory):
        os.makedirs(experiment_directory)

    # import ipdb
    # ipdb.set_trace()
    # Save the parameters of this run to a file
    if utils_dist.get_rank() == 0:
        save_experiment_params(args, experiment_tag, experiment_directory)
        print("Save experiment statistics in {}".format(experiment_directory))

    # Parse the config file
    config = load_config(args.config_file)

    train_dataset = get_encoded_dataset(
        config["data"],
        filter_function(
            config["data"],
            split=config["training"].get("splits", ["train", "val"])
        ),
        path_to_bounds=None,
        augmentations=config["data"].get("augmentations", None),
        split=config["training"].get("splits", ["train", "val"])
    )
    # Compute the bounds for this experiment, save them to a file in the
    # experiment directory and pass them to the validation dataset
    # if dist.get_rank() == 0:
    path_to_bounds = os.path.join(experiment_directory, "bounds.npz")
    np.savez(
        path_to_bounds,
        sizes=train_dataset.bounds["sizes"],
        translations=train_dataset.bounds["translations"],
        angles=train_dataset.bounds["angles"]
    )
    print("Saved the dataset bounds in {}".format(path_to_bounds))

    validation_dataset = get_encoded_dataset(
        config["data"],
        filter_function(
            config["data"],
            split=config["validation"].get("splits", ["test"])
        ),
        path_to_bounds=path_to_bounds,
        augmentations=None,
        split=config["validation"].get("splits", ["test"])
    )

    # train_loader = DataLoader(
    #     train_dataset,
    #     batch_size=config["training"].get("batch_size", 128),
    #     num_workers=args.n_processes,
    #     collate_fn=train_dataset.collate_fn,
    #     shuffle=True
    # )
    num_tasks = utils_dist.get_world_size()
    global_rank = utils_dist.get_rank()

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,
                                                                    num_replicas=num_tasks,
                                                                    rank=global_rank, shuffle=True)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config["training"].get("batch_size", 128),
        collate_fn=train_dataset.collate_fn,
        num_workers=args.n_processes,
        sampler=train_sampler,
        drop_last = True,
    )
    if utils_dist.get_rank() == 0:
        print("Loaded {} training scenes with {} object types".format(
            len(train_dataset), train_dataset.n_object_types)
        )
        print("Training set has {} bounds".format(train_dataset.bounds))


    val_sampler = torch.utils.data.distributed.DistributedSampler(validation_dataset)

    val_loader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=config["validation"].get("batch_size", 1),
        collate_fn=validation_dataset.collate_fn,
        num_workers=args.n_processes,
        sampler=val_sampler)

    # val_loader = DataLoader(
    #     validation_dataset,
    #     batch_size=config["validation"].get("batch_size", 1),
    #     num_workers=args.n_processes,
    #     collate_fn=validation_dataset.collate_fn,
    #     shuffle=False
    # )
    print("Loaded {} validation scenes with {} object types".format(
        len(validation_dataset), validation_dataset.n_object_types)
    )
    print("Validation set has {} bounds".format(validation_dataset.bounds))

    # Make sure that the train_dataset and the validation_dataset have the same
    # number of object categories
    assert train_dataset.object_types == validation_dataset.object_types

    # Build the network architecture to be used for training
    network, train_on_batch, validate_on_batch = build_network(
        train_dataset.feature_size, train_dataset.n_classes,
        config, args.weight_file, device=device
    )
    # Build an optimizer object to compute the gradients of the parameters
    optimizer = optimizer_factory(config["training"], network.parameters())
    # Load the checkpoints if they exist in the experiment directory
    load_checkpoints(network, optimizer, experiment_directory, args, device)

    network = network.to(device)
    network = DDP(network, device_ids=[args.gpu], find_unused_parameters=False)


    # Initialize the logger
    if args.with_wandb_logger:
        WandB.instance().init(
            config,
            model=network,
            project=config["logger"].get(
                "project", "autoregressive_transformer"
            ),
            name=experiment_tag,
            watch=False,
            log_frequency=10
        )

    # Log the stats to a file
    StatsLogger.instance().add_output_file(open(
        os.path.join(experiment_directory, "stats.txt"),
        "w"
    ))
#
    epochs = config["training"].get("epochs", 150)
    steps_per_epoch = config["training"].get("steps_per_epoch", 500)
    save_every = config["training"].get("save_frequency", 100)
    val_every = config["validation"].get("frequency", 100)

    # num_points = config["network"].get("shape_dimensions")
    num_points = 512

    stage = config["data"].get("stage", "second")


    if stage=="second" and "shape" in config["network"].get("type"):
        vqvae = vqvae_model(pretrained=False).to(device)
        # vqvae = vqvae_512_1024_2048_cross(pretrained=False, **config["pretrain_params"]).to(device)
        pretraind_path = config["network"].get("pretrained_path")
        vqvae.load_state_dict(torch.load(pretraind_path, map_location='cpu')['model'], strict=True)


    # Do the training
    for i in range(args.continue_from_epoch, epochs):
        network.train()

        for b, sample in zip(range(steps_per_epoch), yield_forever(train_loader)):
            # Move everything to device
            if stage == "first":
                for k, v in sample.items():
                    sample[k] = v.to(device)

                batch_loss = train_on_batch(network, optimizer, sample, config)
                StatsLogger.instance().print_progress(i + 1, b + 1, batch_loss)
            else:
                sample["mask"] = 0
                for k, v in sample.items():
                    if k == "shapes":  # transform from surface to shape embeddings
                        surface = sample[k].to(device)
                        B, L, _, _ = surface.shape
                        length = sample['lengths']
                        ind = torch.arange(L).long().expand(B, L)
                        length = length.unsqueeze(1).expand_as(ind)
                        mask = ind < length
                        sample["mask"] = mask.to(device)
                        if mask.sum() == 0:
                            sample[k] = torch.zeros((B, 0, num_points, 4)).to(device)
                            # sample['texture'] = torch.zeros((B, 0, 384, 128, 32)).to(device)
                            continue
                        with torch.no_grad():
                            _, _, centers_quantized, _, _, encodings, centersd = vqvae.encode(surface[mask])

                        centers_quantized, encodings, centersd = sort(centers_quantized, encodings, centersd)
                        shape_embeddings = torch.cat((centersd, encodings[..., None]), dim=-1)

                        sample[k] = torch.zeros((B, L, *shape_embeddings.shape[1:]), dtype=torch.float32).to(
                            shape_embeddings.device)
                        sample[k][mask] = shape_embeddings  # vae is strong enough
                        # import ipdb
                        # ipdb.set_trace()
                    elif k == "shapes_tr":
                        surface = sample[k].to(device)
                        B, L, _, _ = surface.shape
                        surface = surface.reshape(-1, *surface.shape[2:])
                        with torch.no_grad():
                            _, _, centers_quantized, _, _, encodings, centersd = vqvae.encode(surface)
                        centers_quantized, encodings, centersd = sort(centers_quantized, encodings, centersd)
                        shape_embeddings = torch.cat((centersd, encodings[..., None]), dim=-1)
                        # ipdb.set_trace()
                        shape_embeddings = shape_embeddings.reshape(B, L, *shape_embeddings.shape[1:])
                        sample[k] = shape_embeddings  # vae is strong enough
                    else:
                        sample[k] = v.to(device)
                batch_loss = train_on_batch(network, optimizer, sample, config)
                StatsLogger.instance().print_progress(i + 1, b + 1, batch_loss)

        if utils_dist.get_rank() == 0 and (i % save_every) == 0:
            save_checkpoints(
                i,
                network,
                optimizer,
                experiment_directory,
            )
        StatsLogger.instance().clear()

        if val_every and i % val_every == 0 and i > 0:
            print("====> Validation Epoch ====>")
            network.eval()
            for b, sample in enumerate(val_loader):
                # Move everything to device
                for k, v in sample.items():
                    if k == "shapes":  # transform from surface to shape embeddings
                        surface = sample[k].to(device)
                        B, L, _, _ = surface.shape
                        if L > 6:
                            centers_quantized, encodings, centersd = [], [], []
                            with torch.no_grad():
                                for spp in range(B):
                                    _, _, centers, _, _, encoding, centerd = vqvae.encode(surface[spp])
                                    centers_quantized.append(centers)
                                    encodings.append(encoding)
                                    centersd.append(centerd)
                                centers_quantized = torch.cat(centers_quantized, dim=0)
                                encodings = torch.cat(encodings, dim=0)
                                centersd = torch.cat(centersd, dim=0)
                        else:
                            surface = surface.reshape(-1, *surface.shape[2:])
                            try:
                                with torch.no_grad():
                                    _, _, centers_quantized, _, _, encodings, centersd = vqvae.encode(surface)
                            except:
                                try:
                                    assert surface.shape[0] == 0  # for the condition (b, 0, 512, 4)
                                except:  # this means cuda out of memory.
                                    import ipdb
                                    ipdb.set_trace()
                                sample[k] = torch.zeros((B, 0, num_points, 4)).to(device)
                                continue
                        # import ipdb
                        # ipdb.set_trace()
                        centers_quantized, encodings, centersd = sort(centers_quantized, encodings, centersd)
                        shape_embeddings = torch.cat((centersd, encodings[..., None]), dim=-1)
                        shape_embeddings = shape_embeddings.reshape(B, L, *shape_embeddings.shape[1:])
                        sample[k] = shape_embeddings  # vae is strong enough
                    elif k == "shapes_tr":
                        surface = sample[k].to(device)
                        B, L, _, _ = surface.shape
                        surface = surface.reshape(-1, *surface.shape[2:])
                        # import ipdb
                        try:
                            with torch.no_grad():
                                _, _, centers_quantized, _, _, encodings, centerd = vqvae.encode(surface)
                        except:
                            assert surface.shape[0] == 0
                        centers_quantized, encodings, centers = sort(centers_quantized, encodings, centerd)
                        shape_embeddings = torch.cat((centers, encodings[..., None]), dim=-1)
                        # ipdb.set_trace()
                        shape_embeddings = shape_embeddings.reshape(B, L, *shape_embeddings.shape[1:])
                        sample[k] = shape_embeddings  # vae is strong enough
                    else:
                        sample[k] = v.to(device)
                batch_loss = validate_on_batch(network, sample, config)
                if utils_dist.get_rank() == 0:
                    StatsLogger.instance().print_progress(-1, b+1, batch_loss)
            if utils_dist.get_rank() == 0:
                StatsLogger.instance().clear()
            print("====> Validation Epoch ====>")


if __name__ == "__main__":
    main(sys.argv[1:])