import math
import sys
from typing import Iterable, Optional

import torch
import torch.nn.functional as F

from timm.utils import ModelEma

import utils
import kaolin



def laplace_regularizer_const(mesh_verts, mesh_faces):
    term = torch.zeros_like(mesh_verts)
    norm = torch.zeros_like(mesh_verts[..., 0:1])

    v0 = mesh_verts[mesh_faces[:, 0], :]
    v1 = mesh_verts[mesh_faces[:, 1], :]
    v2 = mesh_verts[mesh_faces[:, 2], :]

    term.scatter_add_(0, mesh_faces[:, 0:1].repeat(1,3), (v1 - v0) + (v2 - v0))
    term.scatter_add_(0, mesh_faces[:, 1:2].repeat(1,3), (v0 - v1) + (v2 - v1))
    term.scatter_add_(0, mesh_faces[:, 2:3].repeat(1,3), (v0 - v2) + (v1 - v2))

    two = torch.ones_like(v0) * 2.0
    norm.scatter_add_(0, mesh_faces[:, 0:1], two)
    norm.scatter_add_(0, mesh_faces[:, 1:2], two)
    norm.scatter_add_(0, mesh_faces[:, 2:3], two)

    term = term / torch.clamp(norm, min=1.0)

    return torch.mean(term**2)


def train_batch(model, surface, images, depths, condinfo, criterion, smooth_weight = 1e-1, sigma_weight = 1e-2,):
    import ipdb

    sdf_reg_loss_entropy, v_list, f_list, z_e_x, z_q_x, sigma, loss_commit, perplexity, depth_out, antilias_mask, all_detach = model(surface, condinfo)
    # ipdb.set_trace()
    # loss_vol = criterion(outputs[:, :1024], labels[:, :1024])
    # loss_near = criterion(outputs[:, 1024:], labels[:, 1024:])
    # loss_point = kaolin.metrics.pointcloud.chamfer_distance(pred_points, points).mean()
    loss_alpha = criterion[0](antilias_mask, images[..., None])
    # mask = ~torch.isinf(depths[..., 0])
    # ipdb.set_trace()[images>0], [images>0]

    loss_depth = criterion[1](depth_out, depths[..., None])
    # loss_depth = criterion[1](depth_out, depths[..., None])
    # loss_depth = criterion[1](depth_out[mask], -depths[mask][..., :1])
    # is this a laplacian mesh smoothing loss?
    smooth_loss = 0
    for i in range(len(v_list)):
        smooth_loss += laplace_regularizer_const(v_list[i], f_list[i])
    smooth_loss = smooth_loss.mean() * smooth_weight
    loss_sigma = sigma.mean() * sigma_weight
    loss = loss_alpha + loss_depth + loss_commit + loss_sigma
    # loss = loss_alpha +  loss_depth + smooth_loss + loss_commit +  0.01 * loss_sigma
    # loss = loss_vol + 0.1 * loss_near + loss_commit + 0.0001 * loss_sigma

    return loss, v_list, f_list, loss_alpha.item(), loss_depth ,sdf_reg_loss_entropy.item(), loss_commit.item(), sigma.item(), smooth_loss.item(), all_detach, antilias_mask, depth_out


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    smooth_weight:float = 1e-1, sigma_weight:float = 1e-2, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, log_writer=None,
                    start_steps=None, lr_schedule_values=None, wd_schedule_values=None,
                    num_training_steps_per_epoch=None, update_freq=None):
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    if loss_scaler is None:
        model.zero_grad()
        model.micro_steps = 0
    else:
        optimizer.zero_grad()

    detach_nums =0

    # for data_iter_step, (points, labels, surface, _) in enumerate(
    #         metric_logger.log_every(data_loader, print_freq, header)):
    #     print(points.shape)
    log_step = torch.randint(0, 100, (1,))
    for data_iter_step, (surface, images, depths, condinfo, model_name) in enumerate(
            metric_logger.log_every(data_loader, print_freq, header)):
        step = data_iter_step // update_freq
        if step >= num_training_steps_per_epoch:
            continue
        it = start_steps + step  # global training iteration
        # Update LR & WD for the first acc
        if lr_schedule_values is not None or wd_schedule_values is not None and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        surface = surface.to(device, non_blocking=True)
        images = images.to(device, non_blocking=True)
        depths = depths.to(device, non_blocking= True)
        condinfo = condinfo.to(device, non_blocking = True)

        if loss_scaler is None:
            raise NotImplementedError
        else:
            with torch.cuda.amp.autocast(enabled=True):
                # loss, pred_points, loss_point.item(), sdf_reg_loss.item(), loss_commit.item(), loss_sigma.item()
                loss, v_list, f_list, loss_alpha, loss_depth, sdf_reg_loss_entropy, loss_commit, loss_sigma, smooth_loss, all_detach, antilias_mask, depth_out= \
                    train_batch(model, surface, images, depths, condinfo, criterion, smooth_weight, sigma_weight)

        if step == log_step:
            # import ipdb
            # ipdb.set_trace()
            depths_out = depth_out.flatten(1, 2).squeeze(-1).detach().cpu().numpy()
            mask_out = antilias_mask.flatten(1, 2).squeeze(-1).detach().cpu().numpy()
            log_name = model_name

        detach_nums += all_detach
        loss_value = loss.item()
        # if torch.isnan(loss):
        #     import ipdb
        #     ipdb.set_trace()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        if loss_scaler is None:
            raise NotImplementedError
        else:
            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss /= update_freq
            grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                    parameters=model.parameters(), create_graph=is_second_order,
                                    update_grad=(data_iter_step + 1) % update_freq == 0)
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)
            loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()


        # pred = torch.zeros_like(pred_points[:, :1024])
        # pred[pred_points[:, :1024] >= 0] = 1
        #
        # accuracy = (pred == labels[:, :1024]).float().sum(dim=1) / labels[:, :1024].shape[1]
        # accuracy = accuracy.mean()
        # intersection = (pred * labels[:, :1024]).sum(dim=1)
        # union = (pred + labels[:, :1024]).gt(0).sum(dim=1) + 1e-5
        # iou = intersection * 1.0 / union
        # iou = iou.mean()
        pred_points = []
        for (mesh_verts, mesh_faces) in zip(v_list, f_list):
            pred_points.append(kaolin.ops.mesh.sample_points(mesh_verts.unsqueeze(0), mesh_faces, 2048)[0][0])
                #.detach()

        pred_points = torch.stack(pred_points, dim =0).detach()
        # import ipdb
        # ipdb.set_trace()
        chamfer = kaolin.metrics.pointcloud.chamfer_distance(pred_points, surface).mean()
        metric_logger.update(loss=loss_value)
        metric_logger.update(chamfer_distance=chamfer)
        metric_logger.update(loss_scale=loss_scale_value)
        metric_logger.update(sdf_reg_loss=sdf_reg_loss_entropy)
        metric_logger.update(depth_loss=loss_depth)
        metric_logger.update(alpha_loss=loss_alpha)
        metric_logger.update(smooth_loss=smooth_loss)
        metric_logger.update(loss_commit=loss_commit)
        metric_logger.update(loss_sigma=loss_sigma)

        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(loss_scale=loss_scale_value, head="opt")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")
            log_writer.update(sdf_reg_loss=sdf_reg_loss_entropy, head="opt")
            log_writer.update(smooth_loss=smooth_loss, head="opt")

            log_writer.set_step()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    print("steps:", it,"Detatch num:", detach_nums)
    return_values = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    return_values['depth_log'] = depths_out
    return_values['mask_log'] = mask_out
    return_values['name_log'] = log_name
    return return_values


import ipdb


@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.BCEWithLogitsLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    model.eval()

    for batch in metric_logger.log_every(data_loader, 200, header):
        points, labels, surface, _ = batch
        surface = surface.to(device, non_blocking=True)
        points = points.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            N = 100000

            output, _, _, _, _, perplexity = model(surface, points)
            loss = criterion(output, labels)


        # pred = torch.zeros_like(output)
        # pred[output >= 0] = 1
        #
        # accuracy = (pred == labels).float().sum(dim=1) / labels.shape[1]
        # accuracy = accuracy.mean()
        # intersection = (pred * labels).sum(dim=1)
        # union = (pred + labels).gt(0).sum(dim=1)
        #
        # # if union.sum()==0:
        # # ipdb.set_trace()
        #
        # iou = intersection * 1.0 / union + 1e-5
        # iou = iou.mean()

        iou = torch.array([0])
        batch_size = surface.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['iou'].update(iou.item(), n=batch_size)

    metric_logger.synchronize_between_processes()

    print('* iou{iou.global_avg:.4f} loss {losses.global_avg:.3f}'
          .format(iou=metric_logger.iou, losses=metric_logger.loss))
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}