# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import os
import sys
import datetime
import time
import math
import json
import struct
import pickle
from pathlib import Path
import redis

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision import models as torchvision_models
from torch.utils.data.dataset import Dataset

import utils
import vision_transformer4k as vits
from vision_transformer4k import DINOHead
from tqdm import tqdm
import faiss

from einops import rearrange, repeat, reduce

from losses import DINOLoss
from utils import run_kmeans, suppress_stdout_stderr
from tqdm import tqdm

from torch.multiprocessing import Pool
from itertools import permutations

torchvision_archs = sorted(name for name in torchvision_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(torchvision_models.__dict__[name]))

def get_args_parser():
    parser = argparse.ArgumentParser('DINO4K', add_help=False)

    # Model parameters
    parser.add_argument('--arch', default='vit4k_xs', type=str,
        choices=['vit4k_xs', 'vit_tiny', 'vit_small', 'vit_base', 'xcit', 'deit_tiny', 'deit_small'] \
                + torchvision_archs + torch.hub.list("facebookresearch/xcit:main"),
        help="""Name of architecture to train. For quick experiments with ViTs,
        we recommend using vit_tiny or vit_small.""")
    parser.add_argument('--patch_size', default=16, type=int, help="""Size in pixels
        of input square patches - default 16 (for 16x16 patches). Using smaller
        values leads to better performance but requires more memory. Applies only
        for ViTs (vit_tiny, vit_small and vit_base). If <16, we recommend disabling
        mixed precision training (--use_fp16 false) to avoid unstabilities.""")
    parser.add_argument('--out_dim', default=65536, type=int, help="""Dimensionality of
        the DINO head output. For complex and large datasets large values (like 65k) work well.""")
    parser.add_argument('--norm_last_layer', default=True, type=utils.bool_flag,
        help="""Whether or not to weight normalize the last layer of the DINO head.
        Not normalizing leads to better performance but can make the training unstable.
        In our experiments, we typically set this paramater to False with vit_small and True with vit_base.""")
    parser.add_argument('--momentum_teacher', default=0.996, type=float, help="""Base EMA
        parameter for teacher update. The value is increased to 1 during training with cosine schedule.
        We recommend setting a higher value with small batches: for example use 0.9995 with batch size of 256.""")
    parser.add_argument('--use_bn_in_head', default=False, type=utils.bool_flag,
        help="Whether to use batch normalizations in projection head (Default: False)")

    # Temperature teacher parameters
    parser.add_argument('--warmup_teacher_temp', default=0.04, type=float,
        help="""Initial value for the teacher temperature: 0.04 works well in most cases.
        Try decreasing it if the training loss does not decrease.""")
    parser.add_argument('--teacher_temp', default=0.04, type=float, help="""Final value (after linear warmup)
        of the teacher temperature. For most experiments, anything above 0.07 is unstable. We recommend
        starting with the default value of 0.04 and increase this slightly if needed.""")
    parser.add_argument('--warmup_teacher_temp_epochs', default=0, type=int,
        help='Number of warmup epochs for the teacher temperature (Default: 30).')

    # Training/Optimization parameters
    parser.add_argument('--use_fp16', type=utils.bool_flag, default=True, help="""Whether or not
        to use half precision for training. Improves training time and memory requirements,
        but can provoke instability and slight decay of performance. We recommend disabling
        mixed precision if the loss is unstable, if reducing the patch size or if training with bigger ViTs.""")
    parser.add_argument('--weight_decay', type=float, default=0.04, help="""Initial value of the
        weight decay. With ViT, a smaller value at the beginning of training works well.""")
    parser.add_argument('--weight_decay_end', type=float, default=0.4, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")
    parser.add_argument('--clip_grad', type=float, default=3.0, help="""Maximal parameter
        gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can
        help optimization for larger ViT architectures. 0 for disabling.""")
    parser.add_argument('--batch_size_per_gpu', default=128, type=int,
        help='Per-GPU batch-size : number of distinct images loaded on one GPU.')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs of training.')
    parser.add_argument('--freeze_last_layer', default=1, type=int, help="""Number of epochs
        during which we keep the output layer fixed. Typically doing so during
        the first epoch helps training. Try increasing this value if the loss does not decrease.""")
    parser.add_argument("--lr", default=0.0005, type=float, help="""Learning rate at the end of
        linear warmup (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.""")
    parser.add_argument("--warmup_epochs", default=10, type=int,
        help="Number of epochs for the linear learning-rate warm up.")
    parser.add_argument('--min_lr', type=float, default=1e-6, help="""Target LR at the
        end of optimization. We use a cosine LR schedule with linear warmup.""")
    parser.add_argument('--optimizer', default='adamw', type=str,
        choices=['adamw', 'sgd', 'lars'], help="""Type of optimizer. We recommend using adamw with ViTs.""")
    parser.add_argument('--drop_path_rate', type=float, default=0.1, help="stochastic depth rate")

    # Multi-crop parameters
    parser.add_argument('--global_crops_scale', type=float, nargs='+', default=(0.4, 1.),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for large global view cropping. When disabling multi-crop (--local_crops_number 0), we
        recommand using a wider range of scale ("--global_crops_scale 0.14 1." for example)""")
    parser.add_argument('--local_crops_number', type=int, default=8, help="""Number of small
        local views to generate. Set this parameter to 0 to disable multi-crop training.
        When disabling multi-crop we recommend to use "--global_crops_scale 0.14 1." """)
    parser.add_argument('--local_crops_scale', type=float, nargs='+', default=(0.05, 0.4),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for small local view cropping of multi-crop.""")

    # Misc
    parser.add_argument('--data_path', default='/path/to/imagenet/train/', type=str,
        help='Please specify path to the ImageNet training data.')
    parser.add_argument('--output_dir', default="./results", type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--saveckp_freq', default=20, type=int, help='Save checkpoint every x epochs.')
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    
    # Dataset
    parser.add_argument('--dataset_dir', default="/remote-home/share/Carboxy/Code/HIPT/Hierarchical-Pretraining/extract_features/tcga_lung_4k.json", type=str, help='Path to dataset json file.')
    parser.add_argument("--num_cluster", default=4, type=int)
    parser.add_argument("--min_num_region", default=32, type=int)
    parser.add_argument("--loss_type", default='intra_inter', type=str) # intra, inter, intra_inter
    parser.add_argument("--intra_num_crops", default=10, type=int)
    parser.add_argument("--intra_loss_weight", default=1.0, type=float)
    parser.add_argument("--inter_loss_weight", default=1.0, type=float)
    parser.add_argument('--log_data', action='store_true', default=True, help='log data using tensorboard')
    parser.add_argument('--debug_mode', action='store_true', default=False, help='log data using tensorboard')
    parser.add_argument("--warm_epoch", default=20, type=int)
    parser.add_argument('--extra_mlp', action='store_true', default=False, help='Use projection head for clustering')
    parser.add_argument('--temperature_contrastive', default=0.1, help='temperature for PCL', type=float)
    parser.add_argument('--use_density', action='store_true', default=False, help='Use density')
    parser.add_argument('--which_feat_to_cluster_stu', default='mlp', type=str)
    parser.add_argument('--which_feat_to_cluster_tea', default='mlp', type=str)
    parser.add_argument('--neighbor_num', default=1, type=int)
    return parser


def train_dino(args):
    utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    # ============ preparing data ... ============
    transform = DataAugmentationDINO4K(
        args.local_crops_number,
        ignore_reshape=True
    )

    # center-crop augmentation 
    transform_eval = transforms.Compose([
            transforms.CenterCrop(14),
            ])  
    
    # Using custom dataset for our [256 x 384] tensors
    # dataset = SeqDataset(dataroot=args.data_path, transform=transform)
    # dataset = SeqDataset(dataroot=args.data_path, transform=transform)
    database = redis.Redis(host='localhost', port=6379)
    dataset = RegionDataset(dataset_dir=args.dataset_dir, transform=transform, database=database, debug_mode=args.debug_mode)
    dataset_eval = RegionDataset(dataset_dir=args.dataset_dir, transform=transform_eval, database=database, debug_mode=args.debug_mode)
    
    sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)
    sampler_eval = torch.utils.data.DistributedSampler(dataset_eval, shuffle=False)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    # dataloader for center-cropped tensors, use larger batch size to increase speed
    data_loader_eval = torch.utils.data.DataLoader(
        dataset_eval,
        sampler=sampler_eval,
        batch_size=args.batch_size_per_gpu*5,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    print(f"Data loaded: there are {len(dataset)} images.")

    # ============ building student and teacher networks ... ============
    # we changed the name DeiT-S for ViT-S to avoid confusions
    args.arch = args.arch.replace("deit", "vit")
    # if the network is a Vision Transformer (i.e. vit_tiny, vit_small, vit_base)
    if args.arch in vits.__dict__.keys():
        student = vits.__dict__[args.arch](
            patch_size=args.patch_size,
            drop_path_rate=args.drop_path_rate,  # stochastic depth
        )
        teacher = vits.__dict__[args.arch](patch_size=args.patch_size)
        embed_dim = student.embed_dim
    # if the network is a XCiT
    elif args.arch in torch.hub.list("facebookresearch/xcit:main"):
        student = torch.hub.load('facebookresearch/xcit:main', args.arch,
                                 pretrained=False, drop_path_rate=args.drop_path_rate)
        teacher = torch.hub.load('facebookresearch/xcit:main', args.arch, pretrained=False)
        embed_dim = student.embed_dim
    # otherwise, we check if the architecture is in torchvision models
    elif args.arch in torchvision_models.__dict__.keys():
        student = torchvision_models.__dict__[args.arch]()
        teacher = torchvision_models.__dict__[args.arch]()
        embed_dim = student.fc.weight.shape[1]
    else:
        print(f"Unknow architecture: {args.arch}")

    # multi-crop wrapper handles forward with inputs of different resolutions
    student = utils.MultiCropWrapper(student, DINOHead(
        embed_dim,
        args.out_dim,
        use_bn=args.use_bn_in_head,
        norm_last_layer=args.norm_last_layer,
        extra_mlp=args.extra_mlp
    ), args.which_feat_to_cluster_stu)
    teacher = utils.MultiCropWrapper(
        teacher,
        DINOHead(embed_dim, args.out_dim, args.use_bn_in_head, extra_mlp=args.extra_mlp),
        args.which_feat_to_cluster_tea
    )
    # move networks to gpu
    student, teacher = student.cuda(), teacher.cuda()
    # synchronize batch norms (if any)
    if utils.has_batchnorms(student):
        student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
        teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)

        # we need DDP wrapper to have synchro batch norms working...
        teacher = nn.parallel.DistributedDataParallel(teacher, device_ids=[args.gpu])
        teacher_without_ddp = teacher.module
    else:
        # teacher_without_ddp and teacher are the same thing
        teacher_without_ddp = teacher
    # student = nn.parallel.DistributedDataParallel(student, device_ids=[args.gpu], find_unused_parameters=True)
    student = nn.parallel.DistributedDataParallel(student, device_ids=[args.gpu], find_unused_parameters=False)
    # teacher and student start with the same weights
    teacher_without_ddp.load_state_dict(student.module.state_dict())
    # there is no backpropagation through the teacher, so no need for gradients
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"Student and Teacher are built: they are both {args.arch} network.")

    # ============ preparing loss ... ============
    dino_loss = DINOLoss(
        args.out_dim,
        args.local_crops_number + 2,  # total number of crops = 2 global crops + local_crops_number
        args.warmup_teacher_temp,
        args.teacher_temp,
        args.warmup_teacher_temp_epochs,
        args.epochs,
    ).cuda()

    assert args.loss_type in ['intra', 'inter', 'intra_inter']
    if args.loss_type=='intra':
        ncrops_tea = 1
    elif args.loss_type=='inter':
        ncrops_tea = args.neighbor_num
    elif args.loss_type=='intra_inter':
        ncrops_tea = 1 + args.neighbor_num
    dino_loss_proto = DINOLoss(
        args.out_dim,
        args.intra_num_crops,  # total number of crops = 2 global crops + local_crops_number
        args.warmup_teacher_temp,
        args.teacher_temp,
        args.warmup_teacher_temp_epochs,
        args.epochs,
        ncrops_tea=ncrops_tea,
    ).cuda()

    losses = {'dino_loss': None, 'dino_loss_proto': None}
    losses.update({'dino_loss': dino_loss})
    losses.update({'dino_loss_proto': dino_loss_proto})

    # ============ preparing Tensorboard ... ============
    if args.log_data:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(args.output_dir, flush_secs=15)
    else:
        writer = None

    # ============ save args ... ============
    args_dict = args.__dict__
    with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
        json.dump(args_dict, f, indent=2)
    for key, val in args.__dict__.items():
        if type(val) in [int, float]:
            writer.add_scalar('train_args/'+key, val)

    # ============ preparing optimizer ... ============
    params_groups = utils.get_params_groups(student)
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params_groups)  # to use with ViTs
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params_groups, lr=0, momentum=0.9)  # lr is set by scheduler
    elif args.optimizer == "lars":
        optimizer = utils.LARS(params_groups)  # to use with convnet and large batches
    # for mixed precision training
    fp16_scaler = None
    if args.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    # ============ init schedulers ... ============
    lr_schedule = utils.cosine_scheduler(
        args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256.,  # linear scaling rule
        args.min_lr,
        args.epochs, len(data_loader),
        warmup_epochs=args.warmup_epochs,
    )
    wd_schedule = utils.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs, len(data_loader),
    )
    # momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = utils.cosine_scheduler(args.momentum_teacher, 1,
                                               args.epochs, len(data_loader))
    print(f"Loss, optimizer and schedulers ready.")

    # ============ optionally resume training ... ============
    to_restore = {"epoch": 0}
    utils.restart_from_checkpoint(
        os.path.join(args.output_dir, "checkpoint.pth"),
        run_variables=to_restore,
        student=student,
        teacher=teacher,
        optimizer=optimizer,
        fp16_scaler=fp16_scaler,
        dino_loss=dino_loss,
    )
    start_epoch = to_restore["epoch"]

    start_time = time.time()
    print("Starting DINO training !")
    for epoch in range(start_epoch, args.epochs):
        data_loader.sampler.set_epoch(epoch)
        data_loader_eval.sampler.set_epoch(epoch)

        # ============ training one epoch of DINO ... ============
        train_stats = train_one_epoch(student, teacher, teacher_without_ddp, losses,
            data_loader, data_loader_eval, optimizer, lr_schedule, wd_schedule, momentum_schedule,
            epoch, fp16_scaler, args, writer)

        # ============ writing logs ... ============
        save_dict = {
            'student': student.state_dict(),
            'teacher': teacher.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'args': args,
            'dino_loss': dino_loss.state_dict(),
        }
        if fp16_scaler is not None:
            save_dict['fp16_scaler'] = fp16_scaler.state_dict()
        utils.save_on_master(save_dict, os.path.join(args.output_dir, 'checkpoint.pth'))
        if args.saveckp_freq and epoch % args.saveckp_freq == 0:
            utils.save_on_master(save_dict, os.path.join(args.output_dir, f'checkpoint{epoch:04}.pth'))
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}
        if utils.is_main_process():
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def train_one_epoch(student, teacher, teacher_without_ddp, losses, data_loader, data_loader_eval,
                    optimizer, lr_schedule, wd_schedule, momentum_schedule, epoch,
                    fp16_scaler, args, writer):
    if epoch > args.warm_epoch:
        # Cluster each WSI
        feat_dim_dict = {'mlp': 256, 'backbone': 192}
        feat_dim = feat_dim_dict[args.which_feat_to_cluster_tea]
        region_features_mlp = torch.zeros(len(data_loader_eval.dataset), 256).cuda()
        region_features_cluster = torch.zeros(len(data_loader_eval.dataset), feat_dim).cuda()
        slide_inds = torch.zeros(len(data_loader_eval.dataset), dtype=torch.long).cuda()
        region_inds = torch.zeros(len(data_loader_eval.dataset), dtype=torch.long).cuda()
        repeat_inds = torch.zeros(len(data_loader_eval.dataset), dtype=torch.long).cuda()
        teacher.eval()
        for images_batch, _, slide_inds_batch, region_inds_batch, inds_batch in tqdm(data_loader_eval):
            images_batch = images_batch.cuda(non_blocking=True)
            slide_inds_batch = slide_inds_batch.cuda(non_blocking=True)
            region_inds_batch = region_inds_batch.cuda(non_blocking=True)
            inds_batch = inds_batch.cuda(non_blocking=True)
            with torch.no_grad():
                # teacher_output = teacher.forward_backbone(images_batch)  # only the 2 global views pass through the teacher
                _, teacher_output_cluster, teacher_output_mlp = teacher(images_batch) # only the 2 global views pass through the teacher
            region_features_mlp[inds_batch, :] = teacher_output_mlp
            region_features_cluster[inds_batch, :] = teacher_output_cluster
            slide_inds[inds_batch] = slide_inds_batch
            region_inds[inds_batch] = region_inds_batch
            repeat_inds[inds_batch] = 1

        dist.barrier()        
        dist.all_reduce(region_features_mlp, op=dist.ReduceOp.SUM)
        dist.all_reduce(region_features_cluster, op=dist.ReduceOp.SUM)
        dist.all_reduce(slide_inds, op=dist.ReduceOp.SUM)
        dist.all_reduce(region_inds, op=dist.ReduceOp.SUM)  
        dist.all_reduce(repeat_inds, op=dist.ReduceOp.SUM) 

        # account for the few samples that are computed more than once  
        region_features_mlp = region_features_mlp / repeat_inds[:, None]
        region_features_cluster = region_features_cluster / repeat_inds[:, None]
        slide_inds = slide_inds / repeat_inds
        region_inds = region_inds / repeat_inds 

        # normalize 
        region_features_mlp = F.normalize(region_features_mlp, dim=-1, p=2)
        region_features_cluster = F.normalize(region_features_cluster, dim=-1, p=2)

        # clustering results
        slide_inds_unique, slide_inds_cnt = slide_inds.unique(return_counts=True)
        max_num_cluster = args.num_cluster
        centroids = torch.zeros(len(slide_inds_unique), max_num_cluster, feat_dim).cuda() # [N, K, 192]
        density = torch.zeros(len(slide_inds_unique), max_num_cluster).cuda() # [N, K]
        im2cluster = torch.zeros(len(slide_inds_unique), slide_inds_cnt.max(), dtype=torch.long).cuda() # [N, L]
        repeat_inds = torch.zeros(len(slide_inds_unique), dtype=torch.long).cuda()
        
        # split tensors by gpu index
        num_gpu = torch.cuda.device_count()
        slide_inds_unique, slide_inds_cnt = slide_inds.unique(return_counts=True)
        slide_inds_cnt_cumsum = torch.cumsum(slide_inds_cnt, dim=0)
        region_features_split = region_features_cluster.split(slide_inds_cnt.tolist())
        slide_inds_split = slide_inds_unique.long()
        gpu_inds_split = torch.fmod(slide_inds_unique, num_gpu).long()
        gpu_rank = dist.get_rank()
        # print(gpu_rank)

        # dist k-means
        num_regions = slide_inds_cnt
        ignore_cluster = num_regions < args.min_num_region
        slide_inds_this_gpu = slide_inds_split[gpu_inds_split==gpu_rank]
        for slide_ind in tqdm(slide_inds_this_gpu, desc="GPU ID={}".format(gpu_rank)):
            slide_ind = int(slide_ind)
            num_region_this = num_regions[slide_ind]
            repeat_inds[slide_ind] = 1
            if num_region_this < args.min_num_region:
                continue
            with suppress_stdout_stderr(): # normalize=False
                centroids_this, density_this, im2cluster_this = \
                    run_kmeans(x=region_features_split[slide_ind].cpu().numpy(), num_cluster=args.num_cluster, temperature=args.temperature_contrastive, gpu_ind=gpu_rank, use_density=args.use_density, normalize=True)
            centroids[slide_ind] = centroids_this
            density[slide_ind] = density_this
            im2cluster[slide_ind, :num_region_this] = im2cluster_this

        dist.barrier()     
        dist.all_reduce(centroids, op=dist.ReduceOp.SUM)
        dist.all_reduce(density, op=dist.ReduceOp.SUM)
        dist.all_reduce(im2cluster, op=dist.ReduceOp.SUM)  
        dist.all_reduce(repeat_inds, op=dist.ReduceOp.SUM) 
        dist.barrier()   
        # assert repeat_inds.min() > 0
        # account for the few samples that are computed more than once  
        centroids = centroids / repeat_inds[:, None, None]
        density = density / repeat_inds[:, None]
        im2cluster = im2cluster / repeat_inds[:, None]
        im2cluster = im2cluster.long()
        if dist.get_rank() == 0:
            cluster_result = {'im2cluster':im2cluster,'centroids':centroids,'density':density}
            try:
                torch.save(cluster_result, os.path.join(args.output_dir, 'clusters_%d'%epoch))  
            except:
                pass   
        dist.barrier()    

        # find neighbors of slide
        perms = list(permutations([i for i in range(args.num_cluster)]))
        N = len(slide_inds_unique)
        sim_perms = torch.zeros(N, N, len(perms))
        for ii, perm in enumerate(perms):
            centroids_perm = centroids[:, perm, :]
            sim_perms[:, :, ii] = torch.einsum('nkd,mkd->nm', centroids, centroids_perm)
        matching_sim, _ = torch.max(sim_perms, dim=-1, keepdim=False) # [N, N]
        neighbor_num = args.neighbor_num # r
        matching_sim[matching_sim==0] -= 4
        _, neighbor_slide_inds = torch.topk(matching_sim, neighbor_num+1, dim=-1, largest=True, sorted=True)
        neighbor_slide_inds = neighbor_slide_inds[:, 1:] # remove self, [N, r]

        # construct indices of neighbor slides and neighbor prototypes
        neighbor_slide_inds= neighbor_slide_inds.long().cuda() # [N, K, r]
        neighbor_proto_inds = torch.zeros(N, args.num_cluster, neighbor_num).long().cuda() # [N, K, r]
        for ii in range(neighbor_num):
            neighbors_this_r = neighbor_slide_inds[:, ii]
            target = centroids[neighbors_this_r, :, :] # [N, K, D]
            sim = torch.einsum('nkd,nld->nkl', centroids, target)
            neighbor_proto_inds[:, :, ii] = torch.argmax(sim, dim=-1)
        cluster_result = {'im2cluster':[im2cluster],'centroids':[centroids],\
                          'density':[density], 'neighbor_proto_inds': [neighbor_proto_inds]}

    teacher.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
    for it, (images, _, slide_inds, region_inds, inds) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        # update weight decay and learning rate according to their schedule
        it = len(data_loader) * epoch + it  # global training iteration
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[it]

        # move images to gpu
        slide_inds = slide_inds.cuda()
        region_inds = region_inds.cuda()
        images = [im.cuda(non_blocking=True) for im in images]
        num_crops = args.intra_num_crops
        # teacher and student forward passes + compute dino loss
        with torch.cuda.amp.autocast(fp16_scaler is not None):
            teacher_output, teacher_output_cluster, teacher_output_mlp = teacher(images[:2])  # only the 2 global views pass through the teacher
            student_output, student_output_cluster, student_output_mlp = student(images)
            # DINO Loss
            dino_loss = losses['dino_loss'](student_output, teacher_output, epoch)
            
            dino_loss_intra = dino_loss * 0.0
            dino_loss_inter = dino_loss * 0.0

            if epoch>args.warm_epoch:
                # student_output_proto = student.module.forward_head(student_output_backbone)
                dino_loss_intra = torch.Tensor([0.0]).cuda()
                dino_loss_inter = torch.Tensor([0.0]).cuda()
                n_loss_terms = 0
                for n, (im2cluster,prototypes,neighbor_proto_inds) in enumerate(zip(cluster_result['im2cluster'],cluster_result['centroids'],cluster_result['neighbor_proto_inds'])):

                    # ignore images without clusterings
                    ignore_inds = ignore_cluster[slide_inds] # [M*B], True means ignore, False means keep
                    slide_inds = slide_inds[~ignore_inds] # [M*B']
                    region_inds = region_inds[~ignore_inds] # [M*B']
                    student_output = student_output[~(ignore_inds.repeat(args.intra_num_crops))]
                    
                    # intra
                    if args.loss_type in ['intra', 'intra_inter']:
                        pos_intra_proto_inds = im2cluster[slide_inds, region_inds] # [M*B']
                        pos_intra_prototypes = prototypes[slide_inds, pos_intra_proto_inds, :] # [M*B', 192]
                        if args.use_bn_in_head:
                            teacher_output_intra_proto = teacher.module.forward_last_layer(pos_intra_prototypes)
                        else:
                            teacher_output_intra_proto = teacher.forward_last_layer(pos_intra_prototypes)
                        dino_loss_intra += losses['dino_loss_proto'](student_output, teacher_output_intra_proto, epoch, ncrops_tea=1)

                    # inter
                    if args.loss_type in ['inter', 'intra_inter']:
                        pos_intra_proto_inds = im2cluster[slide_inds, region_inds] # [M*B']
                        pos_prototypes = []
                        for r in range(args.neighbor_num):
                            pos_inter_slide_inds = neighbor_slide_inds[slide_inds, r]
                            pos_inter_proto_inds = neighbor_proto_inds[slide_inds, pos_intra_proto_inds, r]
                            # pos_inter_prototypes = prototypes[slide_inds, pos_inter_proto_inds, :] # [M*B', 192]
                            pos_inter_prototypes = prototypes[pos_inter_slide_inds, pos_inter_proto_inds, :] # [M*B', 192]
                            pos_prototypes.append(pos_inter_prototypes)
                        pos_prototypes = torch.cat(pos_prototypes, dim=0)
                        if args.use_bn_in_head:
                            teacher_output_inter_proto = teacher.module.forward_last_layer(pos_prototypes)
                        else:
                            teacher_output_inter_proto = teacher.forward_last_layer(pos_prototypes)
                        dino_loss_inter += losses['dino_loss_proto'](student_output, teacher_output_inter_proto, epoch, ncrops_tea=args.neighbor_num)
                    
                    n_loss_terms += 1

                dino_loss_intra /= n_loss_terms
                dino_loss_inter /= n_loss_terms

            loss = dino_loss + args.intra_loss_weight * dino_loss_intra + args.inter_loss_weight * dino_loss_inter
            print('dino_loss: {:.4f}, intra_loss: {:.4f}, inter_loss: {:.4f}, loss: {:.4f}'\
                  .format(dino_loss.item(), dino_loss_intra.item(), dino_loss_inter.item(), loss.item()))

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), force=True)
            sys.exit(1)

        # student update
        optimizer.zero_grad()
        param_norms = None
        if fp16_scaler is None:
            loss.backward()
            if args.clip_grad:
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            if args.clip_grad:
                fp16_scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

        # EMA update for the teacher
        with torch.no_grad():
            m = momentum_schedule[it]  # momentum parameter
            for param_q, param_k in zip(student.module.parameters(), teacher_without_ddp.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        # logging
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(dino_loss=dino_loss.item())
        metric_logger.update(intra_loss=dino_loss_intra.item())
        metric_logger.update(inter_loss=dino_loss_inter.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])

        if writer:
            writer.add_scalar('train/loss', loss.item(), it)
            writer.add_scalar('train/dino_loss', dino_loss.item(), it)
            writer.add_scalar('train/intra_loss', dino_loss_intra.item(), it)
            writer.add_scalar('train/inter_loss', dino_loss_inter.item(), it)
            writer.add_scalar('train/lr', optimizer.param_groups[0]["lr"], it)
            writer.add_scalar('train/wd', optimizer.param_groups[0]["weight_decay"], it)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

### Custom Dataset Implemented to Load in [256-Length x 384-Dim] Tensors which correspond to extracted ViT-16 features for 4K x 4K patch
class SeqDataset(Dataset):
    def __init__(self, dataroot, transform):
        seq_list = os.listdir(dataroot)
        self.seq_list = [os.path.join(dataroot, fname) for fname in seq_list]
        self.transform = transform
        
    def __getitem__(self, index):
        seq = torch.load(self.seq_list[index]) # [256, 384]
        label = torch.zeros(1,1)
        return self.transform(seq), label

    def __len__(self):
        return len(self.seq_list)

class RegionDataset(Dataset):
    def __init__(self, dataset_dir, transform, database, debug_mode=False):
        with open(dataset_dir, encoding='utf8') as f:
            self.dataset = json.load(f)
        if debug_mode:
            self.dataset_tmp = {}
            for ii in range(20000):
                self.dataset_tmp[str(ii)] = self.dataset[str(ii)]
            self.dataset = self.dataset_tmp
        self.transform = transform
        self.database = database

    def __getitem__(self, index):
        database_ind = self.dataset[str(index)]['dataset_index']
        bytes_data =  self.database.get(database_ind)
        tensor_data = torch.from_numpy(pickle.loads(bytes_data)) # [384, 16, 16]
        label = torch.zeros(1, 1)
        slide_id = self.dataset[str(index)]['slide_id']
        slide_index = self.dataset[str(index)]['slide_index']
        region_index = self.dataset[str(index)]['region_index']
        return self.transform(tensor_data), label, slide_index, region_index, index

    def __len__(self):
        return len(self.dataset)

### Modified Data Augmentaton for DINO for 4K x 4K resolutions for performing local / global crops on features in image grid
class DataAugmentationDINO4K(object):
    def __init__(self, local_crops_number, ignore_reshape=False):
        flip = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
        ])

        # first global crop
        self.global_transfo1 = transforms.Compose([
            transforms.RandomCrop(14),
            transforms.RandomHorizontalFlip(p=0.5),
        ])
        
        # second global crop
        self.global_transfo2 = transforms.Compose([
            transforms.RandomCrop(14),
            transforms.RandomHorizontalFlip(p=0.5),
        ])
        
        # transformation for the local small crops
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose([
            transforms.RandomCrop(6),
            transforms.RandomHorizontalFlip(p=0.5),
        ])
        self.ignore_reshape = ignore_reshape

    def __call__(self, image):
        crops = []
        if not self.ignore_reshape:
            image = image.unfold(0, 16, 16).transpose(0,1)
        crops.append(self.global_transfo1(image))
        crops.append(self.global_transfo2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        return crops

if __name__ == '__main__':
    parser = argparse.ArgumentParser('DINO4K', parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train_dino(args)

## lung
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --master_port=6666 --nproc_per_node=4 Hierarchical-Pretraining/main_SLPD.py --output_dir OUTPUT --epochs 100 --batch_size_per_gpu 128 --intra_num_crops 10 --intra_loss_weight 1.0 --inter_loss_weight 1.0 --which_feat_to_cluster_stu mlp --which_feat_to_cluster_tea mlp --neighbor_num 1 --loss_type intra_inter --dataset_dir /remote-home/share/Carboxy/Code/HIPT/Hierarchical-Pretraining/extract_features/tcga_lung_4k.json --num_cluster 4 --min_num_region 32
## brca
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --master_port=6666 --nproc_per_node=4 Hierarchical-Pretraining/main_SLPD.py --output_dir OUTPUT --epochs 100 --batch_size_per_gpu 128 --intra_num_crops 10 --intra_loss_weight 1.0 --inter_loss_weight 1.0 --which_feat_to_cluster_stu mlp --which_feat_to_cluster_tea mlp --neighbor_num 1 --loss_type intra_inter --dataset_dir /remote-home/share/Carboxy/Code/HIPT/Hierarchical-Pretraining/extract_features/tcga_brca_4k.json --num_cluster 2 --min_num_region 16
