import os
import time
import random
import numpy as np
import logging
import argparse

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.multiprocessing as mp
import torch.distributed as dist
from tensorboardX import SummaryWriter

from MinkowskiEngine import SparseTensor
from util import config
from util.util import AverageMeter, intersectionAndUnionGPU, \
    poly_learning_rate, save_checkpoint, \
    export_pointcloud, convert_labels_with_palette, extract_clip_feature
# from dataset.feature_loader import FusedFeatureLoader, collation_fn
from dataset.point_loader import Point3DLoader, collation_fn_eval_all, collation_fn_raw
from models.disnet import DisNet as Model
from tqdm import tqdm
import clip
import pyviz3d.visualizer as viz

def worker_init_fn(worker_id):
    '''Worker initialization.'''
    random.seed(time.time() + worker_id)

def get_parser():
    '''Parse the config file.'''

    parser = argparse.ArgumentParser(description='OpenScene 3D distillation.')
    parser.add_argument('--config', type=str,
                        help='config file')
    parser.add_argument('opts',
                        default=None,
                        help='see config/scannet/distill_openseg.yaml for all options',
                        nargs=argparse.REMAINDER)
    args_in = parser.parse_args()
    assert args_in.config is not None
    cfg = config.load_cfg_from_cfg_file(args_in.config)
    if args_in.opts:
        cfg = config.merge_cfg_from_list(cfg, args_in.opts)
    os.makedirs(cfg.save_path, exist_ok=True)
    model_dir = os.path.join(cfg.save_path, 'model')
    result_dir = os.path.join(cfg.save_path, 'result')
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(result_dir + '/last', exist_ok=True)
    os.makedirs(result_dir + '/best', exist_ok=True)
    return cfg


def get_logger():
    '''Define logger.'''

    logger_name = "main-logger"
    logger_in = logging.getLogger(logger_name)
    logger_in.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(filename)s line %(lineno)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger_in.addHandler(handler)
    return logger_in

def main_process():
    return not args.multiprocessing_distributed or (
        args.multiprocessing_distributed and args.rank % args.ngpus_per_node == 0)

def obtain_text_features(query):
    '''obtain the CLIP text feature for a query.'''

    if not os.path.exists('saved_text_embeddings'):
        os.makedirs('saved_text_embeddings')

    if 'openseg' in args.feature_2d_extractor:
        model_name="ViT-L/14@336px"
        postfix = '_768' # the dimension of CLIP features is 768
    elif 'lseg' in args.feature_2d_extractor or 'clipcrop' in args.feature_2d_extractor:
        model_name="ViT-B/32"
        postfix = '_512' # the dimension of CLIP features is 512
    else:
        raise NotImplementedError
    print("Loading CLIP {} model...".format(model_name))
    clip_pretrained, _ = clip.load(model_name, device='cuda', jit=False)
    print("Finish loading")
    q = ['nothing', query]
    text = clip.tokenize(q)
    text = text.cuda()
    text_features = clip_pretrained.encode_text(text)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return text_features

def main():
    '''Main function.'''
    args = get_parser()
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(
        str(x) for x in args.train_gpu)
    cudnn.benchmark = True
    if args.manual_seed is not None:
        random.seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)
    
    # By default we use shared memory for training
    if not hasattr(args, 'use_shm'):
        args.use_shm = True

    print(
        'torch.__version__:%s\ntorch.version.cuda:%s\ntorch.backends.cudnn.version:%s\ntorch.backends.cudnn.enabled:%s' % (
            torch.__version__, torch.version.cuda, torch.backends.cudnn.version(), torch.backends.cudnn.enabled))

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    args.ngpus_per_node = len(args.train_gpu)
    if len(args.train_gpu) == 1:
        args.sync_bn = False
        args.distributed = False
        args.multiprocessing_distributed = False
        args.use_apex = False

    if args.multiprocessing_distributed:
        args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=args.ngpus_per_node,
                 args=(args.ngpus_per_node, args))
    else:
        main_worker(args.train_gpu, args.ngpus_per_node, args)

def get_model(cfg):
    '''Get the 3D model.'''

    model = Model(cfg=cfg)
    return model

def main_worker(gpu, ngpus_per_node, argss):
    global args
    global best_iou
    args = argss

    if args.distributed:
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, 
                                world_size=args.world_size, rank=args.rank)

    model = get_model(args)
    if main_process():
        global logger, writer
        logger = get_logger()
        writer = SummaryWriter(args.save_path)
        logger.info(args)
        logger.info("=> creating model ...")

    # ####################### Optimizer ####################### #
    optimizer = torch.optim.Adam(model.parameters(), lr=args.base_lr)
    args.index_split = 0

    if args.distributed:
        torch.cuda.set_device(gpu)
        args.batch_size = int(args.batch_size / ngpus_per_node)
        args.batch_size_val = int(args.batch_size_val / ngpus_per_node)
        args.workers = int(args.workers / ngpus_per_node)
        model = torch.nn.parallel.DistributedDataParallel(
            model.cuda(), device_ids=[gpu])
    else:
        model = model.cuda()

    # Training
    if args.resume:
        if os.path.isfile(args.resume):
            if main_process():
                logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage.cuda())
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'], strict=True)
            optimizer.load_state_dict(checkpoint['optimizer'])
            best_iou = checkpoint['best_iou']
            if main_process():
                logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            if main_process():
                logger.info("=> no checkpoint found at '{}'".format(args.resume))        
    # Pre-trained Openscene for testing
    if os.path.isfile(args.model_path):
        if main_process():
            logger.info("=> loading checkpoint '{}'".format(args.model_path))
        checkpoint = torch.load(args.model_path, map_location=lambda storage, loc: storage.cuda())
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        if main_process():
            logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.model_path, checkpoint['epoch']))
    else:
        raise RuntimeError("=> no checkpoint found at '{}'".format(args.model_path))
    
    # ####################### Data Loader ####################### #
    if not hasattr(args, 'input_color'): # by default we do not use the point color as input
        args.input_color = False
    training = False
    if training:
        criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label).cuda(gpu) # for evaluation
        pass
    testing = True
    if testing:
        val_data = Point3DLoader(datapath_prefix=args.data_root,
                                 voxel_size=args.voxel_size,
                                 split='pcl', aug=False,
                                 memcache_init=args.use_shm,
                                 eval_all=True,
                                 input_color=args.input_color)
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_data) if args.distributed else None
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size_val,
                                                shuffle=False,
                                                num_workers=args.workers, pin_memory=True,
                                                drop_last=False, collate_fn=collation_fn_raw,
                                                sampler=val_sampler)
        return inference(val_loader,model)
        
    # ####################### Training wrapper ####################### #
    return 0

def apply_threshold(softmax_tensor, threshold):
    max_values, _ = torch.max(softmax_tensor, dim=1)
    binary_tensor = torch.where(max_values >= threshold, torch.tensor(1).to('cuda'), torch.tensor(0).to('cuda'))    
    return binary_tensor

def inference(val_loader, model):
    '''Validation.'''

    torch.backends.cudnn.enabled = False

    # obtain the CLIP feature. Under development
    visualize = True
    if visualize:
        q1 = 'cushion'
        q2 = 'sofa right across the TV'
        q3 = 'christmas stockings'
        q4 = 'present'
        q5 ='wall art with a green background'
        text_features = obtain_text_features(q1)
        with torch.no_grad():
            for batch_data in tqdm(val_loader):
                (coords, feat, inds, path, color) = batch_data
                sinput = SparseTensor(
                    feat.cuda(non_blocking=True), coords.cuda(non_blocking=True))
                output = model(sinput)
                output = output[inds,:]
                output = output.half() @ text_features.t()
                output_sm = torch.softmax(output, dim=1)

                select = torch.max(output_sm, 1)[1]
                binary_thresh = apply_threshold(output_sm, 0.75)
                # Applying threshold on predicted softmax
                output = torch.logical_and(select, binary_thresh)
                
                # Visualization
                scene_name = path[0].replace(path[0][:-12],'').replace('.pth','')
                gt_data = '../OpenSun3D.github.io/challenge/benchmark_data/gt_development_scenes/'+scene_name+'.txt'
                label = open(gt_data, 'r').readlines()
                label=np.array([int(s.strip('\n')) for s in label])

                locs_in, feats_in = torch.load(path[0])
                point = locs_in
                v = viz.Visualizer()
                v.add_points(f'pcl color', np.array(point), np.array(color), point_size=20, visible=True)
                color = np.array(color)
                color[torch.where(output==1)[0].cpu(),:] = (255,0,0)
                v.add_points(f'semantic ideal', np.array(point), np.array(color), point_size=20, visible=True)
                color[np.where(label!=0)[0],:]=np.array((0,255,0))
                v.add_points(f'GT label', np.array(point), np.array(color), point_size=20, visible=True)
                v.save('work_dirs/viz')
                return 0
            
            ### Pending ###


if __name__ == '__main__':
    main()