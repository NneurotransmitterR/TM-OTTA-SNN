import argparse
import os
import random
import time
import logging
import torch
import torchvision
from tqdm import tqdm
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from datetime import datetime
from tqdm import tqdm
from spikingjelly.activation_based import functional
import tm_module
from tm_module import softmax_entropy
from fold_bn import fold_vbn
from utils.functions import seed_all, get_logger
from datasets.cifar10c import CIFAR10C, COMMON_CORRUPTIONS, ALL_CORRUPTIONS
from models.spiking_resnet import spiking_resnet20_cifar, spiking_resnet19_m_cifar
from models.neurons import BNLIFNode


def parse_args():
    parser = argparse.ArgumentParser(description='CIFAR-10-C Experiments')
    parser.add_argument('--batch-size', '-b', type=int, default=64, help='batch size for testing (default: 64)')
    parser.add_argument('--adapt-batches', type=int, help='number of total batches to adapt')
    parser.add_argument('--checkpoint', type=str, help='path to the pretrained model checkpoint')
    parser.add_argument('--lr', type=float, help='learning rate for TM-ENT')
    parser.add_argument('--lr-scheduler', type=str, default='none', choices=['cosine', 'step', 'none'],
                        help='learning rate scheduler (default: none)')
    parser.add_argument('--optim', type=str, default='adam', choices=['adam', 'sgd'], help='optimizer (default: adam)')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0, help='weight decay (default: 0)')
    parser.add_argument('--seed', type=int, default=1000, help='random seed (default: 1000)')
    parser.add_argument('--model', type=str, default='resnet20', choices=['resnet20', 'resnet19_m'], 
                        help='model architecture (default: resnet20)')
    parser.add_argument('--spiking_neuron', '-s', type=str, default='BNLIFNode', choices=['BNLIFNode'], 
                        help='spiking neuron type (default: BNLIFNode)')
    parser.add_argument('--time', '-T', type=int, default=4, help='number of SNN simulation time steps, '
                        'should be consistent with training (default: 4)')
    parser.add_argument('--mpbn', type=int, choices=[0, 1], default=1, help='use MPBN (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, help='batches to wait before logging status')
    parser.add_argument('--data-dir', type=str, default='./data', help='path to the dataset')
    parser.add_argument('--log-dir', type=str, default='./logs', help='path to save the logs')
    parser.add_argument('--gpu', type=int, default=0, help='GPU id to use (default: 0)')
    parser.add_argument('--cpu', action='store_true', help='use CPU anyway')
    parser.add_argument('--method', type=str, default='Source', choices=['Source', 'TM-NORM', 'TM-ENT'],
                        help='method to use (default: TM-NORM)')
    parser.add_argument('--fold-bn', action='store_true', help='fold MPBNs')
    parser.add_argument('--running-stats', action='store_true', help='use running stats for TM')
    parser.add_argument('--normalize-residual', action='store_true', help='use normalize residual for TM')
    parser.add_argument('--corruptions', type=list[str], default=COMMON_CORRUPTIONS, 
                        help='corruption types (default: COMMON_CORRUPTIONS)')
    parser.add_argument('--severity', type=int, default=5, choices=[1, 2, 3, 4, 5], help='severity level (default: 5)')
    return parser.parse_args()


def make_model_adapt(model, args, logger):
    logger.info("Setting up model for adaptation...")
    if args.method == 'Source':
        model.eval()
        if args.fold_bn:
            model = fold_vbn(model, args.normalize_residual)
        logger.info("Model is ready for Source (without adaptation)...")
    elif args.method == 'TM-NORM':
        model.eval()
        if args.fold_bn:
            model = fold_vbn(model, args.normalize_residual, args.running_stats)
        else:
            for m in model.modules():
                if isinstance(m, (BNLIFNode,)):
                    m.vbn.track_running_stats = False
                    m.vbn.running_mean = None
                    m.vbn.running_var = None
        logger.info("Model is ready for TM-NORM...")
    elif args.method == 'TM-ENT':
        model = tm_module.configure_model(model, args.fold_bn, args.normalize_residual, args.running_stats)
        tm_module.check_model(model)
        params, paramnames = tm_module.collect_params(model, args.fold_bn)
        logger.info(f"Params to be optimized: {paramnames}")
        if args.optim == 'adam':
            optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
        elif args.optim == 'sgd':
            optimizer = optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        model = tm_module.TM(model, optimizer)
        logger.info("Model is ready for TM-ENT...")
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm, nn.Dropout)):
            m.eval()
    return model


def prepare_model(device, args, logger):
    if args.spiking_neuron == 'BNLIFNode':
        spiking_neuron = BNLIFNode
    if args.model == 'resnet20':
        model = spiking_resnet20_cifar(T=args.time, spiking_neuron=spiking_neuron, mpbn=args.mpbn).to(device)
    elif args.model == 'resnet19_m':
        model = spiking_resnet19_m_cifar(T=args.time, spiking_neuron=spiking_neuron, mpbn=args.mpbn).to(device)
    if os.path.isfile(args.checkpoint):
        logger.info(f"Loading checkpoint from {args.checkpoint}...")
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
    else:
        raise FileNotFoundError(f"Checkpoint not found at {args.checkpoint}, you need a trained model to start")
    model = make_model_adapt(model, args, logger)
    return model


def adapt(model, device, adapt_loader, args, logger):
    model.train()
    losses = []
    correct = 0
    total = 0
    
    total_batches = len(adapt_loader) if args.adapt_batches is None else min(args.adapt_batches, len(adapt_loader))
    pbar = tqdm(enumerate(adapt_loader), total=total_batches, desc='Processing')

    for batch_idx, (data, target) in pbar:
        if batch_idx == total_batches:
            break
        data, target = data.to(device), target.to(device)
        
        if args.method in ['Source', 'TM-NORM']:
            if args.method == 'Source':
                model.eval()
            with torch.no_grad():
                output = model(data).mean(0)
                adapt_labels = output.argmax(1)
                total += float(target.size(0))
                correct += float(adapt_labels.eq(target).sum().item())
                test_entropy = softmax_entropy(output).mean(0)
            functional.reset_net(model)
        
        elif args.method in ['TM-ENT']:
            output = model(data).mean(0)
            adapt_labels = output.argmax(1)
            total += float(target.size(0))
            correct += float(adapt_labels.eq(target).sum().item())
            test_entropy = softmax_entropy(output).mean(0)
   
        if batch_idx % args.log_interval == 0:
            current_acc = 100. * correct / total
            pbar.set_postfix({
                'Acc': current_acc  # update the progress bar with current accuracy
            })
            logger.info(f"Entropy: {test_entropy:.4f}  Running accuracy: {current_acc:.2f}%")
        
    epoch_acc = 100. * correct / total
    logger.info(f"Adaptation finished, accuracy: {epoch_acc:.2f}%, error rate: {100 - epoch_acc:.2f}%")
    
    return epoch_acc


if __name__ == '__main__':
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    seed_all(args.seed)
    
    # Check CUDA availability
    if args.cpu:
        device = torch.device('cpu')
    else:
        device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
        
    # Create directories
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # Configure logger
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f'{current_time}_{args.model}_{args.time}dt_{args.spiking_neuron}_mpbn{args.mpbn}' \
                   f'_bs{args.batch_size}_lr{args.lr}_{args.method}_fold_bn{args.fold_bn}' \
                   f'rs{args.running_stats}_nr{args.normalize_residual}_{args.fold_bn}.log'
    log_filepath = os.path.join(args.log_dir, 'adapt', log_filename)
    logger = get_logger(log_filepath, 'ADAPT')
    logger.info(f"Arguments: {vars(args)}")
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    acc_list = []
    if args.method == 'Source':
        args.batch_size = 128
        args.lr = 0
    elif args.lr is None:
        if args.batch_size == 1:
            args.lr = 0.00025 / 16
        else:
            args.lr = (0.00025 / 64) * args.batch_size * 2 if args.batch_size < 32 else 0.00025

    for corruption_type in args.corruptions:
        adapt_dataset = CIFAR10C(root=args.data_dir, corruptions=[corruption_type], severity=args.severity, 
                                 transform=transform_test, download=True)
        adapt_loader = torch.utils.data.DataLoader(
            adapt_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        
        logger.info(f"Adapting to {corruption_type}...")
        start_time = time.time()

        model = prepare_model(device, args, logger)

        acc = adapt(model, device, adapt_loader, args, logger)
        acc_list.append(acc)

        end_time = time.time()
        elapsed_time = (end_time - start_time) / 60
        logger.info(f"Adaptation to {corruption_type} completed in {elapsed_time:.2f} minutes")

    acc_dict = {k: v for k, v in zip(args.corruptions, acc_list)}
    err_dict = {k: 100 - v for k, v in acc_dict.items()}
    avg_running_acc = sum(acc_list) / len(acc_list)
    logger.info("Experiment finished, results:")
    logger.info(f"acc_dict: {acc_dict}")
    logger.info(f"err_dict: {err_dict}")
    logger.info(f"Average accuracy on all corruptions of severity {args.severity}: {avg_running_acc:.2f}%")
