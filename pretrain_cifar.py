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
import torch.optim as optim
import torchvision.transforms as transforms
from datetime import datetime
from tqdm import tqdm
from spikingjelly.activation_based import functional
from models.spiking_resnet import spiking_resnet20_cifar, spiking_resnet19_m_cifar
from models.neurons import BNLIFNode
from utils.augmentation import Cutout
from utils.functions import get_logger, seed_all


def parse_args():
    parser = argparse.ArgumentParser(description='CIFAR-10 PyTorch Training')
    parser.add_argument('--batch-size', '-b', type=int, default=128, help='batch size for training (default: 128)')
    parser.add_argument('--test-batch-size', type=int, default=128, help='batch size for testing (default: 128)')
    parser.add_argument('--epochs', '-e', type=int, default=200, help='number of total epochs to train (default: 200)')
    parser.add_argument('--resume', type=str, default=None, help='path to the checkpoint (default: None)')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate (default: 0.001)')
    parser.add_argument('--lr-scheduler', type=str, default='cosine', choices=['cosine', 'step', 'none'],
                        help='learning rate scheduler (default: cosine)')
    parser.add_argument('--optim', type=str, default='adam', choices=['adam', 'sgd'], help='optimizer (default: adam)')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0, help='weight decay (default: 0)')
    parser.add_argument('--seed', type=int, default=1000, help='random seed (default: 1000)')
    parser.add_argument('--model', type=str, default='resnet20', choices=['resnet20', 'resnet19_m'], 
                        help='model architecture (default: resnet20)')
    parser.add_argument('--spiking-neuron', '-s', type=str, default='BNLIFNode', choices=['BNLIFNode'], 
                        help='spiking neuron type (default: BNLIFNode)')
    parser.add_argument('--time', '-T', type=int, default=4, help='number of SNN simulation time steps (default: 4)')
    parser.add_argument('--mpbn', type=int, choices=[0, 1], default=1, help='use MPBN (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, help='batches to wait before logging training status')
    parser.add_argument('--data-dir', type=str, default='./data', help='path to the dataset')
    parser.add_argument('--save-dir', type=str, default='./checkpoints', help='path to save the model')
    parser.add_argument('--log-dir', type=str, default='./logs', help='path to save the logs')
    parser.add_argument('--gpu', type=int, default=0, help='GPU id to use (default: 0)')
    parser.add_argument('--cpu', action='store_true', help='use CPU anyway')
    parser.add_argument('--test-only', action='store_true', help='only test the model from the checkpoint')
    return parser.parse_args()


def train(model, device, train_loader, optimizer, criterion, epoch, args, logger):
    model.train()
    losses = []
    correct = 0
    total = 0
    
    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Epoch {epoch}')
    for batch_idx, (data, target) in pbar:
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data).mean(0)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        functional.reset_net(model)

        losses.append(loss.item())
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        if batch_idx % args.log_interval == 0:
            current_acc = 100. * correct / total
            current_loss = np.mean(losses[-args.log_interval:])
            pbar.set_postfix({
                'Loss': current_loss,
                'Acc': current_acc  # update the progress bar with current loss and accuracy
            })
            logger.info(f"Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} "
                        f"({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {current_loss:.6f}\t"
                        f"Accuracy: {current_acc:.2f}%")
    
    epoch_loss = np.mean(losses)
    epoch_acc = 100. * correct / total
    logger.info(f"Train Epoch: {epoch}\tAverage Loss: {epoch_loss:.6f}\tAccuracy: {epoch_acc:.2f}%")
    
    return epoch_loss, epoch_acc


def test(model, device, test_loader, criterion, logger):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc='Test'):
            data, target = data.to(device), target.to(device)
            output = model(data).mean(0)
            test_loss += criterion(output, target).item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            functional.reset_net(model)
    
    test_loss /= len(test_loader)
    accuracy = 100. * correct / total
    
    logger.info(f"Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{total} ({accuracy:.2f}%)")
    
    return test_loss, accuracy


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
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # Configure logger
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f'{current_time}_{args.model}_{args.time}dt_{args.spiking_neuron}_mpbn{args.mpbn}' \
                   f'_bs{args.batch_size}_lr{args.lr}_maxepochs{args.epochs}.log'
    log_filepath = os.path.join(args.log_dir, 'pretrain', log_filename)
    logger = get_logger(log_filepath, 'TRAIN')
    logger.info(f"Arguments: {vars(args)}")
    
    # Data preprocessing and augmentation
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        Cutout(n_holes=1, length=16),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # Load datasets
    trainset = torchvision.datasets.CIFAR10(
        root=args.data_dir, train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    testset = torchvision.datasets.CIFAR10(
        root=args.data_dir, train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=args.test_batch_size, shuffle=False, num_workers=4)
    
    # Create model
    if args.spiking_neuron == 'BNLIFNode':
        spiking_neuron = BNLIFNode
    if args.model == 'resnet20':
        model = spiking_resnet20_cifar(T=args.time, spiking_neuron=spiking_neuron, mpbn=args.mpbn).to(device)
    elif args.model == 'resnet19_m':
        model = spiking_resnet19_m_cifar(T=args.time, spiking_neuron=spiking_neuron, mpbn=args.mpbn).to(device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    if args.optim == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optim == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    start_epoch = 1
    best_acc = 0
    best_epoch = 0

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        last_epoch = checkpoint['epoch']
        start_epoch = last_epoch + 1
        best_acc = checkpoint['acc']
        logger.info(f"Checkpoint loaded: '{args.resume}'")
        logger.info(f"Current accuracy: {best_acc:.2f}%, epoch: {start_epoch - 1}")
        logger.info(f"Resuming from epoch {start_epoch}")

    if args.test_only:
        logger.info("Starting testing...")
        logger.info(f"Recorded acc in checkpoint: {best_acc:.2f}%")
        test_loss, test_acc = test(model, device, testloader, criterion, logger)
        logger.info(f"Test accuracy: {test_acc:.2f}%")
        exit(0)
    
    # Training loop
    start_time = time.time()
    logger.info("Starting training...")
    for epoch in range(start_epoch, args.epochs + 1):
        train_loss, train_acc = train(model, device, trainloader, optimizer, criterion, epoch, args, logger)

        test_loss, test_acc = test(model, device, testloader, criterion, logger)

        scheduler.step()
        
        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'acc': best_acc,
                'epoch': best_epoch,
                'args': vars(args)
            }
            torch.save(checkpoint, os.path.join(args.save_dir, f'{args.model}_best.pth'))
    
        # Print epoch summary
        logger.info(f'Epoch {epoch}/{args.epochs}: '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
              f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%, '
              f'LR: {optimizer.param_groups[0]["lr"]:.6f}, '
              f'Current Best Acc: {best_acc:.2f}% in epoch {best_epoch}')
    
    # Training complete
    elapsed_time = time.time() - start_time
    logger.info(f'Training completed in {elapsed_time/60:.2f} minutes')
    logger.info(f'Best test accuracy: {best_acc:.2f}% in epoch {best_epoch}')
