#!/bin/bash
## train
python pretrain_cifar.py --model resnet20

## adapt 
python adapt_cifar_c.py --model resnet20 --checkpoint checkpoints/resnet20_best.pth \
 --method Source

python adapt_cifar_c.py --model resnet20 --checkpoint checkpoints/resnet20_best.pth \
 --method TM-NORM --fold-bn

python adapt_cifar_c.py --model resnet20 --checkpoint checkpoints/resnet20_best.pth \
 --method TM-ENT --fold-bn
