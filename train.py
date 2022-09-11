import argparse
import os
import time

from utils.mytrainer import MyTrainer

args = None


def parse_args():
    parser = argparse.ArgumentParser(description='Train ')
    parser.add_argument('--labeled-data-dir', default='')  
    parser.add_argument('--unlabeled-data-dir', default='')
    parser.add_argument('--save-dir', default='./checkpoint')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--resume', default='./checkpoint/baseline_msunet.tar')
    parser.add_argument('--max-epoch', type=int, default=1200)
    parser.add_argument('--val-epoch', type=int, default=1)
    parser.add_argument('--val-start', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--is-gray', type=bool, default=False)
    parser.add_argument('--downsample-ratio', type=int, default=8)
    parser.add_argument('--seed', default=time.time())
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    trainer = MyTrainer(args)
    trainer.setup()
    trainer.train()
