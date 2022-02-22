"""
This code is modified from Hengyuan Hu's repository.
https://github.com/hengyuan-hu/bottom-up-attention-vqa
"""
import os
import argparse
from encoder_train import Training
from config import cfg
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='datasets')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--hidden', type=int, default=1280)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=1204, help='random seed')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    cfg.DATA_DIR = args.data_dir
    cfg.EPOCHS = args.epochs
    cfg.HIDDEN_SIZE = args.hidden
    cfg.BATCH_SIZE = args.batch_size
    cfg.LR = args.lr

    trainer = Training(cfg)
    trainer.start_training()


 
  