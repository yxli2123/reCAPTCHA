import argparse
from torch.utils.data import DataLoader
import torch.nn as nn

from model import ResNet
from utils import *
from dataloader import CC


def main():
    parser = argparse.ArgumentParser(description='reCAPTCHA')

    # Experiment Parameters
    parser.add_argument('--mode',           type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--exp_dir',        type=str, default='')
    parser.add_argument('--exp_name',       type=str, default='ntire')
    parser.add_argument('--run_name',       type=str, default='1116')

    # Dataset Parameters
    parser.add_argument('--data_dir',       type=str, default='./data_image/captcha_20/')
    parser.add_argument('--info_file',      type=str, default='./data_image/data_info.json')

    # Training Parameters
    parser.add_argument('--epochs',         type=int, default=5000)
    parser.add_argument('--batch_size',     type=int, default=32)
    parser.add_argument('--lr',             type=float, default=1e-4)
    parser.add_argument('--threads',        type=int, default=4)
    parser.add_argument('--valid_interval', type=int, default=5000)

    # Model Parameters
    parser.add_argument('--model_name', type=str, default='EDVR', choices=['EDVR', 'BasicVSR'])
    parser.add_argument('--num_features', type=int, default=128)
    parser.add_argument('--num_frames', type=int, default=7)
    parser.add_argument('--groups', type=int, default=8)
    parser.add_argument('--front_RBs', type=int, default=5)
    parser.add_argument('--back_RBs', type=int, default=20)
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--scale_factor', type=int, default=1)

    # Test Parameters
    parser.add_argument('--test_source_dir', type=str, default='/gdata1/liyx/datasets/valid/VCC_QP_37_cut/')
    parser.add_argument('--test_target_dir', type=str, default='/gdata1/liyx/datasets/valid/GT/')
    parser.add_argument('--chop_H', type=int, default=0)
    parser.add_argument('--chop_W', type=int, default=0)
    parser.add_argument('--margin', type=int, default=4)

    # Mics
    parser.add_argument('--random_seed', type=int, default=1012)
    parser.add_argument('--num_gpus', type=int, default=1)

    args = parser.parse_args()

    # Set random seed
    setup_seed(args.seed)

    # Multi-GPU
    device_ids = range(args.num_gpus)
    device = torch.device(device_ids[0]) if args.num_gpus != 0 else torch.device('cpu')

    # Train params
    n_epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr

    # Build Datasets
    train_set = CC(args.info_file, args.image_dir, split='train', num_character=2)
    valid_set = CC(args.info_file, args.image_dir, split='valid', num_character=2)
    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=args.threads, shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, num_workers=args.threads, shuffle=True, drop_last=True)
    tokenizer = train_set.tokenizer

    # Build Models
    VM = ResNet(num_character=2)    # ResNet50
    LM = VM.LM                      # BERT-base Chinese

    # Build Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(VM.parameters(), lr)
    optimizer.zero_grad()

    for epoch in range(n_epochs):
        for batch in train_loader:
            # Load batch to device
            batch = {k: v.to(device) for k, v in batch.items()}

            # Predict
            y_pr = VM(batch)
            y_gt = batch['label']

            # Compute Loss and Backward Pass
            loss = criterion(y_pr, y_gt)
            loss.backward()
            optimizer.zero_grad()

