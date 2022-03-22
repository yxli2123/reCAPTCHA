import argparse
import json
import os
import random

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torch.nn as nn

from model import ResNet
from utils import *
from dataloader import CC


def main():
    parser = argparse.ArgumentParser(description='reCAPTCHA')

    # Experiment Parameters
    parser.add_argument('--mode',           type=str,   default='train', choices=['train', 'test'])
    parser.add_argument('--exp_dir',        type=str,   default='../exp_log')
    parser.add_argument('--exp_name',       type=str,   default='captcha_single_05')
    parser.add_argument('--run_name',       type=str,   default='bs_2')

    # Dataset Parameters
    parser.add_argument('--data_dir',       type=str,   default='../data_image/captcha_single_05/')
    parser.add_argument('--info_file',      type=str,   default='../data_image/data_single_info.json')
    parser.add_argument('--num_char',       type=int,   default=1)
    parser.add_argument('--topk',           type=int,   default=5)

    # Training Parameters
    parser.add_argument('--epochs',         type=int,   default=5000)
    parser.add_argument('--batch_size',     type=int,   default=2)
    parser.add_argument('--lr',             type=float, default=1e-4)
    parser.add_argument('--threads',        type=int,   default=4)
    parser.add_argument('--valid_interval', type=int,   default=10000)
    parser.add_argument('--print_interval', type=int,   default=1000)
    parser.add_argument('--save_interval',  type=int,   default=10000)

    # Model Parameters
    parser.add_argument('--VM_name',        type=str,   default='ResNet50')
    parser.add_argument('--LM_name',        type=str,   default='bert-base-chinese')
    parser.add_argument('--ckpt',           type=str,   default=None)

    # Mics
    parser.add_argument('--random_seed',    type=int,   default=769)
    parser.add_argument('--num_gpus',       type=int,   default=1)

    args = parser.parse_args()

    # Set random seed
    setup_seed(args.random_seed)

    # Multi-GPU
    device_ids = range(args.num_gpus)
    device = torch.device(device_ids[0]) if args.num_gpus != 0 else torch.device('cpu')

    if args.mode == 'train':
        # Train params
        n_epochs = args.epochs
        batch_size = args.batch_size
        lr = args.lr

        ############################
        #            DATA          #
        ############################
        train_set = CC(args.info_file, args.data_dir, split='train', num_character=args.num_char)
        valid_set = CC(args.info_file, args.data_dir, split='valid', num_character=args.num_char)
        train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=args.threads, shuffle=True,
                                  drop_last=True)
        valid_loader = DataLoader(valid_set, batch_size=batch_size, num_workers=args.threads, shuffle=True,
                                  drop_last=True)
        tokenizer = train_set.tokenizer

        ############################
        #           MODEL          #
        ############################
        # Build Models
        VM = ResNet(num_character=args.num_char)  # ResNet50
        # with torch.no_grad():  # Remove if fine-tuned
        #    LM = VM.LM                            # BERT-base Chinese

        # Checkpoint
        if args.ckpt:
            checkpoint = torch.load(args.ckpt, map_location=device)
            VM.load_state_dict(checkpoint['model_state_dict'])
        VM = VM.to(device)

        ############################
        #         OPTIMIZER        #
        ############################
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(VM.parameters(), lr)
        optimizer.zero_grad()

        ############################
        #           TRAIN          #
        ############################
        # Train log
        train_iter = 0
        log_dir = os.path.join(args.exp_dir, args.exp_name, args.run_name)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        writer = SummaryWriter(log_dir)

        for epoch in range(n_epochs):
            for t, batch in enumerate(train_loader):
                train_iter += 1
                VM.train()
                # Load batch to device
                batch = {k: v.to(device) for k, v in batch.items()}

                # Predict
                y_pr = VM(batch['image'])  # (B*num_char, vocab_size)
                y_gt = torch.cat(torch.chunk(batch['label'], args.num_char, dim=1), dim=0).squeeze()  # (B*num_char, 1)

                # Compute Loss and Backward Pass
                loss = criterion(y_pr, y_gt)
                loss.backward()
                optimizer.zero_grad()

                if train_iter % args.print_interval == 0:
                    writer.add_scalar('train/loss', loss.item(), train_iter)
                    writer.add_images('train/images', batch['image'], train_iter)

                if train_iter % args.valid_interval == 0 or t == train_loader.__len__() - 1:
                    metric = test(valid_loader, VM, tokenizer, device, args)
                    writer.add_scalar('valid/loss', metric['loss'].item(), train_iter)
                    writer.add_scalar('valid/acc_single', metric['acc_single'], train_iter)
                    writer.add_scalar('valid/acc_pair', metric['acc_pair'], train_iter)
                    writer.add_scalar('valid/acc_topk', metric['acc_topk'], train_iter)
                    random_sample = random.choices(metric['results'], k=10)
                    for k, sample in random_sample:
                        writer.add_text(f'valid/results{k}', str(random_sample), train_iter)

                if train_iter % args.save_interval == 0:
                    torch.save(VM.state_dict(), os.path.join(log_dir, f"{train_iter}.pth"))

    elif args.mode == 'test':
        test_set = CC(args.info_file, args.data_dir, split='test', num_character=args.num_char)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, num_workers=args.threads, shuffle=False,
                                 drop_last=False)
        tokenizer = test_set.tokenizer

        # Build Models
        VM = ResNet(num_character=args.num_char)  # ResNet50

        # Checkpoint
        if args.ckpt:
            checkpoint = torch.load(args.ckpt, map_location=device)
            VM.load_state_dict(checkpoint)

        metric = test(test_loader, VM, tokenizer, device, args)
        with open('./results.json', 'w') as fp:
            json.dump(metric, fp, indent=4)


@torch.no_grad()
def test(dataloader, model, tokenizer, device, args):
    y_pr = []
    y_gt = []
    for t, batch in enumerate(dataloader):
        model.eval()
        # Load batch to device
        batch = {k: v.to(device) for k, v in batch.items()}

        # Predict
        y_pr.append(model(batch['image']))  # (B*num_char, vocab_size)
        y_gt.append(torch.cat(torch.chunk(batch['label'], args.num_char, dim=1), dim=0))  # (B*num_char, 1)

    y_gt = torch.cat(y_gt, dim=0)
    y_pr = torch.cat(y_pr, dim=0)
    metric = evaluate(y_gt, y_pr, tokenizer, args.num_char, args.topk)

    return metric


if __name__ == '__main__':
    main()
