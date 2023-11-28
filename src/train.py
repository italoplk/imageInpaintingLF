from opt import get_args
from utils.metrics import *
from utils.lr import CustomExpLr
import random
import time


import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR, ExponentialLR
from torch.utils.data import Subset
import albumentations as A
from torch.utils.data import DataLoader

from tqdm import tqdm
#import wandb
import os
import shutil

from dataset import MPAIDataset
from models import ModelCNR, ModelUnet, ModelConv, ModelConvUnet
from models_vit import InpaitingViT, InpaitingGViT

# train.py --model conv --epochs 1000 --train-repeats 31 --test-repeats 50 --batch-size 16 --n-crops 4 --save /scratch/output_ImageInpainting/train_conv_mydecoder --project-name train_conv_mydecoder --lr-scheduler custom_exp

best_loss_val = float('inf')
best_loss_train = float('inf')
best_loss_test = float('inf')

def main():
    global best_loss_val
    global best_loss_train
    global best_loss_test

    args = get_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
    
    args.save = f'{args.save}_cs{args.context_size}_ps{args.predictor_size}_n_crops{args.n_crops}_trainrep{args.train_repeats}_testrep{args.test_repeats}'
    os.makedirs(args.save, exist_ok=True)
    
    #wandb.login()

    
    if(args.model == 'conv'):
        model = ModelConv(
            nFilters=32,
            nBottleneck=512,
            context_size=args.context_size,
            predictor_size=args.predictor_size,
            my_decoder=args.my_decoder
        ) 
    elif(args.model == 'conv_unet'):
        model = ModelConvUnet(
            nFilters=32,
            nBottleneck=512,
            context_size=args.context_size,
            predictor_size=args.predictor_size
        ) 
    elif(args.model == 'cnr'):
        model = ModelCNR(
            nFilters=32,
            nBottleneck=512,
            context_size=args.context_size,
            predictor_size=args.predictor_size
        ) 
    elif(args.model == 'cnr_unet'):
        model = ModelUnet(
            nFilters=32,
            nBottleneck=512,
            context_size=args.context_size,
            predictor_size=args.predictor_size
        ) 
    elif(args.model == 'vit'):
        model = InpaitingViT(
            # IDM NVIEWS
            context_size = args.context_size,
            predictor_size= args.predictor_size,
            dim = 768,
            depth = 6,
            heads = 12,
            mlp_dim = 2048,
            channels = 1,
            dim_head=32,
            dropout = 0.1,
            emb_dropout = 0.1
        )

        # IDM MLP_DIM
    elif(args.model == 'gvit'):
        model = InpaitingGViT(
            context_size = args.context_size,
            predictor_size= args.predictor_size,
            dim = 768,
            depth = 6,
            heads = 12,
            mlp_dim = 2048,
            channels = 1,
            dim_head=32,
            dropout = 0.1,
            emb_dropout = 0.1,
            knn=9,
            dense=True
        )
    else:
        raise NotImplementedError(f'{args.model} model not implemented')

    if(torch.cuda.is_available()):
        model = model.cuda()
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    criterion = nn.L1Loss().to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr = args.lr, betas=(0.9,0.999)
    )
    torch.nn.utils.clip_grad_value_(model.parameters(), 1)

    #scheduler = ExponentialLR(optimizer=optimizer,gamma=args.lr_gamma)
    
    args.lr_scheduler = args.lr_scheduler.lower()
    if args.lr_scheduler == "steplr":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    elif args.lr_scheduler == "cosineannealinglr":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs - args.lr_warmup_epochs, eta_min=args.lr_min
        )
    elif args.lr_scheduler == "exponentiallr":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_gamma)
    elif args.lr_scheduler == "custom_exp":
        scheduler = CustomExpLr(optimizer=optimizer, initial_learning_rate=args.lr, decay_steps=args.epochs, decay_rate=args.lr_gamma)
    else:
        # scheduler = None
        raise RuntimeError(
            f"Invalid lr scheduler '{args.lr_scheduler}'. Only StepLR, CosineAnnealingLR and ExponentialLR "
            "are supported."
        )


    # optionally resume from a checkpoint
    args.start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"=> loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume)

            args.start_epoch = checkpoint['epoch']
            best_loss_val = checkpoint['best_loss_val']

            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))


    transform = A.Compose(
        [
            A.Rotate(limit=(-90,-90), p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5)
        ]
    ) 

    train_data = MPAIDataset(
        path = args.train_path,
        context_size=args.context_size,
        predictor_size=args.predictor_size,
        bit_depth=args.bit_depth,
        transforms=transform,
        repeats = args.train_repeats * args.n_crops,
    )
    val_data = MPAIDataset(
        path = args.val_path,
        context_size=args.context_size,
        predictor_size=args.predictor_size,
        bit_depth=args.bit_depth,
        repeats = args.val_repeats * args.n_crops
    )
    test_data = MPAIDataset(
        path = args.test_path,
        context_size=args.context_size,
        predictor_size=args.predictor_size,
        bit_depth=args.bit_depth,
        repeats = args.test_repeats * args.n_crops,
    )
#TODO idm increase workers maybe
    train_loader = DataLoader(train_data, shuffle=True, batch_size=args.batch_size, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_data, shuffle=False, batch_size=args.batch_size, num_workers=8, pin_memory=True)
    test_loader = DataLoader(test_data, shuffle=False, batch_size=1, num_workers=8, pin_memory=True)


    # wandb.init(
    #   # Set the project where this run will be logged
    #   project='image_inpainting',
    #   # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
    #   name=f'{args.project_name}_cs{args.context_size}_ps{args.predictor_size}_n_crops{args.n_crops}_trainrep{args.train_repeats}_testrep{args.test_repeats}',
    #   config={
    #     'epochs': args.epochs,
    #     'batch_size': args.batch_size,
    #     'TrainingSet':args.train_path,
    #     'ValidationSet': args.val_path,
    #     'TestSet': args.test_path,
    #     'Model': args.model,
    #     'Context':args.context_size,
    #     'Predictor':args.predictor_size,
    #     'bit':args.bit_depth
    #     })
    
    for epoch in range(args.start_epoch, args.epochs):

        print(f'\nStarting epoch {epoch}')
        # train for one epoch
        loss_train = train(train_loader, model, criterion, optimizer, epoch, device, args)
        best_loss_train = min(loss_train, best_loss_train)

        # evaluate on validation set
        loss_val = validate(val_loader, model, criterion, device, epoch)
        
        scheduler.step()
        
        # remember best loss and save checkpoint
        is_best = loss_val < best_loss_val
        best_loss_val = min(loss_val, best_loss_val)

        check = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_loss_val': best_loss_val,
            'optimizer' : optimizer.state_dict(),
            'scheduler' : scheduler.state_dict()
        }

        # save best & last
        save_checkpoint(check, is_best, out_dir=args.save)

        # save epoch
        save_checkpoint(check, is_best=False, out_dir=args.save, filename=f'ep_{epoch}.pth.tar')

        loss_test = validate(test_loader, model, criterion,device, epoch,tag='test/loss')
        best_loss_test = min(loss_test, best_loss_test)

        print(f"train loss, {loss_train:.3f}, val loss, {loss_val:.3f}", file=open("/scratch/results_evc/epoch_mse.csv",'a'))
        print(f"Epoch: {epoch}\n", file=open(f"/scratch/results_evc/batch_mse.csv", 'a'))
    #     wandb.log({
    #         'train/best_loss':best_loss_train,
    #         'val/best_loss':best_loss_val,
    #         'test/best_loss':best_loss_test,
    #     },step = epoch)
    #
    # wandb.finish()


def train(train_loader, model, criterion, optimizer, epoch, device, args):

    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    pbar = enumerate(train_loader)
    pbar = tqdm(pbar, 
        total=len(train_loader))
    pbar.set_description(f'Train [{epoch} / {args.epochs}] ')

    end = time.time()
    for i, (features, target) in pbar:
        # measure data loading time
        data_time.update(time.time() - end)

        # move data to the same device as model
        features = features.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        output, _,_ = model(features)
        loss = criterion(output, target)


        # record loss
        losses.update(loss.item(), features.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # pbar.set_postfix({
        #     'loss': f'{losses.val:.3f} ({losses.avg:.3f})',
        # })
        print(f"loss, {losses.val:.3f}, loss_avg, {losses.avg:.3f}", file=open(f"/scratch/results_evc/batch_mse.csv", 'a'))

        #if i % args.print_freq == 0:
        #    progress.display(i + 1)
    progress.display_summary()
    # wandb.log({
    #     'train/loss':losses.avg,
    #     'lr': optimizer.param_groups[0]['lr']
    # },step = epoch)


    return losses.avg


def validate(val_loader, model, criterion, device, epoch, tag='val/loss'):

    def run_validate(loader):
        with torch.no_grad():
            end = time.time()
            pbar = enumerate(loader)
            pbar = tqdm(pbar, 
                total=len(loader))
            pbar.set_description(f'Validation ')
            for i, (features, target) in pbar:

                features = features.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)

                # compute output
                output, _,_ = model(features)
                loss = criterion(output, target)

                # record loss
                losses.update(loss.item(), features.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                pbar.set_postfix({
                    'loss': f'{losses.val:.3f} ({losses.avg:.3f})',
                })

    batch_time = AverageMeter('Time', ':6.3f', Summary.AVERAGE)
    losses = AverageMeter('Loss', ':.4e', Summary.AVERAGE)

    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses],
        prefix='Eval: ')

    # switch to evaluate mode
    model.eval()

    run_validate(val_loader)

    progress.display_summary()
    # wandb.log({
    #     tag:losses.avg,
    # }, step = epoch)

    return losses.avg


def save_checkpoint(state, is_best, out_dir, filename='checkpoint.pth.tar'):
    torch.save(state, f'{out_dir}/{filename}')
    if is_best:
        shutil.copyfile(f'{out_dir}/{filename}', f'{out_dir}/model_best.pth.tar')


    
if __name__ == '__main__':
    main()