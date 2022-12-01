import os
import sys
import pprint
import json
import torch
import datetime
import numpy as np


from loguru import logger
from numbers import Number
from test_cam import val_loc_one_epoch
from scm_nbdev.models.deit import *
from scm_nbdev.models.conformer import *
from scm_nbdev.utils import mkdir, Logger
from scm_nbdev.cams_deit import evaluate_cls_loc
from scm_nbdev.config.default import config as cfg
from scm_nbdev.core.functions import prepare_env
from scm_nbdev.core.lr_scheduler import LRScheduler
from scm_nbdev.config.default import cfg_from_list, cfg_from_file, update_config
from scm_nbdev.core.engine import create_data_loader, AverageMeter, accuracy, list2acc, adjust_lr_by_scheduler

from re import compile
from torch.utils.tensorboard import SummaryWriter
from timm.optim import create_optimizer
from timm.models import create_model as create_deit_model

CUBV2=False

def create_model(cfg, args):
    logger.info('==> Preparing networks for baseline...')
    # use gpu
    device = torch.device("cuda")
    assert torch.cuda.is_available(), "CUDA is not available"
    # model and optimizer
    model = create_deit_model(
            cfg.MODEL.ARCH,
            pretrained=False,
            num_classes=cfg.DATA.NUM_CLASSES,
            drop_rate=args.drop,
            drop_path_rate=args.drop_path,
            drop_block_rate=None,
        )
    if args.resume:
        checkpoint = torch.load(args.resume)
        pretrained_dict = {}

        for k, v in checkpoint['state_dict'].items():
            if not 'head' in k:
                k_ = '.'.join(k.split('.')[1:])
                pretrained_dict.update({k_: v})

        model.load_state_dict(pretrained_dict, strict=False)
        logger.info('load pretrained ts-cam model.')
    optimizer = create_optimizer(args, model)
    model = torch.nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count()))).to(device)
    # loss
    cls_criterion = torch.nn.CrossEntropyLoss().to(device)
    logger.info('Preparing networks done!')
    return device, model, optimizer, cls_criterion


def main():
    #update cfg attr directly, INGENIOUS
    args = update_config()
    # create checkpoint directory
    ds_dir= cfg.DATA.DATADIR.split("/")[-1]
    workdir= f"{ds_dir}_{cfg.MODEL.ARCH}"
    cfg.BASIC.SAVE_DIR= os.path.join(cfg.BASIC.SAVE_ROOT, workdir)
    cfg.BASIC.ROOT_DIR = "."
    log_dir = os.path.join(cfg.BASIC.SAVE_DIR, 'log'); mkdir(log_dir)
    ckpt_dir = os.path.join(cfg.BASIC.SAVE_DIR, 'ckpt'); mkdir(ckpt_dir)
    log_file = os.path.join(cfg.BASIC.SAVE_DIR, 'Log_' + cfg.BASIC.TIME + '.txt')
    # prepare running environment for the whole project
    prepare_env(cfg)

    # start loging
    sys.stdout = Logger(log_file)
    pprint.pprint(cfg)
    writer = SummaryWriter(log_dir)

    train_loader, test_loader, val_loader = create_data_loader(cfg, os.path.join(cfg.BASIC.ROOT_DIR, cfg.DATA.DATADIR))
    device, model, optimizer, cls_criterion = create_model(cfg, args)

    best_gtknown = 0
    best_top1_loc = 0
    update_train_step = 0
    update_val_step = 0
    opt_thred = -1
    lr_scheduler= LRScheduler(
        name= 'warmcos',
        lr= cfg.SOLVER.START_LR,
        iters_per_epoch=len(train_loader),
        total_epochs=cfg.SOLVER.NUM_EPOCHS,
        warmup_epochs= cfg.SOLVER.WARMUP_EPOCHS,
    )
    for epoch in range(1, cfg.SOLVER.NUM_EPOCHS+1):
        update_train_step, loss_train, cls_top1_train, cls_top5_train = \
            train_one_epoch(train_loader, model, device, cls_criterion,
                            optimizer, epoch, writer, cfg, update_train_step, lr_scheduler)
        if CUBV2:
            eval_results = val_loc_one_epoch(val_loader, model, device, )
        else:
            eval_results = val_loc_one_epoch(test_loader, model, device, )
        eval_results['epoch'] = epoch
        with open(os.path.join(cfg.BASIC.SAVE_DIR, 'val.txt'), 'a') as val_file:
            val_file.write(json.dumps(eval_results))
            val_file.write('\n')        

        loc_gt_known = eval_results['GT-Known_top-1']
        thred = eval_results['det_optThred_thr_50.00_top-1']
        # if loc_top1_val > best_top1_loc:
        #     best_top1_loc = loc_top1_val
        #     torch.save({
        #         "epoch": epoch,
        #         'state_dict': model.state_dict(),
        #         'best_map': best_gtknown
        #     }, os.path.join(ckpt_dir, 'model_best_top1_loc.pth'))
        if loc_gt_known > best_gtknown:
            best_gtknown = loc_gt_known
            torch.save({
                "epoch": epoch,
                'state_dict': model.state_dict(),
                'best_map': best_gtknown
            }, os.path.join(ckpt_dir, f'model_best.pth'))
            opt_thred = thred

        logger.info("Best GT_LOC: {}".format(best_gtknown))
        logger.info("Best TOP1_LOC: {}".format(best_gtknown))

        writer.add_scalar('acc_iter/best_gt_loc', best_gtknown, epoch)

        # torch.save({
        #     "epoch": epoch,
        #     'state_dict': model.state_dict(),
        #     'best_map': best_gtknown
        # }, os.path.join(ckpt_dir, 'model_epoch{}.pth'.format(epoch)))

        logger.info(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M'))

    if CUBV2:
        logger.info('Testing...')
        checkpoint = torch.load(os.path.join(ckpt_dir, f'model_best.pth'))
        pretrained_dict = {}

        for k, v in checkpoint['state_dict'].items():
            k_ = '.'.join(k.split('.')[1:])
            pretrained_dict.update({k_: v})

        model.load_state_dict(pretrained_dict)
        eval_results = val_loc_one_epoch(test_loader, model, device, opt_thred=opt_thred)
        for k, v in eval_results.items():
            if isinstance(v, np.ndarray):
                v = [round(out, 2) for out in v.tolist()]
            elif isinstance(v, Number):
                v = round(v, 2)
            else:
                raise ValueError(f'Unsupport metric type: {type(v)}')
            logger.info(f'\n{k} : {v}')
        with open(os.path.join(cfg.BASIC.SAVE_DIR, 'test.txt'), 'a') as test_file:
            test_file.write(json.dumps(eval_results))
            test_file.write('\n')  

def train_one_epoch(train_loader, model, device, criterion, optimizer, epoch,
                    writer, cfg, update_train_step, lr_scheduler):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    log_var = ['module.layers.[0-9]+.fuse._loss_rate', 'module.layers.[0-9]+.thred']
    log_scopes = [compile(log_scope) for log_scope in log_var]
    
    model.train()
    for i, (input, target) in enumerate(train_loader):
        # update iteration steps
        update_train_step += 1

        target = target.to(device)
        input = input.to(device)
        vars = {}
        for log_scope in log_scopes:
            vars.update({key:val for key, val in model.named_parameters()
                    if log_scope.match(key)})
        
        cls_logits = model(input)
        loss = criterion(cls_logits, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        adjust_lr_by_scheduler(lr_scheduler, optimizer, update_train_step)

        prec1, prec5 = accuracy(cls_logits.data.contiguous(), target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))
        lr= optimizer.param_groups[0]['lr']
        writer.add_scalar('loss_iter/train', loss.item(), update_train_step)
        writer.add_scalar('loss_iter/lr', lr, update_train_step)
        writer.add_scalar('acc_iter/train_top1', prec1.item(), update_train_step)
        writer.add_scalar('acc_iter/train_top5', prec5.item(), update_train_step)
        
        for k, v in vars.items():
            writer.add_scalar(k, v.item(), update_train_step)

        if i % cfg.BASIC.DISP_FREQ == 0 or i == len(train_loader)-1:
            logger.info(('Train Epoch: [{0}][{1}/{2}],lr: {lr}\t'
                   'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                   'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                   'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i + 1, len(train_loader), loss=losses,
                top1=top1, top5=top5, lr=optimizer.param_groups[-1]['lr'])))
    return update_train_step, losses.avg, top1.avg, top5.avg



if __name__ == "__main__":
    main()