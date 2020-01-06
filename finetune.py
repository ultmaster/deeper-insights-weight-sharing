""" Search cell """
import math
import multiprocessing
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter

import utils
from config import SearchConfig
from models.search_cnn import SearchCNNController
from search import validate


def main(config, writer, logger, checkpoint, base_step):
    logger.info("Pretrained checkpoint: {}".format(checkpoint))

    # set seed
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    # get data with meta info
    input_size, input_channels, n_classes, train_data, test_data = utils.get_data(
        config.dataset, config.data_path, cutout_length=config.cutout_length, validation=True)

    net_crit = nn.CrossEntropyLoss().cuda()
    model = SearchCNNController(input_channels, config.init_channels, n_classes, config.n_layers, net_crit,
                                n_nodes=config.n_nodes, stem_multiplier=config.stem_multiplier,
                                bn_momentum=config.bn_momentum)
    model.cuda()
    model.load_state_dict(torch.load(checkpoint))

    base_epoch_number = base_step // (len(train_data) // config.batch_size)
    assert config.w_lr_scheduler == "cosine"
    base_lr = config.w_lr_min + (config.w_lr - config.w_lr_min) * \
              (1 + math.cos(math.pi * base_epoch_number / config.epochs)) / 2
    logger.info("Learning rate: {}".format(base_lr))

    # weights optimizer
    w_optim = optim.SGD(model.weights(), base_lr, momentum=config.w_momentum,
                        weight_decay=config.w_weight_decay)
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=config.batch_size,
                                               num_workers=config.workers,
                                               pin_memory=True,
                                               shuffle=True)
    valid_loader = torch.utils.data.DataLoader(test_data,
                                               batch_size=config.batch_size,
                                               num_workers=config.workers,
                                               pin_memory=True,
                                               shuffle=True)

    # training loop
    best_top1 = 0.
    for epoch in range(config.finetune_epochs):
        lr = w_optim.param_groups[0]["lr"]

        # training
        train(config, writer, logger, train_loader, valid_loader, model, w_optim, lr, epoch)

        # validation
        cur_step = (epoch + 1) * len(train_loader)
        if config.finetune_max_steps is None:
            top1 = validate(config, writer, logger, valid_loader, model, epoch, cur_step,
                            total_epochs=config.finetune_epochs)
        elif cur_step >= config.finetune_max_steps:
            break

        # save
        if best_top1 < top1:
            best_top1 = top1

    logger.info("Final best Prec@1 = {:.4%}".format(best_top1))
    print("")


def train(config, writer, logger, train_loader, valid_loader, model, w_optim, lr, epoch):
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    losses = utils.AverageMeter()

    cur_step = epoch * len(train_loader)
    writer.add_scalar('train/lr', lr, cur_step)

    for step, (trn_X, trn_y) in enumerate(train_loader):
        trn_X, trn_y = trn_X.cuda(non_blocking=True), trn_y.cuda(non_blocking=True)
        N = trn_X.size(0)

        model.train()

        w_optim.zero_grad()
        logits = model(trn_X)
        loss = model.criterion(logits, trn_y)
        loss.backward()
        if config.w_grad_clip != 0:
            # gradient clipping
            nn.utils.clip_grad_norm_(model.weights(), config.w_grad_clip)
        w_optim.step()

        prec1, prec5 = utils.accuracy(logits, trn_y, topk=(1, 5))
        losses.update(loss.item(), N)
        top1.update(prec1.item(), N)
        top5.update(prec5.item(), N)

        if config.finetune_max_steps is not None or step % config.print_freq == 0 or step == len(train_loader) - 1:
            logger.info(
                "Train: [{:2d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                "Prec@1 {top1.val:.1%} ({top1.avg:.1%}) Prec@5 {top5.val:.1%} ({top5.avg:.1%})".format(
                    epoch + 1, config.finetune_epochs, step, len(train_loader) - 1,
                    losses=losses, top1=top1, top5=top5))

        writer.add_scalar('train/loss', loss.item(), cur_step)
        writer.add_scalar('train/top1', prec1.item(), cur_step)
        writer.add_scalar('train/top5', prec5.item(), cur_step)
        cur_step += 1

        if config.finetune_max_steps is not None:
            validate(config, writer, logger, valid_loader, model, epoch, cur_step,
                     total_epochs=config.finetune_epochs)
            if cur_step >= config.finetune_max_steps:
                break

    logger.info("Train: [{:2d}/{}] Final Prec@1 {:.4%}".format(epoch + 1, config.finetune_epochs, top1.avg))


def worker(config, step):
    search_dir = os.path.join("checkpoints", config.name, step)
    for i, ckpt in enumerate(sorted(os.listdir(search_dir))):
        ckpt_name = ckpt.split(".")[0]
        output_path = os.path.join(config.path, "finetune_{}".format(step))
        os.makedirs(output_path, exist_ok=True)

        writer = SummaryWriter(log_dir=os.path.join(output_path, "tb"))
        writer.add_text('config', config.as_markdown(), 0)
        logger = utils.get_logger(os.path.join(output_path, "finetune_{}_{}_{}.log".format(step, ckpt_name,
                                                                                           config.name)), i)
        config.print_params(logger.info)

        logger.info("Finetuning start - Subgraph name: {}".format(ckpt_name))
        main(config, writer, logger, os.path.join(search_dir, ckpt), int(step))

        writer.close()


if __name__ == "__main__":
    config = SearchConfig()

    checkpoint_dir = os.path.join("checkpoints", config.name)
    if config.finetune_from_step is not None:
        if isinstance(config.finetune_from_step, list):
            search_dir = os.path.join("checkpoints", config.name)
            universe = sorted(os.listdir(search_dir))
            for d in config.finetune_from_step:
                worker(config, universe[d])
        else:
            worker(config, config.finetune_from_step)
    else:
        step_pool = list(os.listdir(checkpoint_dir))
        random.shuffle(step_pool)
        gpu_map = utils.get_gpu_map()
        with multiprocessing.Pool(len(gpu_map)) as pool:
            pool.starmap(worker, [(config, step) for step in step_pool])
