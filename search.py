""" Search cell """
import itertools
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter

import utils
from architect import Architect, SubgraphSearchOptimizer
from config import SearchConfig
from models.search_cnn import SearchCNNController
from rl import nni_tools
from visualize import plot


def main(config, writer, logger):
    logger.info("Logger is set - training start")

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

    # weights optimizer
    w_optim = optim.SGD(model.weights(), config.w_lr, momentum=config.w_momentum,
                        weight_decay=config.w_weight_decay)

    if not config.search_all_alpha:
        # alphas optimizer
        alpha_optim = optim.Adam(model.alphas(), config.alpha_lr, betas=(0.5, 0.999),
                                 weight_decay=config.alpha_weight_decay)
        # split data to train/validation
        n_train = len(train_data)
        indices = list(range(n_train))
        split = n_train // 2
        train_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[:split])
        valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[split:])
        train_loader = torch.utils.data.DataLoader(train_data,
                                                   batch_size=config.batch_size,
                                                   sampler=train_sampler,
                                                   num_workers=config.workers,
                                                   pin_memory=True)
        valid_loader = torch.utils.data.DataLoader(train_data,
                                                   batch_size=config.batch_size,
                                                   sampler=valid_sampler,
                                                   num_workers=config.workers,
                                                   pin_memory=True)
    else:
        alpha_optim = SubgraphSearchOptimizer(logger, config, model, w_optim)

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

    if config.w_lr_scheduler == "cosine":
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(w_optim, T_max=config.epochs, eta_min=config.w_lr_min)
    elif config.w_lr_scheduler == "plateau":
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(w_optim, mode="max", patience=config.w_lr_patience,
                                                            factor=config.w_lr_factor, verbose=True)
    else:
        raise NotImplementedError
    architect = Architect(model, config.w_momentum, config.w_weight_decay)

    # training loop
    best_top1 = 0.
    best_genotype = None
    final_result_reported = False
    for epoch in range(config.epochs):

        if config.cutoff_epochs is not None and epoch >= config.cutoff_epochs:
            logger.info("Cutoff epochs detected, exiting.")
            break

        lr = w_optim.param_groups[0]["lr"]
        logger.info("Current learning rate: {}".format(lr))

        if lr < config.w_lr_min:
            logger.info("Learning rate is less than {}, exiting.".format(config.w_lr_min))
            break

        if not config.search_all_alpha:
            model.print_alphas(logger)
            valid_loader_for_training = valid_loader
        else:
            # make dummy input
            valid_loader_for_training = itertools.cycle([(torch.tensor(1), torch.tensor(1))])

        # training
        train(config, writer, logger, train_loader, valid_loader_for_training,
              model, architect, w_optim, alpha_optim, lr, epoch, valid_loader)

        if config.w_lr_scheduler == "cosine":
            lr_scheduler.step()

        # validation
        if config.validate_epochs == 0 or (epoch + 1) % config.validate_epochs != 0:
            logger.info("Valid: Skipping validation for epoch {}".format(epoch + 1))
            continue

        cur_step = (epoch + 1) * len(train_loader)
        if config.search_all_alpha:
            top1 = validate_all(config, writer, logger, valid_loader, model, epoch, cur_step, alpha_optim)
            if best_top1 < top1:
                best_top1 = top1

            # checkpoint saving is not supported yet
        else:
            top1 = validate(config, writer, logger, valid_loader, model, epoch, cur_step)

            # log
            # genotype
            genotype = model.genotype()
            logger.info("genotype = {}".format(genotype))

            # genotype as a image
            plot_path = os.path.join(config.plot_path, "EP{:02d}".format(epoch + 1))
            caption = "Epoch {}".format(epoch + 1)
            plot(genotype.normal, plot_path + "-normal", caption)

            # save
            if best_top1 < top1:
                best_top1 = top1
                best_genotype = genotype
                is_best = True
            else:
                is_best = False
            utils.save_checkpoint(model, config.path, is_best)

        if config.nni:
            nni_tools.report_result(top1, epoch + 1 == config.epochs)
            if epoch + 1 == config.epochs:
                final_result_reported = True

        if config.w_lr_scheduler == "plateau":
            lr_scheduler.step(top1)
        print("")

    if config.nni and not final_result_reported:
        try:
            nni_tools.report_result(top1, True)
        except:
            logger.warning("Final result not reported and top1 not found")

    logger.info("Final best Prec@1 = {:.4%}".format(best_top1))
    if best_genotype is not None:
        logger.info("Best Genotype = {}".format(best_genotype))
    print("")


def train(config, writer, logger, train_loader, valid_loader, model, architect, w_optim, alpha_optim, lr, epoch,
          valid_loader_for_testing):
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    losses = utils.AverageMeter()

    cur_step = epoch * len(train_loader)
    writer.add_scalar('train/lr', lr, cur_step)

    for step, ((trn_X, trn_y), (val_X, val_y)) in enumerate(zip(train_loader, valid_loader)):
        trn_X, trn_y = trn_X.cuda(non_blocking=True), trn_y.cuda(non_blocking=True)
        val_X, val_y = val_X.cuda(non_blocking=True), val_y.cuda(non_blocking=True)
        N = trn_X.size(0)

        model.train()

        current_archit = None
        if isinstance(alpha_optim, optim.Optimizer):
            # phase 2. architect step (alpha)
            alpha_optim.zero_grad()
            architect.unrolled_backward(trn_X, trn_y, val_X, val_y, lr, w_optim)
            alpha_optim.step()
        else:
            # process as a SubgraphSearchOptimizer
            # the optimizer will automatically do save and restore and other things...
            if config.designated_training:
                while True:
                    current_archit = next(alpha_optim)
                    if current_archit in config.designated_training:
                        break
            else:
                current_archit = next(alpha_optim)

        # phase 1. child network step (w)
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

        will_validate = config.dense_validation_steps > 0 and \
                        (cur_step + config.dense_validation_steps >= config.epochs * len(train_loader) or
                         (config.cutoff_epochs is not None and
                          cur_step + config.dense_validation_steps >= config.cutoff_epochs * len(train_loader)))

        if will_validate or step % config.print_freq == 0 or step == len(train_loader) - 1:
            current_archit_info = "Archit: {} ".format(current_archit) if current_archit is not None else ""
            logger.info(
                "Train: [{:2d}/{}] Step {:03d}/{:03d} {}Loss {losses.avg:.3f} "
                "Prec@1 {top1.val:.1%} ({top1.avg:.1%}) Prec@5 {top5.val:.1%} ({top5.avg:.1%})".format(
                    epoch + 1, config.epochs, step, len(train_loader) - 1, current_archit_info,
                    losses=losses, top1=top1, top5=top5))

        if will_validate:
            logger.info("Train: [{:2d}/{}] Step {:03d}/{:03d} {}Entering dense validation steps. Validating...".format(
                epoch + 1, config.epochs, step, len(train_loader) - 1, current_archit_info
            ))
            validate_all(config, writer, logger, valid_loader_for_testing, model, epoch, cur_step, alpha_optim)

        writer.add_scalar('train/loss', loss.item(), cur_step)
        writer.add_scalar('train/top1', prec1.item(), cur_step)
        writer.add_scalar('train/top5', prec5.item(), cur_step)
        cur_step += 1

    logger.info("Train: [{:2d}/{}] Final Prec@1 {:.4%}".format(epoch + 1, config.epochs, top1.avg))


def validate_all(config, writer, logger, valid_loader, model, epoch, cur_step, alpha_optim):
    # save state of training
    last_index = alpha_optim.save()

    if config.save_weights_on_validation:
        checkpoint_dir = os.path.join(config.checkpoint_path, "{:06d}".format(cur_step))
        os.makedirs(checkpoint_dir, exist_ok=True)

    validate_list = dict()
    for val_idx in alpha_optim.validate_instances:
        instance = alpha_optim[val_idx]  # init the model to the state
        if config.save_weights_on_validation:
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, instance + ".pth.tar"))

        validate_list[instance] = validate(config, writer, logger, valid_loader,
                                           model, epoch, cur_step, instance)
    sorted_top = sorted(list(validate_list.items()), key=lambda d: d[1], reverse=True)
    accuracy_list = [t[1] for t in sorted_top]
    writer.add_histogram("val/hist", accuracy_list, cur_step)
    writer.add_scalar("val/best_top1", accuracy_list[0], cur_step)
    logger.info("Valid: Top accuracy: {}".format(sorted_top))
    top1 = sorted_top[0][1]

    # restore state of training
    alpha_optim.restore(last_index)

    return top1


def validate(config, writer, logger, valid_loader, model, epoch, cur_step, instance=None, total_epochs=None):
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    losses = utils.AverageMeter()
    current_archit_info = "Archit: {} ".format(instance) if instance is not None else ""
    if total_epochs is None:
        total_epochs = config.epochs

    model.eval()

    with torch.no_grad():
        for step, (X, y) in enumerate(valid_loader):
            X, y = X.cuda(non_blocking=True), y.cuda(non_blocking=True)
            N = X.size(0)

            logits = model(X)
            loss = model.criterion(logits, y)

            prec1, prec5 = utils.accuracy(logits, y, topk=(1, 5))
            losses.update(loss.item(), N)
            top1.update(prec1.item(), N)
            top5.update(prec5.item(), N)

            if step % config.print_freq == 0 or step == len(valid_loader) - 1:
                logger.info(
                    "Valid: [{:2d}/{}] Step {:03d}/{:03d} {}Loss {losses.avg:.3f} "
                    "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(
                        epoch + 1, total_epochs, step, len(valid_loader) - 1, current_archit_info,
                        losses=losses, top1=top1, top5=top5))

    suffix = "/" + instance if instance is not None else ""
    writer.add_scalar('val/loss' + suffix, losses.avg, cur_step)
    writer.add_scalar('val/top1' + suffix, top1.avg, cur_step)
    writer.add_scalar('val/top5' + suffix, top5.avg, cur_step)

    logger.info("Valid: [{:2d}/{}] {}Final Prec@1 {:.4%}".format(epoch + 1, total_epochs,
                                                                 current_archit_info, top1.avg))

    return top1.avg


def get_current_node_count():
    if "PAI_CURRENT_TASK_ROLE_NAME" not in os.environ:
        return 1
    task_role = os.environ["PAI_CURRENT_TASK_ROLE_NAME"]
    return int(os.environ["PAI_TASK_ROLE_TASK_COUNT_" + task_role])


def get_current_node_index():
    if "PAI_CURRENT_TASK_ROLE_CURRENT_TASK_INDEX" not in os.environ:
        return 0
    return int(os.environ["PAI_CURRENT_TASK_ROLE_CURRENT_TASK_INDEX"])


if __name__ == "__main__":
    config = SearchConfig()

    if config.nni:
        if config.nni == "gt_mock":
            nni_tools.mock_result()
        else:
            config.designated_subgraph = [nni_tools.get_param()]
            config.path = nni_tools.get_output_dir()

            # tensorboard
            writer = SummaryWriter(log_dir=os.path.join(config.path, "tb"))
            writer.add_text('config', config.as_markdown(), 0)

            logger = utils.get_logger(os.path.join(config.path, "{}.log".format(config.name)))
            config.print_params(logger.info)

            main(config, writer, logger)

            writer.close()

    elif config.shared_policy == "group":
        config.shared_policy = "all"
        if "designated_partition" in config.shared_policy_kwargs:
            mapping = config.shared_policy_kwargs["designated_partition"]
            group_number = max(mapping) + 1
        else:
            random.seed(config.seed)
            total = config.shared_policy_kwargs["total"]
            group_number = config.shared_policy_kwargs["groups"]
            mapping = [i % group_number for i in range(total)]
            if "stable_partition" in config.shared_policy_kwargs:
                stable_partition = config.shared_policy_kwargs["stable_partition"]
                if isinstance(stable_partition, int):
                    seed = config.shared_policy_kwargs["stable_partition"]
                    print("Using stable partition, seed={}".format(seed))
                    random_state = np.random.RandomState(seed)
                    random_state.shuffle(mapping)
                elif stable_partition == "sequential":
                    mapping = [i // (total // group_number) for i in range(total)]
                else:
                    raise NotImplementedError("Stable partition not understood")
            else:
                random.shuffle(mapping)
        node_count = get_current_node_count()
        node_index = get_current_node_index()

        assert len(set(mapping)) == group_number
        for group_id in range(group_number):
            if group_id % node_count != node_index:
                continue

            config.designated_subgraph = [i for i, t in enumerate(mapping) if t == group_id]

            # tensorboard
            writer = SummaryWriter(log_dir=os.path.join(config.path, "tb_%d" % group_id))
            writer.add_text('config', config.as_markdown(), 0)

            logger = utils.get_logger(os.path.join(config.path, "{}_{}.log".format(config.name, group_id)), group_id)
            config.print_params(logger.info)
            logger.info("Running as role: %d/%d" % (node_index, node_count))
            logger.info("Selected subgraphs: {}".format(config.designated_subgraph))

            main(config, writer, logger)

            writer.close()
    else:
        # tensorboard
        writer = SummaryWriter(log_dir=os.path.join(config.path, "tb"))
        writer.add_text('config', config.as_markdown(), 0)

        logger = utils.get_logger(os.path.join(config.path, "{}.log".format(config.name)))
        config.print_params(logger.info)

        main(config, writer, logger)

        writer.close()
