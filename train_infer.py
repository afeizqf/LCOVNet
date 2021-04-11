""" Training augmented model """
import os
import torch
import torch.nn as nn
import numpy as np
from tensorboardX import SummaryWriter
from ptflops import get_model_complexity_info
import utils
import data_generator_3D as data_generator_3D
import time
import SimpleITK as sitk
import sys
from config import TrainConfig
from model import LCOVNet
from apex import amp


config = TrainConfig()

device = torch.device("cuda")

# tensorboard
writer = SummaryWriter(log_dir=os.path.join(config.path, "tb"))
writer.add_text('config', config.as_markdown(), 0)

logger = utils.get_logger(os.path.join(config.path, "{}.log".format(config.name)))
config.print_params(logger.info)

def main():
    logger.info("Logger is set - training start")

    # set default gpu device id
    torch.cuda.set_device(config.gpus[0])

    # set seed
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    torch.backends.cudnn.benchmark = True


    criterion = utils.log_loss().to(device)
    d = torch.device(type='cuda', index=config.gpus[0])
    model = LCOVNet(config.input_channels, config.n_classes).to(device=d)
    with torch.cuda.device(config.gpus[0]):
        net = model
        macs, params = get_model_complexity_info(net, (1, 240, 160, 48), as_strings=True,
                                                 print_per_layer_stat=True, verbose=True)
        logger.info("{:<30}  {:<8}".format('Computational complexity: ', macs))
        logger.info("{:<30}  {:<8}".format('Number of parameters: ', params))

    # model size
    mb_params = utils.param_size(model)
    logger.info("Model size = {:.3f} MB".format(mb_params))
    # weights optimizer
    optimizer = torch.optim.SGD(model.parameters(), config.lr, momentum=config.momentum,
                                weight_decay=config.weight_decay)
    model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

    train_loader = data_generator_3D.Covid19TrainSet()
    valid_loader = data_generator_3D.Covid19EvalSet()
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config.epochs)

    best_dice = 0.
    # training loop
    summ_writer = SummaryWriter(config.training_summary_dir)
    for epoch in range(config.epochs):

        # training
        train(train_loader, model, optimizer, criterion, epoch, summ_writer)
        lr_scheduler.step()
        # validation
        cur_step = (epoch+1) * len(train_loader)
        mean_dice = validate(valid_loader, model, criterion, epoch, summ_writer, best_dice)

        # save
        if best_dice < mean_dice:
            best_dice = mean_dice
            is_best = True
        else:
            is_best = False
        utils.save_checkpoint(model, config.path, is_best)
        print("")

    logger.info("Final best Dice = {:.4%}".format(best_dice))
    utils.save_results(best_dice, config.path)
    summ_writer.close()

def train(train_loader, model, optimizer, criterion, epoch, summ_writer):
    losses = utils.AverageMeter()
    cur_step = epoch*len(train_loader)
    cur_lr = optimizer.param_groups[0]['lr']
    logger.info("Epoch {} LR {}".format(epoch, cur_lr))
    writer.add_scalar('train/lr', cur_lr, cur_step)
    model.train()
    #all_dice = np.empty().astype(np.float32)
    all_dice = []
    for step, (name, X, y) in enumerate(train_loader):
        X, y = torch.from_numpy(X).to(device, non_blocking=True), torch.from_numpy(y).to(device, non_blocking=True)
        N = X.size(0)

        optimizer.zero_grad()
        logits = model(X)

        loss = criterion(logits, y)
        #loss.backward()
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        # gradient clipping
        nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()

        losses.update(loss.item(), N)

        if step % config.print_freq == 0 or step == len(train_loader)-1:
           logger.info(
                "Train: [{:2d}/{}] Step {:03d}/{:03d} Loss {:.3f} ".format(
                    epoch+1, config.epochs, step, len(train_loader), losses.avg,
                    ))

        writer.add_scalar('train/loss', loss.item(), cur_step)

        logits[logits >= 0.5] = 1
        logits[logits < 0.5] = 0
        predict = logits.cpu().detach().numpy()
        y = y.cpu().detach().numpy()
        dice_i = utils.evaluate(predict, y)
        all_dice.append(dice_i)
        cur_step += 1
    dice_mean = 0

    for i in all_dice:
        dice_mean += i/len(all_dice)

    train_avg_loss = losses.avg
    train_avg_dice = dice_mean
    loss_scalers = {'train': train_avg_loss}
    summ_writer.add_scalars('loss', loss_scalers, epoch + 1)

    dice_scalers = {'train': train_avg_dice}
    summ_writer.add_scalars('avg_dice', dice_scalers, epoch + 1)

    if (epoch+1) % 50 == 0:
        chpt_prefx = config.training_checkpoint_prefix
        save_dict = {'epoch': epoch + 1,
                     'model_state_dict': model.state_dict(),
                     'optimizer_state_dict': optimizer.state_dict(),
                     'amp': amp.state_dict()}
        save_name = "{0:}_{1:}.pt".format(chpt_prefx, epoch + 1)
        torch.save(save_dict, save_name)
    print("train_avg_loss", train_avg_loss)
    print("train_avg_dice", train_avg_dice)

def validate(valid_loader, model, criterion, epoch, summ_writer, best_dice):
    losses = utils.AverageMeter()

    model.eval()
    all_dice = np.zeros([len(valid_loader)]).astype(np.float32)
    all_dice = []
    totel_time = 0
    start_time = time.time()
    size_z = 48
    with torch.no_grad():
        for i, (name, image, label) in enumerate(valid_loader):
            image = torch.from_numpy(image)
            predict = np.zeros(shape=label.shape, dtype=label.dtype)
            z = image.shape[4]
            m = z // size_z if z % size_z == 0 else z // size_z + 1
            start_time = time.time()
            for k in range(m):
                if (k+1)*size_z <= z:
                    max_z = (k+1)*size_z
                else:
                    max_z = z
                min_z = max_z - size_z
                image_k = image[:, :, :, :, min_z:max_z].float().to(device, non_blocking=True)
                predict_k = model(image_k)
                predict_k[predict_k >= 0.5] = 1
                predict_k[predict_k < 0.5] = 0
                predict[:, :, :, :, min_z:max_z] = predict_k.cpu().detach().numpy()
            totel_time = totel_time + time.time() - start_time
            all_dice.append(utils.evaluate(predict, label))

    dice_len = len(all_dice)
    dice_np = np.empty(shape=[dice_len])
    #list_image = []
    for i in range(dice_len):
        dice_np[i] = all_dice[i]
        logger.info("{}  dice: {:.4%} ".format(i, all_dice[i]))
    logger.info("mean: {}".format(dice_np.mean()))
    logger.info("std : {}".format(dice_np.std()))

    if best_dice < dice_np.mean():
        chpt_prefx = config.validing_checkpoint_prefix
        save_dict = {'epoch': epoch + 1,
                     'model_state_dict': model.state_dict(),
                     'amp': amp.state_dict()}
        fname = "{}/best.pt".format(chpt_prefx)
        if os.path.isfile(fname):
            os.remove(fname)
        save_name = "{}/best.pt".format(chpt_prefx)
        torch.save(save_dict, save_name)

    dice_scalers = {'vadil': dice_np.mean()}
    summ_writer.add_scalars('vadil_avg_dice', dice_scalers, epoch + 1)

    avg_time = totel_time / dice_len
    logger.info("average testing time : {}".format(avg_time))

    mean_dice = np.mean(all_dice, axis = 0)
    writer.add_scalar('val/dice', mean_dice, epoch)
    writer.add_scalar('val/loss', losses.avg, epoch)
    logger.info("Valid: [{:2d}/{}] average dice: {:.4%} ".format(epoch+1, config.epochs, mean_dice))

    return mean_dice



def save_nd_array_as_image(data, image_name, reference_name = None):
    """
    save a 3D or 2D numpy array as medical image or RGB image
    inputs:
        data: a numpy array with shape [D, H, W] or [C, H, W]
        image_name: the output file name
    outputs: None
    """
    data_dim = len(data.shape)
    assert(data_dim == 2 or data_dim == 3)
    if (image_name.endswith(".nii.gz") or image_name.endswith(".nii") or
        image_name.endswith(".mha")):
        assert(data_dim == 3)
        save_array_as_nifty_volume(data, image_name, reference_name)

def save_array_as_nifty_volume(data, image_name, reference_name = None):
    """
    save a numpy array as nifty image
    inputs:
        data: a numpy array with shape [Depth, Height, Width]
        image_name: the ouput file name
        reference_name: file name of the reference image of which affine and header are used
    outputs: None
    """
    img = sitk.GetImageFromArray(data)
    if(reference_name is not None):
        img_ref = sitk.ReadImage(reference_name)
        img.CopyInformation(img_ref)
    sitk.WriteImage(img, image_name)


if __name__ == "__main__":
    main()
