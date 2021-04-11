""" Utilities """
import os
import logging
import shutil
import torch
import torchvision.datasets as dset
import numpy as np
import torch.nn as nn
device = torch.device("cuda")

def get_logger(file_path):
    """ Make python logger """
    # [!] Since tensorboardX use default logger (e.g. logging.info()), we should use custom logger
    logger = logging.getLogger('darts')
    log_format = '%(asctime)s | %(message)s'
    formatter = logging.Formatter(log_format, datefmt='%m/%d %I:%M:%S %p')
    file_handler = logging.FileHandler(file_path)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.setLevel(logging.INFO)

    return logger


def param_size(model):
    """ Compute parameter size in MB """
    n_params = sum(
        np.prod(v.size()) for k, v in model.named_parameters() if not k.startswith('aux_head'))
    return n_params / 1024. / 1024.


class AverageMeter():
    """ Computes and stores the average and current value """
    def __init__(self):
        self.reset()

    def reset(self):
        """ Reset all statistics """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 1 # avoid the count of some calsses in the first batch is zero

    def update(self, val, n):
        """ Update statistics """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def evaluate(logits, label):
    logits = logits.astype(np.float32)
    label = label.astype(np.float32)
    inter = np.dot(logits.flatten(), label.flatten())
    union = np.sum(logits) + np.sum(label)
    dice = (2 * inter + 1e-5) / (union + 1e-5)
    return dice

def save_results(results, path):
    
    filename = os.path.join(path, 'final_results.txt')
    f = open(filename, 'a')
    f.write('Best dice: {:.5f}\n'.format(results))
    

def save_checkpoint(state, ckpt_dir, is_best=False):
    filename = os.path.join(ckpt_dir, 'checkpoint.pth.tar')
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(ckpt_dir, 'best.pth.tar')
        shutil.copyfile(filename, best_filename)
        
class log_loss(nn.Module):
    def __init__(self, w_dice = 0.5, w_cross = 0.5):
        super(log_loss, self).__init__()
        self.w_dice = w_dice
        self.w_cross = w_cross
    def forward(self, logits, label, smooth = 1.):
        
        area_union = torch.sum(logits * label, dim = (0,2,3), keepdim = True)
        area_logits = torch.sum(logits, dim = (0,2,3), keepdim = True)
        area_label = torch.sum(label, dim = (0,2,3), keepdim = True)
        in_dice = torch.mean(torch.pow((-1) * torch.log((2 * area_union + 1e-7)/(area_logits + area_label + smooth)), 0.3))
        return in_dice
