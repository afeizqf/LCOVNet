""" Config class for training """
import argparse
import os
from functools import partial
import torch


def get_parser(name):
    """ make default formatted parser """
    parser = argparse.ArgumentParser(name, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # print default value always
    parser.add_argument = partial(parser.add_argument, help=' ')
    return parser


def parse_gpus(gpus):
    if gpus == 'all':
        return list(range(torch.cuda.device_count()))
    else:
        return [int(s) for s in gpus.split(',')]


class BaseConfig(argparse.Namespace):
    def print_params(self, prtf=print):
        prtf("")
        prtf("Parameters:")
        for attr, value in sorted(vars(self).items()):
            prtf("{}={}".format(attr.upper(), value))
        prtf("")

    def as_markdown(self):
        """ Return configs as markdown format """
        text = "|name|value|  \n|-|-|  \n"
        for attr, value in sorted(vars(self).items()):
            text += "|{}|{}|  \n".format(attr, value)

        return text


class TrainConfig(BaseConfig):
    def build_parser(self):
        parser = get_parser("Train config")
        parser.add_argument('--name', default="/home/ubuntu/zhaoqianfei/UNet_Mini/LACB_Net_A/log")
        parser.add_argument('--batch_size', type=int, default=1, help='batch size')
        parser.add_argument('--input_channels', type=int, default=1, help='input channels')
        parser.add_argument('--n_classes', type=int, default=1, help='number classes')
        parser.add_argument('--lr', type=float, default=0.0025, help='lr for weights')
        parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
        parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
        parser.add_argument('--grad_clip', type=float, default=5.,
                            help='gradient clipping for weights')
        parser.add_argument('--print_freq', type=int, default=20, help='print frequency')
        parser.add_argument('--gpus', default='0', help='gpu device ids separated by comma. '
                                                        '`all` indicates use all gpus.')
        parser.add_argument('--epochs', type=int, default=400, help='# of training epochs')
        parser.add_argument('--init_channels', type=int, default=12)
        parser.add_argument('--seed', type=int, default=2, help='random seed')
        parser.add_argument('--workers', type=int, default=4, help='# of workers')
        parser.add_argument('--training_summary_dir', default=".../model/unet")
        parser.add_argument('--training_checkpoint_prefix', default=".../model/unet")
        parser.add_argument('--testing_checkpoint_name', default=".../model/unet_400.pt")
        parser.add_argument('--testing_output_dir', default=".../result")
        parser.add_argument('--root_dir', default=".../data_train_test/lalel")
        parser.add_argument('--validing_checkpoint_prefix', default=".../model")

        return parser

    def __init__(self):
        parser = self.build_parser()
        args = parser.parse_args()
        super().__init__(**vars(args))

        self.path = os.path.join('train', self.name)
        self.gpus = parse_gpus(self.gpus)
