import random
from time import localtime, strftime

import numpy as np
import torch


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def get_exp_name(args):
    timestamp = strftime("%Y-%m-%d_%H:%M:%S", localtime())
    exp_name = f"{args.dataset}-{args.model}-{timestamp}"

    return exp_name
