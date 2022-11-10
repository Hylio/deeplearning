from functools import partial
from collections import OrderedDict
import torch
import torch.nn as nn


def drop_path(x, drop_prob : float=0., training: bool=False):
    



