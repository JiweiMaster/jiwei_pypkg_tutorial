import time 
from PIL import Image, ImageEnhance
import numpy as np
import torch.nn.functional as F
import torch
import os
import time
import scipy.io as scio
import random
import itertools
from scipy import misc
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as data
import math
from scipy.ndimage.filters import gaussian_filter, median_filter
import multiprocessing



# 获取当前格式化的时间
def getNowTime(formatter = '%Y-%m-%d_%H_%M_%S'):
    return time.strftime(formatter, time.localtime())

def seed():
    return np.random.rand()*0.1+0.9







    







