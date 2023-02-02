import numpy as np
import matplotlib.pyplot as plt
import argparse

from omegaconf import OmegaConf
from matplotlib import cm
from tqdm import tqdm


def color_depthmap(grey, maxval=None, minval=None):

    if minval is None:
        minval = np.amin(grey)
    if maxval is None:
        maxval = np.amax(grey)

    grey -= minval
    grey[grey < 0] = 0
    grey /= maxval

    rgb = cm.get_cmap(plt.get_cmap('jet'))(grey)[:, :, :3]

    return rgb