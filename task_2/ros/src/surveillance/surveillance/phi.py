import numpy as np


class Identity:
    def __call__(self, zz):
        return zz, np.ones(zz.shape)
