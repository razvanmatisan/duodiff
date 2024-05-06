import os
import unittest

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from data_utils import cifar10_utils, celebA_utils


class TestDataUtils(unittest.TestCase):
    def test_cifar10(self, batch_size=16):
        dataloader = cifar10_utils.get_dataloader("../data/cifar10/")

        x, _ = next(iter(dataloader))
        self.assertEqual(x.shape, torch.Size([batch_size, 3, 32, 32]))

        # plt.imshow(np.transpose(x[0].numpy(), (1, 2, 0)))
        # plt.axis('off')
        # plt.show()

    def test_celebA(self, batch_size=4):
        dataloader = celebA_utils.get_dataloader("../data/", batch_size)

        x, _ = next(iter(dataloader))
        self.assertEqual(x.shape, torch.Size([batch_size, 3, 64, 64]))

        # plt.imshow(np.transpose(x[0].numpy(), (1, 2, 0)))
        # plt.axis('off')
        # plt.show()


if __name__ == "__main__":
    unittest.main()
