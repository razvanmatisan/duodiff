from .celeba import get_celeba_dataloader
from .cifar10 import get_cifar10_dataloader
from .imagenet import get_imagenet_dataloader

__all__ = [get_celeba_dataloader, get_cifar10_dataloader, get_imagenet_dataloader]
