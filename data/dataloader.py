import torch
import numpy as np

from torch.utils.data import Subset
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import Sampler
from torch.utils.data.dataloader import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets.cifar import CIFAR10
from torchvision.datasets import MNIST

DEFAULT_DATA_DIR = "./data/dataset"
DEFAULT_NUM_WORKERS = 8

batch_size = 32

rootset_per_class = 5 
rootset_size = 50




def extract_root_set(
    dataset: Dataset,
    sample_per_class: int = rootset_per_class,
    total_num_samples: int = rootset_size,
    seed: int = None,
):
    num_classes = len(dataset.classes)
    class2sample = {i: [] for i in range(num_classes)}
    select_indices = []
    if seed == None:
        index_pool = range(len(dataset))
    else:
        index_pool = np.random.RandomState(seed=seed).permutation(
            len(dataset))
    for i in index_pool:
        current_class = dataset[i][1]
        if len(class2sample[current_class]) < sample_per_class:
            class2sample[current_class].append(i)
            select_indices.append(i)
        elif len(select_indices) == sample_per_class * num_classes:
            break
    return select_indices, class2sample


class MNISTDataLoader():
    def __init__(
        self,
        batch_size: int = batch_size,
        data_dir: str = DEFAULT_DATA_DIR,
        num_workers: int = DEFAULT_NUM_WORKERS,
        batch_sampler: Sampler = None,
        device: torch.device = 'cpu'
    ):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dims = (3, 32, 32)
        self.num_classes = 10
        self.batch_sampler = batch_sampler
        self.device = device

        mnist_normalize = transforms.Normalize((0.1307, ), (0.3081, ))

        self._train_transforms = [
            transforms.Resize(32),
            transforms.Grayscale(3),
            transforms.ToTensor(),
            mnist_normalize]

        self._test_transforms = [
            transforms.Resize(32),
            transforms.Grayscale(3),
            transforms.ToTensor(),
            mnist_normalize]

        self.prepare_data()


    def prepare_data(self):
        self.setup()
        self.train_set_loader = self.train_dataloader()
        self.test_set_loader = self.test_dataloader()
        self.root_set_loader = self.root_dataloader()


    def setup(self):
        self.train_set = MNIST(
                self.data_dir,
                train=True,
                download=True,
                transform=transforms.Compose(self._train_transforms))

        self.test_set = MNIST(
                self.data_dir,
                train=False,
                download=True,
                transform=transforms.Compose(self._test_transforms))

        self.rootset_indices, self.class2rootsample = extract_root_set(
                self.train_set)

        self.root_set = Subset(self.train_set, self.rootset_indices)

    def train_dataloader(self):
        if self.batch_sampler is None:
            return DataLoader(
                self.train_set,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=True)
        else:
            return DataLoader(
                self.train_set,
                batch_sampler=self.batch_sampler,
                num_workers=self.num_workers,
                shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.test_set,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers)

    def root_dataloader(self):
        return DataLoader(self.root_set,
                          batch_size=len(self.root_set),
                          num_workers=self.num_workers)


class CIFAR10DataLoader():
    def __init__(
        self,
        batch_size: int = batch_size,
        data_dir: str = DEFAULT_DATA_DIR,
        num_workers: int = DEFAULT_NUM_WORKERS,
        batch_sampler: Sampler = None,
        device: torch.device = 'cpu'
    ):

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dims = (3, 32, 32)
        self.num_classes = 10
        self.batch_sampler = batch_sampler

        #cifar10_normalize = transforms.Normalize((0.4914, 0.4822, 0.4465),
        #                                         (0.2023, 0.1994, 0.2010))

        #self._train_transforms = [transforms.ToTensor(), cifar10_normalize]
        self._train_transforms = [transforms.ToTensor()]
        #self._test_transforms = [transforms.ToTensor(), cifar10_normalize]
        self._test_transforms = [transforms.ToTensor()]

        self.prepare_data()


    def prepare_data(self):
        self.setup()
        self.train_set_loader = self.train_dataloader()
        self.test_set_loader = self.test_dataloader()
        self.root_set_loader = self.root_dataloader()


    def setup(self):
        self.train_set = CIFAR10(
                self.data_dir,
                train=True,
                download=True,
                transform=transforms.Compose(self._train_transforms))

        self.test_set = CIFAR10(
                self.data_dir,
                train=False,
                download=True,
                transform=transforms.Compose(self._test_transforms))

        self.rootset_indices, self.class2rootsample = extract_root_set(
                self.train_set)

        self.root_set = Subset(self.train_set, self.rootset_indices)


    def train_dataloader(self):
        if self.batch_sampler is None:
            return DataLoader(
                self.train_set,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=True)
        else:
            return DataLoader(
                self.train_set,
                batch_sampler=self.batch_sampler,
                num_workers=self.num_workers,
                shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.test_set,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers)

    def root_dataloader(self):
        return DataLoader(self.root_set,
                          batch_size=len(self.root_set),
                          num_workers=self.num_workers)