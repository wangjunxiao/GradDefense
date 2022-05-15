"""Repeatable code parts concerning data loading."""


import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize, RandomCrop
import torch.utils.data as data

from PIL import Image
from six.moves import urllib
import tarfile
import os
from os import makedirs, remove, listdir
from os.path import exists, join, basename

from .loss import Classification, PSNR


MULTITHREAD_DATAPROCESSING = 4
PIN_MEMORY = True

cifar10_mean = [0.4914672374725342, 0.4822617471218109, 0.4467701315879822]
cifar10_std = [0.24703224003314972, 0.24348513782024384, 0.26158785820007324]
cifar100_mean = [0.5071598291397095, 0.4866936206817627, 0.44120192527770996]
cifar100_std = [0.2673342823982239, 0.2564384639263153, 0.2761504650115967]
mnist_mean = (0.13066373765468597,)
mnist_std = (0.30810782313346863,)
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]


def load_dataset(dataset, data_path, batchsize=128, augmentations=True, shuffle=True, normalize=True):
    """Return a dataloader with given dataset and augmentation, normalize data?."""
    path = os.path.expanduser(data_path)

    if dataset == 'CIFAR10':
        trainset, validset = _build_cifar10(path, augmentations, normalize)
        loss_fn = Classification()
        dm = cifar10_mean
        ds = cifar10_std
    elif dataset == 'CIFAR100':
        trainset, validset = _build_cifar100(path, augmentations, normalize)
        loss_fn = Classification()
        dm = cifar100_mean
        ds = cifar100_std
    elif dataset == 'MNIST':
        trainset, validset = _build_mnist(path, augmentations, normalize)
        loss_fn = Classification()
        dm = mnist_mean
        ds = mnist_std
    elif dataset == 'Fashion-MNIST':
        trainset, validset = _build_fashion_mnist(path, augmentations, normalize)
        loss_fn = Classification()
        dm = mnist_mean
        ds = mnist_std
    elif dataset == 'MNIST_GRAY':
        trainset, validset = _build_mnist_gray(path, augmentations, normalize)
        loss_fn = Classification()
    elif dataset == 'ImageNet':
        trainset, validset = _build_imagenet(path, augmentations, normalize)
        loss_fn = Classification()
        dm = imagenet_mean
        ds = imagenet_std
    elif dataset == 'BSDS-SR':
        trainset, validset = _build_bsds_sr(path, augmentations, normalize, upscale_factor=3, RGB=True)
        loss_fn = PSNR()
    elif dataset == 'BSDS-DN':
        trainset, validset = _build_bsds_dn(path, augmentations, normalize, noise_level=25 / 255, RGB=False)
        loss_fn = PSNR()
    elif dataset == 'BSDS-RGB':
        trainset, validset = _build_bsds_dn(path, augmentations, normalize, noise_level=25 / 255, RGB=True)
        loss_fn = PSNR()

    if MULTITHREAD_DATAPROCESSING:
        num_workers = min(torch.get_num_threads(), MULTITHREAD_DATAPROCESSING) if torch.get_num_threads() > 1 else 0
    else:
        num_workers = 0

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=min(batchsize, len(trainset)),
                                              shuffle=shuffle, drop_last=True, num_workers=num_workers, pin_memory=PIN_MEMORY)
    validloader = torch.utils.data.DataLoader(validset, batch_size=min(batchsize, len(validset)),
                                              shuffle=False, drop_last=False, num_workers=num_workers, pin_memory=PIN_MEMORY)

    return dm, ds, loss_fn, trainloader, validloader


def _build_cifar10(data_path, augmentations=True, normalize=True):
    """Define CIFAR-10 with everything considered."""
    # Load data
    trainset = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True, transform=transforms.ToTensor())
    validset = torchvision.datasets.CIFAR10(root=data_path, train=False, download=True, transform=transforms.ToTensor())

    if cifar10_mean is None:
        data_mean, data_std = _get_meanstd(trainset)
    else:
        data_mean, data_std = cifar10_mean, cifar10_std

    # Organize preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(data_mean, data_std) if normalize else transforms.Lambda(lambda x: x)])
    if augmentations:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transform])
        trainset.transform = transform_train
    else:
        trainset.transform = transform
    validset.transform = transform

    return trainset, validset


def _build_cifar100(data_path, augmentations=True, normalize=True):
    """Define CIFAR-100 with everything considered."""
    # Load data
    trainset = torchvision.datasets.CIFAR100(root=data_path, train=True, download=True, transform=transforms.ToTensor())
    validset = torchvision.datasets.CIFAR100(root=data_path, train=False, download=True, transform=transforms.ToTensor())

    if cifar100_mean is None:
        data_mean, data_std = _get_meanstd(trainset)
    else:
        data_mean, data_std = cifar100_mean, cifar100_std

    # Organize preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(data_mean, data_std) if normalize else transforms.Lambda(lambda x: x)])
    if augmentations:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transform])
        trainset.transform = transform_train
    else:
        trainset.transform = transform
    validset.transform = transform

    return trainset, validset

def _build_mnist(data_path, augmentations=True, normalize=True):
    """Define MNIST with everything considered."""
    # Load data
    trainset = torchvision.datasets.MNIST(root=data_path, train=True, download=True, transform=transforms.ToTensor())
    validset = torchvision.datasets.MNIST(root=data_path, train=False, download=True, transform=transforms.ToTensor())

    if mnist_mean is None:
        cc = torch.cat([trainset[i][0].reshape(-1) for i in range(len(trainset))], dim=0)
        data_mean = (torch.mean(cc, dim=0).item(),)
        data_std = (torch.std(cc, dim=0).item(),)
    else:
        data_mean, data_std = mnist_mean, mnist_std

    # Organize preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(data_mean, data_std) if normalize else transforms.Lambda(lambda x: x)])
    if augmentations:
        transform_train = transforms.Compose([
            transforms.RandomCrop(28, padding=4),
            transforms.RandomHorizontalFlip(),
            transform])
        trainset.transform = transform_train
    else:
        trainset.transform = transform
    validset.transform = transform

    return trainset, validset

def _build_fashion_mnist(data_path, augmentations=True, normalize=True):
    # Load data
    trainset = torchvision.datasets.FashionMNIST(root=data_path, train=True, download=True, transform=transforms.ToTensor())
    validset = torchvision.datasets.FashionMNIST(root=data_path, train=False, download=True, transform=transforms.ToTensor())

    cc = torch.cat([trainset[i][0].reshape(-1) for i in range(len(trainset))], dim=0)
    data_mean = (torch.mean(cc, dim=0).item(),)
    data_std = (torch.std(cc, dim=0).item(),)
    print(data_mean, data_mean)
    
    # Organize preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(data_mean, data_std) if normalize else transforms.Lambda(lambda x: x)])
    if augmentations:
        transform_train = transforms.Compose([
            transforms.RandomCrop(28, padding=4),
            transforms.RandomHorizontalFlip(),
            transform])
        trainset.transform = transform_train
    else:
        trainset.transform = transform
    validset.transform = transform

    return trainset, validset
        
def _build_mnist_gray(data_path, augmentations=True, normalize=True):
    """Define MNIST with everything considered."""
    # Load data
    trainset = torchvision.datasets.MNIST(root=data_path, train=True, download=True, transform=transforms.ToTensor())
    validset = torchvision.datasets.MNIST(root=data_path, train=False, download=True, transform=transforms.ToTensor())

    if mnist_mean is None:
        cc = torch.cat([trainset[i][0].reshape(-1) for i in range(len(trainset))], dim=0)
        data_mean = (torch.mean(cc, dim=0).item(),)
        data_std = (torch.std(cc, dim=0).item(),)
    else:
        data_mean, data_std = mnist_mean, mnist_std

    # Organize preprocessing
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(data_mean, data_std) if normalize else transforms.Lambda(lambda x: x)])
    if augmentations:
        transform_train = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.RandomCrop(28, padding=4),
            transforms.RandomHorizontalFlip(),
            transform])
        trainset.transform = transform_train
    else:
        trainset.transform = transform
    validset.transform = transform

    return trainset, validset


def _build_imagenet(data_path, augmentations=True, normalize=True):
    """Define ImageNet with everything considered."""
    # Load data
    trainset = torchvision.datasets.ImageNet(root=data_path, split='train', transform=transforms.ToTensor())
    validset = torchvision.datasets.ImageNet(root=data_path, split='val', transform=transforms.ToTensor())

    if imagenet_mean is None:
        data_mean, data_std = _get_meanstd(trainset)
    else:
        data_mean, data_std = imagenet_mean, imagenet_std
    
    data_mean, data_std = imagenet_mean, imagenet_std

    # Organize preprocessing
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(data_mean, data_std) if normalize else transforms.Lambda(lambda x : x)])
    if augmentations:
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(data_mean, data_std) if normalize else transforms.Lambda(lambda x : x)])
        trainset.transform = transform_train
    else:
        trainset.transform = transform
    validset.transform = transform

    return validset, validset


def _get_meanstd(trainset):
    cc = torch.cat([trainset[i][0].reshape(3, -1) for i in range(len(trainset))], dim=1)
    data_mean = torch.mean(cc, dim=1).tolist()
    data_std = torch.std(cc, dim=1).tolist()
    return data_mean, data_std



def _build_bsds_sr(data_path, augmentations=True, normalize=True, upscale_factor=3, RGB=True):
    root_dir = _download_bsd300(dest=data_path)
    train_dir = join(root_dir, "train")
    crop_size = _calculate_valid_crop_size(256, upscale_factor)
    print(f'Crop size is {crop_size}. Upscaling factor is {upscale_factor} in mode {RGB}.')

    trainset = DatasetFromFolder(train_dir, replicate=200,
                                 input_transform=_input_transform(crop_size, upscale_factor),
                                 target_transform=_target_transform(crop_size), RGB=RGB)

    test_dir = join(root_dir, "test")
    validset = DatasetFromFolder(test_dir, replicate=200,
                                 input_transform=_input_transform(crop_size, upscale_factor),
                                 target_transform=_target_transform(crop_size), RGB=RGB)
    return trainset, validset


def _build_bsds_dn(data_path, augmentations=True, normalize=True, upscale_factor=1, noise_level=25 / 255, RGB=True):
    root_dir = _download_bsd300(dest=data_path)
    train_dir = join(root_dir, "train")

    crop_size = _calculate_valid_crop_size(256, upscale_factor)
    patch_size = 64
    print(f'Crop size is {crop_size} for patches of size {patch_size}. '
          f'Upscaling factor is {upscale_factor} in mode RGB={RGB}.')

    trainset = DatasetFromFolder(train_dir, replicate=200,
                                 input_transform=_input_transform(crop_size, upscale_factor, patch_size=patch_size),
                                 target_transform=_target_transform(crop_size, patch_size=patch_size),
                                 noise_level=noise_level, RGB=RGB)

    test_dir = join(root_dir, "test")
    validset = DatasetFromFolder(test_dir, replicate=200,
                                 input_transform=_input_transform(crop_size, upscale_factor),
                                 target_transform=_target_transform(crop_size),
                                 noise_level=noise_level, RGB=RGB)
    return trainset, validset


def _download_bsd300(dest="dataset"):
    output_image_dir = join(dest, "BSDS300/images")

    if not exists(output_image_dir):
        makedirs(dest, exist_ok=True)
        url = "http://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/BSDS300-images.tgz"
        print("downloading url ", url)

        data = urllib.request.urlopen(url)

        file_path = join(dest, basename(url))
        with open(file_path, 'wb') as f:
            f.write(data.read())

        print("Extracting data")
        with tarfile.open(file_path) as tar:
            for item in tar:
                tar.extract(item, dest)

        remove(file_path)

    return output_image_dir


def _calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)


def _input_transform(crop_size, upscale_factor, patch_size=None):
    return Compose([
        CenterCrop(crop_size),
        Resize(crop_size // upscale_factor),
        RandomCrop(patch_size if patch_size is not None else crop_size // upscale_factor),
        ToTensor(),
    ])


def _target_transform(crop_size, patch_size=None):
    return Compose([
        CenterCrop(crop_size),
        RandomCrop(patch_size if patch_size is not None else crop_size),
        ToTensor(),
    ])







class DatasetFromFolder(data.Dataset):
    """Generate an image-to-image dataset from images from the given folder."""

    def __init__(self, image_dir, replicate=1, input_transform=None, target_transform=None, RGB=True, noise_level=0.0):
        """Init with directory, transforms and RGB switch."""
        super(DatasetFromFolder, self).__init__()
        self.image_filenames = [join(image_dir, x) for x in listdir(image_dir) if _is_image_file(x)]

        self.input_transform = input_transform
        self.target_transform = target_transform

        self.replicate = replicate
        self.classes = [None]
        self.RGB = RGB
        self.noise_level = noise_level

    def __getitem__(self, index):
        """Index into dataset."""
        input = _load_img(self.image_filenames[index % len(self.image_filenames)], RGB=self.RGB)
        target = input.copy()
        if self.input_transform:
            input = self.input_transform(input)
        if self.target_transform:
            target = self.target_transform(target)

        if self.noise_level > 0:
            # Add noise
            input += self.noise_level * torch.randn_like(input)

        return input, target

    def __len__(self):
        """Length is amount of files found."""
        return len(self.image_filenames) * self.replicate


def _is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def _load_img(filepath, RGB=True):
    img = Image.open(filepath)
    if RGB:
        pass
    else:
        img = img.convert('YCbCr')
        img, _, _ = img.split()
    return img