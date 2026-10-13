"""CIFAR-100 dataset loading, deterministic train/val splitting and caching."""

import os
from typing import Tuple, Optional
import torch
from torch.utils.data import DataLoader, Subset, random_split
import torchvision
import torchvision.transforms as transforms

from compressiq.configs.schema import DataConfig


# Standard CIFAR-100 normalization parameters
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)


def get_cifar100_transforms(image_size: int = 32, is_training: bool = True) -> transforms.Compose:
    """Return data transforms for CIFAR-100.
    
    If image_size > 32, resize is prepended to adapt to standard architectures.
    """
    transform_list = []
    if image_size != 32:
        transform_list.append(transforms.Resize((image_size, image_size)))

    if is_training:
        transform_list.extend([
            transforms.RandomCrop(image_size, padding=max(2, image_size // 8)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)
        ])
    else:
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)
        ])
    return transforms.Compose(transform_list)


def get_cifar100_dataloaders(
    config: DataConfig,
    data_dir: Optional[str] = None
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Load official CIFAR-100 train & test datasets, partition train into train & val,
    and return (train_loader, val_loader, test_loader).
    
    Args:
        config: DataConfig instance containing split sizes, batch sizes, seeds.
        data_dir: Optional override path for dataset caching.
        
    Returns:
        (train_loader, val_loader, test_loader)
    """
    target_dir = data_dir or config.data_dir
    os.makedirs(target_dir, exist_ok=True)
    need_download = not (os.path.exists(os.path.join(target_dir, "cifar-100-python")) and 
                         os.path.exists(os.path.join(target_dir, "cifar-100-python", "train")))

    # Base transforms
    train_transform = get_cifar100_transforms(image_size=config.image_size, is_training=True)
    eval_transform = get_cifar100_transforms(image_size=config.image_size, is_training=False)

    # Full training dataset from torchvision
    full_train_dataset = torchvision.datasets.CIFAR100(
        root=target_dir,
        train=True,
        download=need_download,
        transform=train_transform
    )

    # Separate val dataset with eval transforms
    val_dataset_eval = torchvision.datasets.CIFAR100(
        root=target_dir,
        train=True,
        download=need_download,
        transform=eval_transform
    )

    # Official test dataset
    test_dataset = torchvision.datasets.CIFAR100(
        root=target_dir,
        train=False,
        download=need_download,
        transform=eval_transform
    )

    total_train = len(full_train_dataset)
    val_size = int(total_train * config.val_split)
    train_size = total_train - val_size

    generator = torch.Generator().manual_seed(config.seed)
    indices = torch.randperm(total_train, generator=generator).tolist()

    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_subset = Subset(full_train_dataset, train_indices)
    val_subset = Subset(val_dataset_eval, val_indices)

    # Note: On Windows, num_workers > 0 can sometimes cause multiprocessing overhead in small scripts,
    # so we clamp safely based on OS and settings.
    workers = min(config.num_workers, 2) if os.name != 'nt' else 0

    train_loader = DataLoader(
        train_subset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available()
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available()
    )

    return train_loader, val_loader, test_loader
