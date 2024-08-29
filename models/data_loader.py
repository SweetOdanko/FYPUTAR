import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import random

class CustomPairDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_labels = []
        self._prepare_dataset()

    def _prepare_dataset(self):
        classes = os.listdir(self.root_dir)
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        for cls in classes:
            cls_folder = os.path.join(self.root_dir, cls)
            if not os.path.isdir(cls_folder):
                continue
            for img_name in os.listdir(cls_folder):
                img_path = os.path.join(cls_folder, img_name)
                self.image_labels.append((img_path, class_to_idx[cls]))
        self.classes = classes

    def __len__(self):
        return len(self.image_labels)

    def __getitem__(self, idx):
        img_path1, label1 = self.image_labels[idx]
        image1 = Image.open(img_path1).convert('RGB')

        if random.random() > 0.5:
            same_class = True
            while True:
                idx2 = random.randint(0, len(self.image_labels) - 1)
                img_path2, label2 = self.image_labels[idx2]
                if label1 == label2:
                    break
        else:
            same_class = False
            while True:
                idx2 = random.randint(0, len(self.image_labels) - 1)
                img_path2, label2 = self.image_labels[idx2]
                if label1 != label2:
                    break

        image2 = Image.open(img_path2).convert('RGB')

        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)

        return (image1, image2), torch.tensor([same_class], dtype=torch.float32)

def get_train_validation_loader(data_dir, batch_size, num_workers, pin_memory):
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
        transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),  # Ensure the size is 224
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    transform_val_test = transforms.Compose([
        transforms.Resize(224),  # Ensure the size is 224
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    train_dataset = CustomPairDataset(root_dir=os.path.join(data_dir, 'train'), transform=transform_train)
    val_dataset = CustomPairDataset(root_dir=os.path.join(data_dir, 'valid'), transform=transform_val_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    return train_loader, val_loader

def get_test_loader(data_dir, batch_size, num_workers, pin_memory):
    transform = transforms.Compose([
        transforms.Resize(224),  # Ensure the size is 224
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    test_dataset = CustomPairDataset(root_dir=os.path.join(data_dir, 'test'), transform=transform)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    return test_loader
