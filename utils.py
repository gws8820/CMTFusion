import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import torch
import os
import numpy as np


class Customdataset(Dataset):
    def __init__(self, transform=None, rgb_dataset=None, ir_dataset=None):
        # Collect file names
        self.image_rgb_paths = sorted([f for f in os.listdir(rgb_dataset) if f.endswith('.png')])
        self.image_ir_paths = sorted([f for f in os.listdir(ir_dataset) if f.endswith('.png')])

        # Ensure matching counts
        assert len(self.image_rgb_paths) == len(self.image_ir_paths), \
            f"Number of RGB images ({len(self.image_rgb_paths)}) and IR images ({len(self.image_ir_paths)}) do not match."

        self.transform = transform
        self.rgb_dataset = rgb_dataset
        self.ir_dataset = ir_dataset

    def __getitem__(self, index):
        # Construct file paths
        rgb_image_path = os.path.join(self.rgb_dataset, self.image_rgb_paths[index])
        ir_image_path = os.path.join(self.ir_dataset, self.image_ir_paths[index])

        # Load images
        img1 = Image.open(rgb_image_path).convert('RGB')
        img2 = Image.open(ir_image_path).convert('RGB')

        if self.transform:
            i, j, h, w = transforms.RandomCrop.get_params(img1, output_size=(256, 256))
            img1 = self.transform(TF.crop(img1, i, j, h, w))
            img2 = self.transform(TF.crop(img2, i, j, h, w))

        return img1, img2

    def __len__(self):
        return len(self.image_rgb_paths)


def get_test_images(paths, height=None, width=None):
    ImageToTensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Grayscale(num_output_channels=1),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    if isinstance(paths, str):
        paths = [paths]
    images = []
    for path in paths:
        image = get_image(path)
        image_np = np.array(image, dtype=np.uint32)
        image = ImageToTensor(image).float().numpy()
    images.append(image)
    images = np.stack(images, axis=0)
    images = torch.from_numpy(images).float()

    return images


def get_image(path):
    image = Image.open(path).convert('RGB')

    return image
