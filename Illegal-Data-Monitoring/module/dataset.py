import imghdr
import os
import sys

from loguru import logger
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

def _convert_image_to_rgb(image):
    return image.convert("RGB")

class ImageDataset(Dataset):
    def __init__(self, root='image', transform=None):
        self.root = root
        self.transform = transform
        # dir does not exist
        try:
            assert os.path.exists(root)
        except:
            logger.error(f'{root} does not exist!')
            sys.exit()

        self.img_paths = []
        for filename in os.listdir(root):
            filepath = os.path.join(root, filename)

            # filepath does not exist
            if not os.path.exists(filepath):
                logger.info(f'ignoring {filepath}, since it does not exist.')
                continue
            
            # is not file
            if not os.path.isfile(filepath):
                logger.info(f'ignoring {filepath}, since it does not exist.')
                continue

            # not image file
            if imghdr.what(filepath) == None: 
                logger.info(f'ignoring {filepath}, since it is not a valid image file.')
                continue
            
            self.img_paths.append(filepath)

        logger.info(f'database size: {len(self.img_paths)}.')
        if len(self.img_paths) == 0:
            logger.error('Database size is 0!')
            sys.exit()
        
        if transform: 
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.Resize((256, 256), interpolation=InterpolationMode.BICUBIC),
                transforms.CenterCrop((224, 224)),
                _convert_image_to_rgb,
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])

    def __getitem__(self, idx):
        path = self.img_paths[idx]
        img = Image.open(path)
        img = self.transform(img)
        return img
    
    def __len__(self):
        return len(self.img_paths)

