from PIL import Image

from torchvision import transforms
import torch
from module.detector import KNNDetector
from module.models import MODELS_DICT
from loguru import logger

class api():
    def __init__(self,model_name = 'vit'):
        self.model_name = model_name
        self._model_class, self.model_path = MODELS_DICT[self.model_name]
        self.model = self._model_class(model_path=self.model_path)
        self.db_dir = 'database/MVTEC'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.detector = KNNDetector(device=self.device)
        self.model = self.model.to(self.device)

    def trans(self,path):
        img = Image.open(path)

        
        if self.model.transform:
            transform = self.model.transform
        else:
            transform = transforms.Compose([
                transforms.Resize((256, 256), interpolation=InterpolationMode.BICUBIC),
                transforms.CenterCrop((224, 224)),
                _convert_image_to_rgb,
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
        
        res = transform(img)
        return res.to(self.device)


    def predict_raw(self,path):
        logger.info(f'input data is {path}')
        example = self.trans(path)
        self.detector.setup_from_db(self.db_dir)

        ## cover user defined parameters
        label,score = self.detector.predict(self.model, example)
        logger.info(f'predict label is {label}')
        logger.info(f'predict score is {score}')
        return label


if __name__ == '__main__':
    fun = api()
    path = "image/train/80f5667578236e1157708219e64e989bbfa4cd23.jpg"
    fun.predict_raw(path)











