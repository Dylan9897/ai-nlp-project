import sys
from PIL import Image

from torchvision import transforms
import torch
from module.detector import KNNDetector
from module.models import MODELS_DICT
from loguru import logger
from io import BytesIO
import json
import time
import base64
from torchvision.transforms import InterpolationMode
from flask import Flask, request, jsonify, Response
import logging

app = Flask(__name__)
logging.getLogger().setLevel(logging.INFO)


@app.route('/test')
def hello():
    return "hello, world"

def _convert_image_to_rgb(image):
    return image.convert("RGB")

class api():
    def __init__(self,model_name = 'vit'):
        self.model_name = model_name
        self._model_class, self.model_path = MODELS_DICT[self.model_name]
        logging.info(self._model_class)
        logging.info(self.model_path)
        self.model = self._model_class(model_path=self.model_path)
        self.db_dir = 'database/5'
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.detector = KNNDetector(device=self.device)

    def trans(self,path):
        img = Image.open(BytesIO(base64.b64decode(path)))
        transform = transforms.Compose([
                transforms.Resize((256, 256), interpolation=InterpolationMode.BICUBIC),
                transforms.CenterCrop((224, 224)),
                _convert_image_to_rgb,
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
        res = transform(img)
        return res


    def predict_raw(self,path):
        example = self.trans(path)
        self.detector.setup_from_db(self.db_dir)
        ## cover user defined parameters
        label,score = self.detector.predict(self.model, example)
        return label,score



fun = api()

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    start = time.time()
    if request.method == 'POST':
        query = request.form['query']
    else:
        query = request.args.get('query', '')
    label,score = fun.predict_raw(query)
    end = time.time()
    result = {
        "label":str(label),
        "score":str(score),
        "cost":str(end-start)
    }

    return jsonify(result)


@app.errorhandler(404)
def notFound(error):
    return jsonify(error=str(error)), 404


@app.errorhandler(500)
def forbiden(error):
    return jsonify(error=str(error)), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5006)









