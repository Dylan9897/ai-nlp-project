import argparse
import json
import os
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from loguru import logger

from module.dataset import ImageDataset
from module.detector import KNNDetector
from module.models import MODELS_DICT

# log
os.makedirs('log',exist_ok=True)
logger.add('log/{time}.log', rotation='500 MB')

## * core function
def detect_abnormal(
    test_dir,
    db_dir=None,
    image_dir='image/train_50',
    model_name='vit',
    k=10,
    thres_ratio=1.0,
    gpu=True,
    batch_size=16,
    num_workers=8):


    _model_class, model_path = MODELS_DICT[model_name]
    model = _model_class(model_path=model_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    detector = KNNDetector(device=device)

    if os.path.exists(db_dir) and os.path.isdir(db_dir):

        detector.setup_from_db(db_dir)

        ## cover user defined parameters
        thres_ratio = detector.thres_ratio
        k = detector.K

    else:

        ## dataset
        base_dataset = ImageDataset(root=image_dir, transform=model.transform)
        # base_dataloader = DataLoader(base_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        base_dataloader = DataLoader(base_dataset, batch_size=batch_size, shuffle=False)
        detector.setup_from_dataset(model, base_dataloader, K=k, thres_ratio=thres_ratio, save_db_dir=db_dir)

    # * test
    test_dir = os.path.realpath(test_dir)
    test_dataset = ImageDataset(root=test_dir, transform=model.transform)
    # test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    abnormal_idx = detector.detect(model, test_dataloader)
    abnormal_paths = [test_dataset.img_paths[_id] for _id in abnormal_idx]

    # * save
    os.makedirs('result', exist_ok=True)
    save_dir = os.path.join('result', datetime.now().strftime('%Y-%m-%d_%H-%M-%S_%f'))
    os.makedirs(save_dir, exist_ok=True)

    result_path = os.path.join(save_dir, 'abnormal.result')
    with open(result_path, 'w') as f:
        for p in abnormal_paths: f.write(f'{p}\n')

    with open(os.path.join(save_dir, 'config.json'), 'w') as f:
        json.dump({
            'db_path': db_dir, 'test_dir': test_dir, 'model_name': model_name, 'k': k,
            'thres_ratio': thres_ratio, 'gpu': gpu, 'batch_size': batch_size, 'num_workers': num_workers
        }, f, indent=1)

    logger.info(f'{len(abnormal_idx)} abnormal images detected, which are saved in {result_path}.')
    return abnormal_paths

## * use example
def main():
    torch.set_grad_enabled(False)

    parser = argparse.ArgumentParser()

    parser.add_argument('--db_dir', type=str, default='database/MVTEC')
    parser.add_argument('--image_dir', type=str, default='data/MVTEC/train')
    parser.add_argument('--model_name', type=str, default='vit')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--k', type=int, default=100)
    parser.add_argument('--thres_ratio', type=float, default=0.92)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--test_dir', type=str, default='data/MVTEC/test/broken')
    args = parser.parse_args()
    detect_abnormal(test_dir=args.test_dir, db_dir=args.db_dir, image_dir=args.image_dir,
                    model_name=args.model_name, k=args.k, thres_ratio=args.thres_ratio, gpu=args.gpu,
                    batch_size=args.batch_size, num_workers=args.num_workers)


if __name__ == '__main__':
    main()


