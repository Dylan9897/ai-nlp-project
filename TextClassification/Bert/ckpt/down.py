#模型下载
import shutil
from modelscope import snapshot_download
model_dir = snapshot_download('iic/nlp_roberta_backbone_large_std')
shutil.move(model_dir,"ckpt")
