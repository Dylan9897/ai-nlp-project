export HF_ENDPOINT=https://hf-mirror.com
python train.py
python train.py \
    --train_file data/datasets/thucnews/train.json \
    --valid_file data/datasets/thucnews/dev.json \
    --num_labels 10 \
    --output output_bert_base_chinese_thucnews