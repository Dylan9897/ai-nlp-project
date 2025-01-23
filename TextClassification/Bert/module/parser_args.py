import argparse

def return_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default='ckpt/nlp_roberta_backbone_large_std', type=str, help="backbone of encoder.")
    parser.add_argument("--train_path", default="data/datasets/thucnews/train.json", type=str, help="The path of train set.")
    parser.add_argument("--dev_path", default="data/datasets/thucnews/dev.json", type=str, help="The path of dev set.")
    parser.add_argument("--save_dir", default="output", type=str, required=False,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--max_seq_len", default=512, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                            "than this will be truncated, sequences shorter will be padded.", )
    parser.add_argument("--batch_size", default=30, type=int, help="Batch size per GPU/CPU for training.", )
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--num_train_epochs", default=20, type=int, help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_ratio", default=0.06, type=float, help="Linear warmup over warmup_ratio * total_steps.")
    parser.add_argument("--valid_steps", default=200, type=int, required=False, help="evaluate frequecny.")
    parser.add_argument("--logging_steps", default=100, type=int, help="log interval.")
    parser.add_argument('--device', default="cuda", help="Select which device to train model, defaults to gpu.")
    parser.add_argument("--img_log_dir", default='logs', type=str, help="Logging image path.")
    parser.add_argument("--img_log_name", default='Model Performance', type=str, help="Logging image file name.")
    parser.add_argument("--num_labels", default=10, type=int, help="Total classes of labels.")
    parser.add_argument("--use_class_weights", default=False, type=bool,
                        help="compute class weights with sample counts of each class.")
    parser.add_argument("--max_scale_ratio", default=10.0, type=float, help="max scale ratio when use class weights.")
    parser.add_argument("--loss_func", default='cross_entropy', type=str, help="choose loss function.",
                        choices=['cross_entropy', 'focal_loss'])
    parser.add_argument("--focal_loss_alpha", default=0.25, type=float, help="alpha of focal loss.")
    parser.add_argument("--focal_loss_gamma", default=2.0, type=float, help="gamma of focal loss.")
    args = parser.parse_args()
    return args
