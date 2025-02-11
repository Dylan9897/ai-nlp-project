# Text-Classify
​	本代码专注于实现文本分类任务，其数据源于一个专门构建的长文本分类数据集。在数据预处理阶段，我们采用了0.15的比例对原始数据进行划分，以构建评估模型性能的测试集与用于学习的训练集。具体而言，训练集包含5950条精心挑选的样本，而测试集则包括了1050条样本。该分类体系涵盖了七个核心领域，分别是：“时尚”、“财经”、“时政”、“家居”、“房产”、“教育”以及“科技”。

## 1、LightGBM

运行脚本：

```bash
cd LightGBM
python train.py
```

运行结果如下：

```python
				precision    recall  f1-score   support

          时尚     0.9595    0.9930    0.9759       143
          财经     0.9400    0.9724    0.9559       145
          时政     0.9930    0.9726    0.9827       146
          家居     1.0000    1.0000    1.0000       166
          房产     0.9816    0.9639    0.9726       166
          教育     1.0000    1.0000    1.0000       134
          科技     0.9589    0.9333    0.9459       150

    accuracy                         0.9762      1050
   macro avg     0.9761    0.9765    0.9762      1050
weighted avg     0.9764    0.9762    0.9762      1050
```

## 2、XGBoost

运行脚本：

```bash
cd XGBoost
python train.py
```

运行结果如下：

```python
				precision    recall  f1-score   support

           时尚     0.9530    0.9930    0.9726       143
           财经     0.9396    0.9655    0.9524       145
           时政     0.9861    0.9726    0.9793       146
           家居     1.0000    1.0000    1.0000       166
           房产     0.9814    0.9518    0.9664       166
           教育     1.0000    1.0000    1.0000       134
           科技     0.9524    0.9333    0.9428       150

    accuracy                         0.9733      1050
   macro avg     0.9732    0.9738    0.9733      1050
weighted avg     0.9736    0.9733    0.9733      1050

```

## 3、TextCNN

## 4、Bert

运行脚本

```bash
export HF_ENDPOINT=https://hf-mirror.com
python train.py \
    --train_file data/datasets/longnews/train.json \
    --valid_file data/datasets/longnews/dev.json \
    --num_labels 7 \
    --output output_bert_base_chinese_thucnews
```

模型训练结果如下：

LongNews

```python
                precision    recall  f1-score   support

          教育       0.97      0.98      0.97       154
          财经       0.97      0.94      0.95       130
          科技       0.97      0.99      0.98       135
          房产       0.95      0.95      0.95       156
          时政       0.95      0.95      0.95       130
          家居       0.96      0.96      0.96       158
          时尚       0.99      1.00      1.00       138

    accuracy                           0.97      1001
   macro avg       0.97      0.97      0.97      1001
weighted avg       0.97      0.97      0.97      1001
```

ThucNews

```python
               precision    recall  f1-score   support

      finance       0.92      0.93      0.92      1000
       realty       0.96      0.95      0.95      1000
       stocks       0.91      0.89      0.90      1000
    education       0.96      0.97      0.97      1000
      science       0.91      0.90      0.91      1000
      society       0.90      0.95      0.93      1000
     politics       0.92      0.92      0.92      1000
       sports       0.98      0.98      0.98      1000
         game       0.97      0.94      0.95      1000
entertainment       0.95      0.97      0.96      1000

     accuracy                           0.94     10000
    macro avg       0.94      0.94      0.94     10000
 weighted avg       0.94      0.94      0.94     10000
```

