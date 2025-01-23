import xgboost as xgb
# from models.XGBoost import Model
from utils.converter import undump


# data = v.transform(words.values.astype('U'))

class api():
    def __init__(self):
        self.v = undump('ckpt/tfidf2.model')
        self.clf = undump('ckpt/xgboost.trans_base.model')

    def predict(self,seq):
        content = self.v.transform([seq])
        pred = self.clf.predict(xgb.DMatrix(content))
        return pred[0]

        



if __name__== "__main__":
    seq = "呃，是的。"
    fun = api()
    res = fun.predict(seq)
    print(res)