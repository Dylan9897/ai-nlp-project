import json
import re
import os
import pandas as pd

columns = ["id","sessionId","dialogue","summary_pred","summary_true","isSame","callResult_pred","callResult_true","isSame","finalResult_pred","finalResult_true","isSame","relation_pred","relation_true","isSame"]
zhcolumns = ["序号","业务编号","通话记录","预测摘要","人工摘要","是否相同","预测通话结果","人工通话结果","是否相同","预测最终结果","人工最终结果","是否相同","预测关系","人工关系","是否相同"]
def return_dict(param):
    """
    将labels中的数据转换成字典
    :param param:
    :return:
    """
    return {unit.split(":")[0]:unit.split(":")[1]  for unit in param}

def cleaner(seq):
    return re.sub("\s+","",seq)
def return_report_xlsx(src,tgt):
    result = []
    with open(src,"r",encoding="utf-8") as fl1:
        with open(tgt,"r",encoding="utf-8") as fl2:
            fl1 = fl1.readlines()
            fl2 = fl2.readlines()
            for i in range(len(fl1)):
                cur = []
                y_pred = json.loads(fl1[i])
                y_true = json.loads(fl2[i])

                y_label = return_dict(y_true["label"])

                cur.append(str(i+1))
                cur.append(y_pred["sessionId"])
                cur.append(y_pred["content"])

                cur.append(y_pred["summary"])
                cur.append(y_label["摘要内容"])
                cur.append(int(y_pred["summary"] == y_label["摘要内容"]))
                try:
                    cur.append(y_pred["result"])
                    cur.append(y_label["通话结果"])

                    cur.append(int(y_pred["result"] == cleaner(y_label["通话结果"])))
                except:

                    cur.append(y_label["通话结果"])
                    cur.append(y_label["通话结果"])

                    cur.append(int(cleaner(y_label["通话结果"]) == cleaner(y_label["通话结果"])))


                cur.append(y_pred["final"])
                cur.append(y_label["最终结果"])
                cur.append(int(y_pred["final"] == cleaner(y_label["最终结果"])))

                cur.append(y_pred["relation"])
                cur.append(y_label["对方身份"])
                cur.append(int(y_pred["relation"] == cleaner(y_label["对方身份"])))

                result.append(cur)
    return result

def write_to_excel(src,tgt):
    result = return_report_xlsx(src,tgt)
    dt = pd.DataFrame(result, columns=columns)

    statistic = []
    # 加入统计信息
    for i,col in enumerate(dt.columns):
        if col == "isSame":
            # 统计相等的个数
            count_of_same = (dt.iloc[:,i]==1).sum()
            statistic.append(str(count_of_same)+"/"+str(len(dt.iloc[:,i]))+"="+str(round(count_of_same/len(dt.iloc[:,i]),2)))

        elif col == "id":
            statistic.append("总计")
        else:
            statistic.append("##")

    new_row = pd.Series(statistic, index=dt.columns)
    dt = dt._append(new_row.to_frame().T)
    dt.columns = zhcolumns

    target_name = src.split(".")[0]
    dt.to_excel("{}.xlsx".format(target_name), index=False)
    return "{}.xlsx".format(target_name)


if __name__ == '__main__':
    src = "data/2024-08-15.json"
    tgt = "data/2024-08-15-tag.jsonl"
    write_to_excel(src, tgt)










