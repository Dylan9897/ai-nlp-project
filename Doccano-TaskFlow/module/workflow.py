import os
from datetime import datetime, timedelta
import json
import pandas as pd
import configparser
import shutil
from module.MyDoccano import MyDoccanoClient
from module.analysis import write_to_excel
from module.send_email import send_email_with_attachment

import subprocess

def return_doccano_config():
    """
    获取配置文件信息
    :return:
    """
    config = configparser.ConfigParser()
    config.read("config/config.ini", encoding="utf-8")
    url = config.get("DOCCANO", "DOCCANO_URL")
    user = config.get("DOCCANO", "USERNAME")
    password = config.get("DOCCANO", "PASSWORD")
    return url, user, password

def return_email_config():
    """
    获取配置文件信息
    :return:
    """
    config = configparser.ConfigParser()
    config.read("config/config.ini", encoding="utf-8")
    SUBJECT = config.get("EMAIL", "SUBJECT")
    BODY = config.get("EMAIL", "BODY")
    SMTP_ADDRESS = config.get("EMAIL", "SMTP_ADDRESS")
    SMTP_PORT = config.get("EMAIL", "SMTP_PORT")
    SENDER_EMAIL = config.get("EMAIL", "SENDER_EMAIL")
    PASSWORD = config.get("EMAIL", "PASSWORD")
    return SUBJECT,BODY,SMTP_ADDRESS,SMTP_PORT,SENDER_EMAIL,PASSWORD


def return_cache():
    """
    获取缓存文件信息
    :return:
    {
        “项目名称”:{
            "id"：项目id，int,
            "time"：创建时间，str,
            "category":类别，str,
            "path"：数据集文件地址，str,
            "count"：数据集数量，int,
            "annotator"：负责人，str,
            "status"：状态，bool
        }
    }
    """
    with open("config/cache.json", "r", encoding="utf-8") as fl:
        cache = json.load(fl)
    return cache

##### 获取当天的文件 #####
def get_last_file(file_root):
    """
    获取文件夹最后一个文件
    :param file_root:
    :return:
    """
    file_list = os.listdir(file_root)

    def extract_date_from_filename(filename):
        return datetime.strptime(filename.split('.')[0], '%Y-%m-%d')

    # 使用定义的函数将文件名按日期排序
    sorted_files = sorted(file_list, key=extract_date_from_filename)
    return sorted_files[-1]

def get_target_file(file_root):
    today = datetime.now().strftime("%Y-%m-%d")
    # 获取当天的文件
    if os.path.exists(os.path.join(file_root, today + ".json")):
        return os.path.join(file_root, today + ".json"),today
    # 获取最后一天的文件
    target_file = get_last_file(file_root)
    return os.path.join(file_root,target_file),target_file.split(".")[0]

##### 创建标注任务 #####
def create_data_file(src,tgt):
    with open(src,"r",encoding="utf-8") as fl:
        with open(tgt,"w",encoding="utf-8") as ft:
            for line in fl:
                line = json.loads(line)
                data = {
                    "sessionId":line["sessionId"],
                    "text":line["content"].replace("0:","\n0:").replace("1:","\n1:"),
                    "label":[
                        "对方身份:"+line["relation"],
                        "通话结果:"+line.get("result","Unknown"),
                        "最终结果:"+line["final"],
                        "摘要内容:"+line["summary"]]
                }
                json_data = json.dumps(data,ensure_ascii=False)
                ft.write(json_data+"\n")

def create_project(
    file_root,
    project_type="Seq2seq",
    suffix="-催记任务",
    format="JSONL",
    annotator="annotator1",
    role_name="annotator"
):
    # 获取目标文件
    target_file,target_file_name = get_target_file(file_root)

    cache_config = return_cache()
    # 文件存在并已经被处理过，则不进行任何操作
    if target_file_name in cache_config and cache_config[target_file_name]["status"]:
        return None

    # 创建缓存文件
    tgt_file_path = "cache/cache.jsonl"
    create_data_file(target_file,tgt_file_path)

    ###### 创建项目
    doccano_url, doccano_username, doccano_password = return_doccano_config()
    # 连接标注平台
    cur_client = MyDoccanoClient(doccano_url, doccano_username, doccano_password)

    # 创建项目
    # project_type: ProjectType,
    #         description: str,
    cur_project = cur_client.create_project(
        name=target_file_name+suffix,
        project_type=project_type,
        description=f"日期为：{target_file_name}"
    )
    cur_project_id = cur_project.id

    # 上传数据集
    cur_client.upload(project_id=cur_project_id, file_paths=["cache/cache.jsonl"], task=project_type, format=format)

    # 分配任务
    cur_client.add_member(project_id=cur_project_id, username=annotator, role_name=role_name)


    # 更新到缓存配置文件
    cache_data = {
        target_file_name:{
            "id":cur_project_id,
            "time": datetime.now().strftime("%Y-%m-%d"),
            "category":project_type,
            "path":target_file,
            "count":0,
            "annotator":annotator,
            "status":False
        }
    }
    cache_config.update(
        cache_data
    )
    with open("config/cache.json", "w", encoding="utf-8") as ft:
        json_data = json.dumps(cache_config,ensure_ascii=False,indent=4)
        ft.write(json_data)
    return True


##### 下载标注任务 #####


# 移动文件
def move_file(src, dst):
    try:
        shutil.move(src, dst)
        print(f"Moved '{src}' to '{dst}'.")
    except FileNotFoundError:
        print(f"Source file '{src}' does not exist.")
    except PermissionError:
        print(f"Permission denied for '{src}' or '{dst}'.")
    except Exception as e:
        print(f"An error occurred: {e}")

# 删除文件
def delete_file(file_path):
    try:
        os.remove(file_path)
        print(f"Deleted '{file_path}'.")
    except FileNotFoundError:
        print(f"File '{file_path}' does not exist.")
    except PermissionError:
        print(f"Permission denied for '{file_path}'.")
    except Exception as e:
        print(f"An error occurred: {e}")

# 解压文件夹
def unzip_file(zip_file_path, destination_folder):
    # 确保目标文件夹存在
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # 构建unzip命令
    command = ['unzip', zip_file_path, '-d', destination_folder]

    # 执行unzip命令
    try:
        result = subprocess.run(command, check=True)
        print(f"Unzipped '{zip_file_path}' to '{destination_folder}'.")
        return result
    except subprocess.CalledProcessError as e:
        print(f"Failed to unzip '{zip_file_path}': {e}")
        return None

def download_task():
    # 获取当前日期
    current_date = datetime.now()
    
    current_date = current_date - timedelta(days=1).date()

    # 计算前一天的日期
    current_date = current_date.strftime("%Y-%m-%d")

    # 获取缓存文件
    cache_config = return_cache()

    # 获取配置信息


    target = cache_config[current_date]
    if target["status"]:
        return None
        
    target["status"] = True
    cache_config[current_date] = target
    with open("config/cache.json", "w", encoding="utf-8") as ft:
        json_data = json.dumps(cache_config,ensure_ascii=False,indent=4)
        ft.write(json_data)

    project_no = target['id']

    # 连接标注平台
    doccano_url, doccano_username, doccano_password = return_doccano_config()
    cur_client = MyDoccanoClient(doccano_url, doccano_username, doccano_password)
    cur_client.list_all_projects()


    # 下载标注任务
    data_path = cur_client.download(project_id=project_no,format="JSONL",dir_name="./cache")

    # 解压到指定文件夹
    unzip_file(data_path,"./cache")
    
    # 将all.jsonl 移动到 data 文件夹
    move_file(src="cache/all.jsonl", dst="data/{}-tag.jsonl".format(current_date))
    
    # 删除下载的 zip 文件夹
    delete_file(data_path)
    
    # 写入excel 报告
    src_file = os.path.join("data", "{}.json".format(current_date))
    tgt_file = os.path.join("data", "{}-tag.jsonl".format(current_date))
    xlsx_path = write_to_excel(src_file,tgt_file)
    print(xlsx_path)
    
    return xlsx_path

################# 推送邮件 #################
def send_email_to_eachother():
    
    SUBJECT,BODY,SMTP_ADDRESS,SMTP_PORT,SENDER_EMAIL,PASSWORD = return_email_config()
    
    
    # 获取当前日期
    current_date = datetime.now()
    
    current_date = current_date - timedelta(days=1).date()

    # 计算前一天的日期
    current_date = current_date.strftime("%Y-%m-%d")
    
    # 获取文件列表
    filename_list = [os.path.join("data", "{}-tag.jsonl".format(current_date)),os.path.join("data", "{}.xlsx".format(current_date))]
    
    # 获取收件人邮箱列表并进行发送
    df = pd.read_excel("config/邮箱账号.xlsx")
    for i in df.index:
        line = df.loc[i]
        if line["use"] == 1:
            send_email_with_attachment(subject=SUBJECT, message=BODY,smtp_address=SMTP_ADDRESS,smtp_port=SMTP_PORT, from_addr=SENDER_EMAIL, to_addr=line["email"], password=PASSWORD ,filename_list=filename_list)




if __name__ == '__main__':

    download_task()







