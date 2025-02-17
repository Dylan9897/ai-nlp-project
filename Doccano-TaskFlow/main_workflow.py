import os

from flask import Flask
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime
import time
import json
import configparser
from module.workflow import create_project,download_task,send_email_to_eachother

app = Flask(__name__)

def scheduled_job_1():
    """
    自动创建标注任务
    :return:
    """
    status = create_project(file_root="data")
    if status:
        print("success create")
    else:
        print("Nothing to Do ...")
        
        
def scheduled_job_2():
    """
    下载标注结果，并总结xlsx报告
    :return:
    """
    status = download_task()
    if status:
        print("success merge")
    else:
        print("Nothing to Do ...")
        
        
def scheduled_job_3():
    """
    下载标注结果，并总结xlsx报告
    :return:
    """
    status = send_email_to_eachother()
    if status:
        print("success to send email")
    else:
        print("Nothing to Do ...")

# 创建一个调度器实例
scheduler = BackgroundScheduler()

# 添加一个作业，每天19点自动创建标注任务
scheduler.add_job(scheduled_job_1, 'cron', hour=19)

# 添加一个作业，每天下17点自动生成报告（前一天）
scheduler.add_job(scheduled_job_2, 'cron', hour=17)

# 添加一个作业，每天下17点30分定时发送邮件（前一天）
scheduler.add_job(scheduled_job_3, 'cron', hour=17,minute=30)

# 开始调度器
scheduler.start()

@app.route('/')
def hello_world():
    return 'Hello, World!'

if __name__ == '__main__':
    app.run()




