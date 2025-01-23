"""
-*- coding: utf-8 -*-

@Author : dongdong
@Time : 2022/12/13 15:17
@Email : handong_xu@163.com
"""
import os
from loguru import logger

# log
os.makedirs('log',exist_ok=True)
logger.add('log/{time}.log', rotation='1 GB')
