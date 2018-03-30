# -*- coding:utf-8 -*-

import csv

filepath = "E:/20180327-adb/2233/chatroom.csv"
with open(filepath) as f:
    for i in f.readlines():
        print(i.split()[0])