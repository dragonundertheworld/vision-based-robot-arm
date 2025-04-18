# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 14:59:34 2023

@author: 唐总
"""

# In[串口通信]
import serial
import time
# 连接串口
serial = serial.Serial('COM3', 9600, timeout=0.1)
if serial.isOpen():
    print('串口已打开')
    data = b'Hello STM32!!!\r\n'  # 发送的数据
    serial.write(data)  # 串口写数据
    print("-" * 40)
    print('Python Send :\n', data)
    print("-" * 40)
    time.sleep(1)
    while True:
        data = serial.read_all()
        if data != b'':
            break
    print("-" * 40)
    print('STM32 Send :\n', data.decode("GBK"))
    print("-" * 40)
else:
    print('串口未打开')
 
# 关闭串口
serial.close()
 
if serial.isOpen():
    print('串口未关闭')
else:
    print('串口已关闭')