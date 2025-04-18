# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 20:27:44 2023

@author: 唐总
"""

# In[原点调试]
import serial
serial = serial.Serial('COM3', 115200, timeout=0.1)
if serial.isOpen():
    print('串口已打开')
    serial.write('217x'.encode())  # 串口写数据
    serial.write('217x'.encode())
    serial.write('200y'.encode())
    serial.write('200y'.encode())
    serial.write('10z'.encode())
    serial.write('10z'.encode())
else:
    print('串口未打开')
# In[特殊点]
import serial
serial = serial.Serial('COM3', 115200, timeout=0.1)
if serial.isOpen():
    print('串口已打开')
    serial.write('230x'.encode())  # 串口写数据
    serial.write('230x'.encode())
    serial.write('230y'.encode())
    serial.write('230y'.encode())
    serial.write('20z'.encode())
    serial.write('20z'.encode())
else:
    print('串口未打开')
# In[复位]
import serial
serial = serial.Serial('COM3', 115200, timeout=0.1)
if serial.isOpen():
    print('串口已打开')
    serial.write('-1z'.encode())
    
else:
    print('串口未打开')
# In[]
# 关闭串口
import serial
serial = serial.Serial('COM3', 115200, timeout=0.1)
serial.close()

if serial.isOpen():
    print('串口未关闭')
else:
    print('串口已关闭')