# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 12:42:39 2023

@author: 唐总
"""

from openpyxl import load_workbook
import matplotlib.pyplot as plt
import warnings
import numpy as np
warnings.filterwarnings("ignore")

wb = load_workbook('./data_analysis.xlsx',data_only=True)
sheet = wb.get_sheet_by_name('Sheet1')
# In[x-delta_y] 拟合一次式
real_x = []
contour_x = []
for rows in sheet['D81':'D93']:
    for column in rows:
        real_x.append(column.value)
        print(column.value)
        
for rows in sheet['B81':'B93']:
    for column in rows:
        contour_x.append(column.value)
        print(column.value)
plt.plot(contour_x,real_x)
plt.xlabel('contour_x')
plt.ylabel('real_x')
plt.show()
# In[拟合]
z = np.polyfit(contour_x,real_x,2)
p = np.poly1d(z)
print(p)
y_pre = p(contour_x)
plt.plot(contour_x,real_x,'*')
plt.plot(contour_x,y_pre)



