# -*- coding: utf-8 -*-
"""
Created on Sat May 22 11:24:08 2021

@author: Chris Chiang
"""

import numpy as np                # 處理數值資料之套件
import matplotlib.pyplot as plt   # 畫圖用套件
import utils                      # 自定義函數
import sklearn.datasets    # 機器學習常用套件
import pandas as pd        # 常拿來處理表格之套件

boston_data = sklearn.datasets.load_boston()
df = pd.DataFrame(data=boston_data["data"],
                  columns=boston_data['feature_names'])
df["target"] = boston_data["target"]

# 讓印出格式顯示至小數點後兩位
pd.options.display.float_format = '{:,.2f}'.format

df.corr()

import seaborn as sns
plt.figure(figsize=(12, 9))
sns.heatmap(df.corr(),  # Pearson correlation
            annot=True,
            vmin=-1, vmax=1)

plt.figure(figsize=(12, 9))
sns.heatmap(df.corr(method='spearman'),
            annot=True,  # 在heatmap中印出值
            vmin=-1, vmax=1)

plt.figure(figsize=(12, 9))
sns.heatmap(df.corr(method='kendall'), annot=True,
            fmt=".2f", vmin=-1, vmax=1, center=0)