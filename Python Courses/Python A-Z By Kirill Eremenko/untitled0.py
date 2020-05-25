# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 06:23:15 2020

@author: 10
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')
stats = pd.read_csv('P4-Demographic-Data.csv')
vis1 = sns.distplot(stats['Internet users'], bins= 30)

vis2 = sns.boxplot(data = stats, x ="Income Group", y = "Birth rate")

vis3 = sns.lmplot(data = stats, x = "Internet users", y="Birth rate", fit_reg = False ,
                  hue = 'Income Group', size=10, scatter_kws={"s":100})
