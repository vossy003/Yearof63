#!/usr/bin/env python
# coding: utf-8

# In[42]:


#Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.linear_model import LinearRegression

#Create WR data set
df = pd.read_csv('https://raw.githubusercontent.com/fantasydatapros/data/master/yearly/2021.csv')
wr = df.drop(['Unnamed: 0','Tm','PassingYds','PassingTD','PassingAtt','RushingYds','RushingTD','RushingAtt','Int'],axis=1)
wr.drop(wr[wr['Pos']!='WR'].index, inplace = True)

#Add per game fields and drop NaN
wr['TgtsperGame']= wr['Tgt']/wr['G']
wr['FPperGame'] =wr['FantasyPoints']/wr['G']
wr = wr.dropna()

#Declare Variables
x = wr[['Age','TgtsperGame','Fumbles']]
y = wr['FPperGame']

#Declare Regression, find intercept, coefficients, Rsquared
reg = LinearRegression()
reg.fit(x,y)
print('Intercept =', reg.intercept_)
print('Coefficients =', reg.coef_)
print('R-Squared = ',reg.score(x,y))

#Create R-Squared Function
def adj_r2(x,y):
    r2 = reg.score(x,y)
    n = x.shape[0]
    p = x.shape[1]
    adjusted_r2 = 1-(1-r2)*(n-1)/(n-p-1)
    return adjusted_r2

#Run Adjusted R2
print('Adjusted R2 =', adj_r2(x,y))


#Calculate P Values
from sklearn.feature_selection import f_regression
f_regression(x,y)
p_values = f_regression(x,y)[1]
print('PValues =', p_values)

#Summary
reg_summary = pd.DataFrame(data = x.columns.values, columns=['Features'])
reg_summary ['Coefficients'] = reg.coef_
reg_summary ['p-values'] = p_values.round(3)
reg_summary

#Remove Age because it's not significant


# In[ ]:





# In[ ]:




