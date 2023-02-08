#!/usr/bin/env python
# coding: utf-8

# In[41]:


#import libraries
from bs4 import BeautifulSoup as bs
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.linear_model import LinearRegression

#scrape table from web using Beautiful Soup
url = "https://www.fantasypros.com/nfl/advanced-stats-wr.php"
result = requests.get(url)
soup = bs(result.content)

table = soup.select('table#data')[0]
columns = table.find('thead').find_all('th')
WR2022_df = pd.read_html(str(table))[0]

WR2022_df.columns = WR2022_df.columns.droplevel(0)

#Declare Variables
x = WR2022_df[['AIR/R','YAC/R','YACON/R']]
y = WR2022_df['Rank']

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


# In[14]:





# In[ ]:




