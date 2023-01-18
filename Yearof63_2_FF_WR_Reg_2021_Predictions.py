#!/usr/bin/env python
# coding: utf-8

# In[59]:


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
x = wr[['TgtsperGame','Fumbles']]
y = wr['FPperGame']


#Import the preprocessing module, scaling the data (finding the standard deviation to scale)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x)


#Declare new scaled variable, and transform the data into scaled variable
x_scaled = scaler.transform(x)

#Declare Regression, find intercept, coefficients, Rsquared
reg = LinearRegression()
reg.fit(x_scaled,y)

#Create R-Squared Function
def adj_r2(x,y):
    r2 = reg.score(x,y)
    n = x.shape[0]
    p = x.shape[1]
    adjusted_r2 = 1-(1-r2)*(n-1)/(n-p-1)
    return adjusted_r2

#Run Adjusted R2
print('Adjusted R2 =', adj_r2(x_scaled,y))

#train_test_split of model
from sklearn.model_selection import train_test_split
x_test, x_train, y_test, y_train = train_test_split(x_scaled,y)

#LinearRegression with Test Data
linear_model = LinearRegression().fit(x_train,y_train)

print('Linear Trainng Score', linear_model.score(x_train,y_train))
print('Linear Testing Score', linear_model.score(x_test,y_test))

#Gradient Boosting Regressor
from sklearn.ensemble import GradientBoostingRegressor
gradient_model = GradientBoostingRegressor(random_state=0).fit(x_train, y_train)
print('Gradient Boosting Training Score', gradient_model.score(x_train, y_train))
print('Gradient Boosting Testing Score', gradient_model.score(x_test, y_test))

#Random Forest Refressor
from sklearn.ensemble import RandomForestRegressor
forest_model = RandomForestRegressor(random_state=0).fit(x_train, y_train)
print('Random Forest Training Score',forest_model.score(x_train, y_train))
print('Random Forest Training Score', forest_model.score(x_test,y_test))

#Create New Data to Predict with
#scale Data
new_data = pd.DataFrame(data=[[8,2],[8,5],[7,2]], columns=['TgtsperGame','Fumbles'])
new_data_scaled = scaler.transform(new_data)
print('Linear model predictions', linear_model.predict(new_data_scaled))
print('Gradient model predictions',gradient_model.predict(new_data_scaled))
print('Forest model predictions', forest_model.predict(new_data_scaled))


# In[23]:





# In[36]:





# In[37]:





# In[ ]:




