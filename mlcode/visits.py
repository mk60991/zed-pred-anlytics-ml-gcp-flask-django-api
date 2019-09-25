# -*- coding: utf-8 -*-
"""
Created on Wed May  8 09:12:25 2019

@author: hp
"""



import mysql.connector as sql
import pandas as pd
import numpy as np

#creating connection
db_connection = sql.connect(host='34.85.64.241', database='jtsboard_new', user='jts', password='Jts5678?')
db_cursor = db_connection.cursor()


#fetching sinle "customer_histories table from mysql database
db_cursor.execute('SELECT id, user_id, customer_id, service_price, date FROM customer_histories')
sql_data1 = db_cursor.fetchall()

#creating dataframe for "customer_histories"
df1=pd.DataFrame(sql_data1, columns=["id", "user_id","customer_id","service_price", "date"])
#print(df1.head())

#converting "custmer histories" dataframe to CSV
df1.to_csv('custhist.csv', index=False)

data=pd.read_csv("custhist.csv")
data.shape

#fetch all data of userid 102
user_102=data[data['user_id']==102.0]
user_102=user_102.reset_index()
del user_102['index']
user_102.head()


#print userid 102 all customerid
grouped=user_102.groupby(user_102['customer_id'])

# user_102.groupby(user_102['customer_id']).groups

user_102.shape
user_102.dtypes
a=user_102['date'][0]


#convert date into datetime
user_102['date']=pd.to_datetime(user_102['date'])


#split date into datetime format
user_102['Day']=user_102['date'].apply(lambda x:x.day)
user_102['Month']=user_102['date'].apply(lambda x:x.month)
user_102['Year']=user_102['date'].apply(lambda x:x.year)

#print nan in all columns rowwise
user_102.isnull().sum()

# Here, there are 6 rows which have missing values,so I am dropping it because without date there is no significance of it
user_102=user_102.dropna(subset=['date'])
user_102['date'].isnull().sum()
user_102.isnull().sum()

#convert day, month, year in int
user_102.Day=user_102.Day.apply(lambda x: int(x))
user_102.Month=user_102.Month.apply(lambda x: int(x))
user_102.Year=user_102.Year.apply(lambda x: int(x))


#group by customerid according to dat, month, year and count it
acpred = user_102[['customer_id','Day','Month','Year']].groupby(['Day','Month','Year']).agg('count')

#assign customerid to custvisits
acpred=pd.DataFrame(g)
acpred.columns=['custvisits']

#indexing
acpred=acpred.reset_index()
acpred=acpred.sort_values('Year')


#models

#features
features=acpred.loc[:,['Day','Month','Year']].values
features=pd.DataFrame(features, columns=["day","month","year"])

#labels
labels=acpred.loc[:,['custvisits']].values
labels=pd.DataFrame(labels, columns=["visits"])



#splitting dataset in training and testing dataset
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
features_train,features_test,labels_train,labels_test=train_test_split(features,labels,test_size=0.03, random_state=0)

"""
#features scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
features_train=sc.fit_transform(features_train)
features_test=sc.transform(features_test)
"""

#random forest algo
from sklearn.ensemble import RandomForestRegressor

lr= RandomForestRegressor(n_estimators=300,random_state=0)  
lr.fit(features_train,labels_train.values.reshape(-1,))
predlr = lr.predict(features_test)  


#testing
pred_features1=np.array([[29,4,2019],[28,4,2019],[27,4,2019],[25,4,2019],[31,12,2018],[15,12,2018],[15,11,2018]])
pred_features1=pd.DataFrame(pred_features1, columns=["d","m","y"])
pred_result1=lr.predict(pred_features1).astype('int64')
pred_result1=pd.DataFrame(pred_result1, columns=["visits"])

pred1=pd.concat([pred_features1, pred_result1], axis=1)
pred1=pred1.sort_values('y')

"""
from sklearn import metrics

print('Mean Absolute Error:', metrics.mean_absolute_error(features_test, predlr))  
print('Mean Squared Error:', metrics.mean_squared_error(features_test, predlr))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(features_test, predlr))) 


#create model
#Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
clf=LinearRegression(copy_X=True, fit_intercept=True, n_jobs=5, normalize=False)
clf.fit(features_train,labels_train)

#prediction on features_test
# Predicting the Test set results
predclf=clf.predict(features_test)

df=pd.DataFrame(predclf)


pred_features2=np.array([[29,4,2019],[28,4,2019],[27,4,2019],[25,4,2019]])
pred_features2=pd.DataFrame(pred_features2, columns=["d","m","y"])
pred_result2=clf.predict(pred_features2).astype('int64')
pred_result2=pd.DataFrame(pred_result2, columns=["cv"])

pred2=pd.concat([pred_features2, pred_result2], axis=1)
pred2=pred2.sort_values('y')

"""






data=pd.read_csv("custhist.csv")
data.shape

#fetch all data of userid 102
user_102=data[data['user_id']==102.0]
user_102=user_102.reset_index()
del user_102['index']
user_102.head()


#print userid 102 all customerid
grouped=user_102.groupby(user_102['customer_id'])

# user_102.groupby(user_102['customer_id']).groups

user_102.shape
user_102.dtypes
a=user_102['date'][0]


#convert date into datetime
user_102['date']=pd.to_datetime(user_102['date'])


#split date into datetime format
user_102['Month']=user_102['date'].apply(lambda x:x.month)
user_102['Year']=user_102['date'].apply(lambda x:x.year)

#print nan in all columns rowwise
user_102.isnull().sum()

# Here, there are 6 rows which have missing values,so I am dropping it because without date there is no significance of it
user_102=user_102.dropna(subset=['date'])
user_102['date'].isnull().sum()
user_102.isnull().sum()

#convert day, month, year in int
user_102.Month=user_102.Month.apply(lambda x: int(x))
user_102.Year=user_102.Year.apply(lambda x: int(x))


#group by customerid according to dat, month, year and count it
acvm= user_102[['customer_id','Month','Year']].groupby(['Month','Year']).agg('count')

#assign customerid to custvisits
acvm=pd.DataFrame(g)
acvm.columns=['visits']

#indexing
acvm=acvm.reset_index()
acvm=acvm.sort_values('Year')


#models

#features
featuresx=acvm.loc[:,['Month','Year']].values
featuresx=pd.DataFrame(featuresx, columns=["month","year"])

#labels
labelsy=acvm.loc[:,['visits']].values
labelsy=pd.DataFrame(labelsy, columns=["visits"])



#splitting dataset in training and testing dataset
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
features_train,features_test,labels_train,labels_test=train_test_split(featuresx,labelsy,test_size=0.03, random_state=0)

"""
#features scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
features_train=sc.fit_transform(features_train)
features_test=sc.transform(features_test)
"""

#random forest algo
from sklearn.ensemble import RandomForestRegressor

clf= RandomForestRegressor(n_estimators=300,random_state=0)  
clf.fit(features_train,labels_train.values.reshape(-1,))
predlr = clf.predict(features_test)  


#testing
pred_features1=np.array([[1,2019],[2,2019],[3,2019],[4,2019],[5,2018]])
pred_features1=pd.DataFrame(pred_features1, columns=["m","y"])
pred_result1=clf.predict(pred_features1).astype('int64')
pred_result1=pd.DataFrame(pred_result1, columns=["visits"])

pred1=pd.concat([pred_features1, pred_result1], axis=1)
pred1=pred1.sort_values('y')


from sklearn import metrics

print('Mean Absolute Error:', metrics.mean_absolute_error(features_test, predlr))  
print('Mean Squared Error:', metrics.mean_squared_error(features_test, predlr))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(features_test, predlr))) 

"""
#create model
#Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
clf=LinearRegression(copy_X=True, fit_intercept=True, n_jobs=5, normalize=False)
clf.fit(features_train,labels_train)

#prediction on features_test
# Predicting the Test set results
predclf=clf.predict(features_test)

df=pd.DataFrame(predclf)


pred_features2=np.array([[29,4,2019],[28,4,2019],[27,4,2019],[25,4,2019]])
pred_features2=pd.DataFrame(pred_features2, columns=["d","m","y"])
pred_result2=clf.predict(pred_features2).astype('int64')
pred_result2=pd.DataFrame(pred_result2, columns=["cv"])

pred2=pd.concat([pred_features2, pred_result2], axis=1)
pred2=pred2.sort_values('y')

"""

data=pd.read_csv("custhist.csv")
data.shape

#fetch all data of userid 102
user_102=data[data['user_id']==102.0]
user_102=user_102.reset_index()
del user_102['index']
user_102.head()


#print userid 102 all customerid
grouped=user_102.groupby(user_102['customer_id'])

# user_102.groupby(user_102['customer_id']).groups

user_102.shape
user_102.dtypes
a=user_102['date'][0]


#convert date into datetime
user_102['date']=pd.to_datetime(user_102['date'])


#split date into datetime format
user_102['Week'] = user_102['date'].dt.strftime('%U')
user_102['Year']=user_102['date'].apply(lambda x:x.year)

#print nan in all columns rowwise
user_102.isnull().sum()

# Here, there are 6 rows which have missing values,so I am dropping it because without date there is no significance of it
user_102=user_102.dropna(subset=['date'])
user_102['date'].isnull().sum()
user_102.isnull().sum()

#convert week, year in int

user_102.Week=user_102.Week.apply(lambda x: int(x))
user_102.Year=user_102.Year.apply(lambda x: int(x))

#group by customerid according to dat, month, year and count it
acvw = user_102[['customer_id','Week', 'Year']].groupby(['Week','Year']).agg('count')

#assign customerid to custvisits
acvw =pd.DataFrame(g)
acvw .columns=['custvisits']

#indexing
acvw =acvw .reset_index()
acvw =acvw .sort_values('Year')


#models

#features
features=acvw .loc[:,['Week','Year']].values
features=pd.DataFrame(features, columns=["week","year"])

#labels
labels=acvw .loc[:,['custvisits']].values
labels=pd.DataFrame(labels, columns=["visits"])



#splitting dataset in training and testing dataset
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
features_train,features_test,labels_train,labels_test=train_test_split(features,labels,test_size=0.03, random_state=0)

"""
#features scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
features_train=sc.fit_transform(features_train)
features_test=sc.transform(features_test)
"""

#random forest algo
from sklearn.ensemble import RandomForestRegressor

lr= RandomForestRegressor(n_estimators=300,random_state=0)  
lr.fit(features_train,labels_train.values.reshape(-1,))
predlr = lr.predict(features_test)  


#testing
pred_features1=np.array([[11,2019],[12,2019],[15,2019],[16,2019],[17,2019],[18,2019],[19,2019],[13,2019],[14,2019]])
pred_features1=pd.DataFrame(pred_features1, columns=["w","y"])
pred_result1=lr.predict(pred_features1).astype('int64')
pred_result1=pd.DataFrame(pred_result1, columns=["visits"])

pred1=pd.concat([pred_features1, pred_result1], axis=1)
pred1=pred1.sort_values('y')

"""
from sklearn import metrics

print('Mean Absolute Error:', metrics.mean_absolute_error(features_test, predlr))  
print('Mean Squared Error:', metrics.mean_squared_error(features_test, predlr))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(features_test, predlr))) 


#create model
#Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
clf=LinearRegression(copy_X=True, fit_intercept=True, n_jobs=5, normalize=False)
clf.fit(features_train,labels_train)

#prediction on features_test
# Predicting the Test set results
predclf=clf.predict(features_test)

df=pd.DataFrame(predclf)


pred_features2=np.array([[29,4,2019],[28,4,2019],[27,4,2019],[25,4,2019]])
pred_features2=pd.DataFrame(pred_features2, columns=["d","m","y"])
pred_result2=clf.predict(pred_features2).astype('int64')
pred_result2=pd.DataFrame(pred_result2, columns=["cv"])

pred2=pd.concat([pred_features2, pred_result2], axis=1)
pred2=pred2.sort_values('y')

"""
import pickle
import numpy as np
#serializing our model to a file called model.pkl
pickle.dump(lr, open("modelvweeks.pkl","wb"))
#loading a model from a file called model.pkl

model_colvweeks = list(features.columns)
print(model_colvweeks)
pickle.dump(model_colvweeks, open('model_columnsvw.pkl',"wb"))
print("Models columns dumped!")



#serializing our model to a file called model.pkl
pickle.dump(clf, open("modelvmonths.pkl","wb"))
#loading a model from a file called model.pkl

model_colvmonths = list(featuresx.columns)
print(model_colvmonths)
pickle.dump(model_colvmonths, open('model_columnsvm.pkl',"wb"))
print("Models columns dumped!")



#serializing our model to a file called model.pkl
pickle.dump(lr, open("modelvdays.pkl","wb"))
#loading a model from a file called model.pkl

model_colvdays = list(features.columns)
print(model_colvdays)
pickle.dump(model_colvdays, open('model_columnsvd.pkl',"wb"))
print("Models columns dumped!")


with open('modelvdays.pkl', 'rb') as handle:
    reg = pickle.load(handle)   

p_features3=np.array([[29,4,2019],[28,4,2019],[27,4,2019],[25,4,2019]])
p_result3=reg.predict(p_features3)




