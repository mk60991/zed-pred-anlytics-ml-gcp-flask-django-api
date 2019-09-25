# -*- coding: utf-8 -*-
"""
Created on Wed May  8 09:06:42 2019

@author: hp
"""


import mysql.connector as sql
import pandas as pd
import numpy as np

#creating connection
db_connection = sql.connect(host='34.85.64.241', database='jtsboard_new', user='jts', password='Jts5678?')
db_cursor = db_connection.cursor()

#fetching sinle "customer_histories table from mysql database
db_cursor.execute('SELECT id, user_id,date,customer_id FROM customer_histories')
sql_data1 = db_cursor.fetchall()

#creating dataframe for "customer_histories"
df1=pd.DataFrame(sql_data1, columns=["ch_id","user_id","date", "customer_id"])
#print(df1.head())

#converting "custmer histories" dataframe to CSV
df1.to_csv('sales_custxyz.csv', index=False)


#fetching  "note services" table from mysql database

db_cursor.execute('SELECT id, user_id, customer_id, customer_history_id, service_id, employee_id, service_price FROM note_services')
sql_data2 = db_cursor.fetchall()

# converting to "note services" dataframe
df2=pd.DataFrame(sql_data2, columns=["ns_id", "user_id", "customer_id", "ch_id", "service_id", "employee_id", "service_price"])

#converting "note services"  dataframe to csv
df2.to_csv('sales_servicexyz.csv', index=False)



#fetching  "note products" table from mysql database

db_cursor.execute('SELECT id, user_id, customer_id, customer_history_id, product_id, employee_id, sale_price FROM note_products')
sql_data3= db_cursor.fetchall()


# converting to "note products" table to  dataframe

df3=pd.DataFrame(sql_data3, columns=["np_id", "user_id", "customer_id", "ch_id", "product_id", "employee_id", "sale_price"])

#converting "note products""  dataframe to csv

df3.to_csv('sales_productxyz.csv',index=False)



#fetching  "note tickets" table from mysql database

db_cursor.execute('SELECT id, user_id, customer_id, customer_history_id, ticket_id, ticket_price,employee_id FROM note_tickets ')
sql_data4 = db_cursor.fetchall()

# converting to "note tickets" table to  dataframe

df4=pd.DataFrame(sql_data4, columns=["nt_id", "user_id", "customer_id", "ch_id","ticket_id", "ticket_price","employee_id"])

#converting "note tickets"  dataframe to csv

df4.to_csv('sales_ticketxyz.csv', index=False)



# read all converted csv file and stored in variable
a1=pd.read_csv("sales_custxyz.csv")
a2=pd.read_csv("sales_servicexyz.csv")
a3=pd.read_csv("sales_productxyz.csv")
a4=pd.read_csv("sales_ticketxyz.csv")


#merging a1 nad a2 dataset
merge1=pd.merge(a1,a2, on=['customer_id','ch_id','user_id'], how='outer')

#merging merge1 and a3 dataset
merge2=pd.merge(merge1,a3, on=['customer_id','ch_id','user_id'], how='outer')

#merging merge2 and a4 dataset
merge3=pd.merge(merge2,a4, on=['customer_id','ch_id','user_id'], how='outer')

# converting final merge data to csv
merge3.to_csv("salesxyz.csv", index=False)

#checking nan value in each column
#print(merge3.isnull().sum())

# read csv file from save place
jts_sales=pd.read_csv("salesxyz.csv")

#converting it into dataframe 
jts_sales=pd.DataFrame(jts_sales)

#fetching nan data from jts_sales
#print(jts_sales.isnull().sum())
#cust_service3=jts_sales.loc[jts_sales['date'] =='2018-08-27' ]
#features
data_clean=jts_sales.iloc[:,[1,2,7,11,14]].values

data_clean=pd.DataFrame(data_clean)


#drop nan row
#data_clean.dropna(axis=0, inplace=True)
#replace nan with '0' from date column 
#data_clean= data_clean.replace(np.nan, 0)

#drop useless row from row according to index
clean=data_clean.drop(data_clean.index[2919:])
#clean[0] = clean[0].apply(pd.to_datetime)
#clean=pd.DataFrame(clean)

#print all nan columns from 0,1,2,3 on date having nan
#null_columns=clean.columns[clean.isnull().any()]
#clean[null_columns].isnull().sum()
#dropp1=clean.drop[clean[0].isnull()][null_columns]

#print all nan columns from 0,1,2,3 on date having nan
clean= clean[pd.notnull(clean[1])]

clean= clean[pd.notnull(clean[0])]

#features
features=clean.iloc[:,[0,1]].values
features=pd.DataFrame(features)

"""
features=pd.to_datetime(features[0])
features= features.replace(np.nan, 0)
#convert date column to datetime and split to individuaal 'year', 'month', 'date'
features[0] = features[0].apply(pd.to_datetime)
features['year'] = [i.year for i in features[0]]
features['month'] = [i.month for i in features[0]]
features['day'] = [i.day for i in features[0]]
"""
#labels
labels=clean.iloc[:,[2,3,4]].values
labels=pd.DataFrame(labels)

#jts_labels[1] = jts_labels[1].replace(np.nan, 0)
#jts_labels=pd.DataFrame(jts_labels)
#result = pd.concat([features, labels[0]], axis=1, join='outer')
  
#labels_cleaning
#cleaning service sales column and extract numerical data
#clean 1st column and extract numerical part
#fill blank row 
labels.iloc[:,0] = labels.iloc[:,0].fillna("b''")


ls=[]
for i in  labels.iloc[:,0]:
    temp = i.split('\\')
    temp = temp[0][1:].strip("'").split(',')
    add = ''
    for j in temp:
        add += j
    if add!='':
        ls.append(float(add))
    else:
        ls.append(0.0)

labels.iloc[:,0] = ls



#clean 1st column and extract numeical part
    #get pure numerical part
labels.iloc[:,1] = labels.iloc[:,1].fillna("bytearray(b''")


ls2=[]
for i2 in  labels.iloc[:,1]:
    temp2 = i2.split('\\')
    temp2 = temp2[0][11:].strip("'").split(',')
    add2 = ''
    for j2 in temp2:
        add2 += j2
    if add2!='':
        ls2.append(add2)
    else:
        ls2.append(0.0) 
labels.iloc[:,1] = ls2

#1 st column extract [pure numerical part]
labels.iloc[:,1] = labels.iloc[:,1].str.strip().str.lower().str.replace(")","").str.replace("'", "")
#handle missing values in column1
    #repplace nan with 0
labels[1] = labels[1].replace(np.nan, 0)
#replace empty rows in 1st column with 0
labels[1] = labels[1].replace('', 0)



#extract second column with numerical part 
labels.iloc[:,2] = labels.iloc[:,2].fillna("bytearray(b")


ls3=[]
for i3 in  labels.iloc[:,2]:
    temp3 = i3.split('\\')
    temp3 = temp3[0][11:].strip("'").split(',')
    add3 = ''
    for j3 in temp3:
        add3 += j3
    if add3!='':
        ls3.append(add3)
    else:
        ls3.append(0.0)
labels.iloc[:,2] = ls3

# clean second column more with redundant word
#jts_labels.iloc[:,2] = jts_labels.iloc[:,2].apply(lambda x: x.replace(")","").replace("'",""))
labels.iloc[:,2] = labels.iloc[:,2].str.strip().str.lower().str.replace(")","").str.replace("'", "")
#handle missing values in column1
    #repplace nan with 0
labels[2] = labels[2].replace(np.nan, 0)
#replace empty rows in 1st column with 0
labels[2] = labels[2].replace('', 0)


#convert all columns '0','1','2' such columns to numeric either in 'int64' and 'float64'
#for i in range(0, len(labels.columns)):
#    labels.iloc[:,i] = pd.to_numeric(labels.iloc[:,i], errors='ignore')

#convert all columns '0','1','2' such columns to numeric either in dtype('O') 'int64' and 'float64'
cols = [0,1,2]
labels[cols] = labels[cols].applymap(np.int64)

#prediction start

#labels

#finally creates 'total_sales' column by adding such columns '0','1','2'
labels["total_sales"]=labels.sum(axis=1, skipna=True)
#df_labels.drop(["total_sales"], axis=1, inplace=True)


jtsx=features.iloc[:,[0,1]].values
#jtsx =pd.to_datetime(jtsx)
jtsx=pd.DataFrame(jtsx,columns=["u","d"])

#calculating total sales by adding service priec ticket price and sale price
jtsy=labels.loc[:,["total_sales"]].values.astype('int64')
jtsy=pd.DataFrame(jtsy, columns=['s'])

#concat two columns from dataframe
jts=pd.concat([jtsx, jtsy], axis=1)

#converting it into csv
jts.to_csv("jtstsales.csv", index=False)

#daywise


#reading csv file
data = pd.read_csv('jtstsales.csv')

#fetch only nan free data
data = data[pd.notnull(data['d'])]
#fetch all data of '102'
user_102=data[data['u']==102.0]

#reset index
user_102=user_102.reset_index()
#delete index
del user_102['index']
user_102.head()

#print single date
a=user_102['d'][0]

#convert date into datetime
user_102['d']=pd.to_datetime(user_102['d'])

#split date into day, month, year columns
user_102['Day']=user_102['d'].apply(lambda x:x.day)
user_102['Month']=user_102['d'].apply(lambda x:x.month)
user_102['Year']=user_102['d'].apply(lambda x:x.year)

user_102.head()

#convert day , month, year into integer
user_102.Day=user_102.Day.apply(lambda x: int(x))
user_102.Month=user_102.Month.apply(lambda x: int(x))
user_102.Year=user_102.Year.apply(lambda x: int(x))

#drop nan all row acording to date columns
user_102=user_102.dropna(subset=['d'])

#sum datewise all columns
tsdg = user_102[['s','Day','Month','Year']].groupby(['Day','Month','Year']).sum()


tsdg=pd.DataFrame(tsdg)
tsdg=tsdg.reset_index()
tsdg=tsdg.sort_values('Year')
 
#features
featuresx=tsdg.loc[:,["Day","Month","Year"]].values
featuresx=pd.DataFrame(featuresx, columns=["day","month","year"])

#labels
labelsy=tsdg.loc[:,"s"].values
labelsy=pd.DataFrame(labelsy, columns=["sales"])

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


from sklearn.ensemble import RandomForestRegressor

lr= RandomForestRegressor(n_estimators=300,random_state=0)  
lr.fit(features_train,labels_train.values.reshape(-1,))
predlr = lr.predict(features_test)  



pred_features1=np.array([[29,4,2019],[28,4,2019],[27,4,2019],[25,4,2019]])
pred_features1=pd.DataFrame(pred_features1, columns=["d","m","y"])
pred_result1=lr.predict(pred_features1).astype('int64')
pred_result1=pd.DataFrame(pred_result1, columns=["s"])

pred1=pd.concat([pred_features1, pred_result1], axis=1)
pred1=pred1.sort_values('y')

"""
from sklearn import metrics

print('Mean Absolute Error:', metrics.mean_absolute_error(features_test, predlr))  
print('Mean Squared Error:', metrics.mean_squared_error(features_test, predlr))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(features_test, predlr))) 

"""

data = pd.read_csv('jtstsales.csv')

#fetch only nan free data
data = data[pd.notnull(data['d'])]
user_102=data[data['u']==102.0]

#indexing
user_102=user_102.reset_index()
del user_102['index']
user_102.head()

#ist date value
a=user_102['d'][0]

#datetime
user_102['d']=pd.to_datetime(user_102['d'])

#splitting
user_102['Month']=user_102['d'].apply(lambda x:x.month)
user_102['Year']=user_102['d'].apply(lambda x:x.year)

#head value
user_102.head()

#month and year to int
user_102.Month=user_102.Month.apply(lambda x: int(x))
user_102.Year=user_102.Year.apply(lambda x: int(x))

#drop nan from date
user_102=user_102.dropna(subset=['d'])


#groupping sales month and year
tsmg = user_102[['s','Month','Year']].groupby(['Month','Year']).sum()

#dataframe
tsmg=pd.DataFrame(tsmg)
tsmg=tsmg.reset_index()
tsmg=tsmg.sort_values('Year')
 
#features
featuresm=tsmg.loc[:,["Month","Year"]].values
featuresm=pd.DataFrame(featuresm, columns=["month","year"])


labelsm=tsmg.loc[:,"s"].values
labelsm=pd.DataFrame(labelsm, columns=["sales"])

#splitting dataset in training and testing dataset
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
features_train,features_test,labels_train,labels_test=train_test_split(featuresm,labelsm,test_size=0.01,random_state=0)

"""
#features scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
features_train=sc.fit_transform(features_train)
features_test=sc.transform(features_test)
"""


from sklearn.ensemble import RandomForestRegressor

clf= RandomForestRegressor(n_estimators=300,random_state=0)  
clf.fit(features_train,labels_train.values.reshape(-1,))
predclf = clf.predict(features_test)  



pred_featuresm=np.array([[1,2019],[2,2019],[3,2019],[4,2019],[5,2019],[6,2019],[7,2019]])
pred_featuresm=pd.DataFrame(pred_featuresm, columns=["m","y"])
pred_resultm=clf.predict(pred_featuresm).astype('int64')
pred_resultm=pd.DataFrame(pred_resultm, columns=["s"])

predm=pd.concat([pred_featuresm, pred_resultm], axis=1)
predm=predm.sort_values('y')

"""
from sklearn import metrics

print('Mean Absolute Error:', metrics.mean_absolute_error(features_test, predclf))  
print('Mean Squared Error:', metrics.mean_squared_error(features_test, predclf))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(features_test, predclf))) 


#create model
#Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
clf=LinearRegression(copy_X=True, fit_intercept=True, n_jobs=10, normalize=False)
clf.fit(features_train,labels_train)

#prediction on features_test
# Predicting the Test set results
predclf=clf.predict(features_test).astype('int64')

df=pd.DataFrame(predclf).astype('int64')


pred_features2=np.array([[1,2019],[2,2019],[3,2019],[4,2019]])
pred_features2=pd.DataFrame(pred_features2, columns=["d","m","y"])
pred_result2=clf.predict(pred_features2).astype('int64')
pred_result2=pd.DataFrame(pred_result2, columns=["s"])

pred2=pd.concat([pred_features2, pred_result2], axis=1)
pred2=pred2.sort_values('y')

"""


data = pd.read_csv('jtstsales.csv')

#fetch only nan free data
data = data[pd.notnull(data['d'])]
user_102=data[data['u']==102.0]

user_102=user_102.reset_index()
del user_102['index']
user_102.head()


a=user_102['d'][0]

#converting date inti datetime
user_102['d']=pd.to_datetime(user_102['d'])

#splitting date into week and year
user_102['Week'] = user_102['d'].dt.strftime('%U')
user_102['Year']=user_102['d'].apply(lambda x:x.year)

#head value
user_102.head()

#splitting weekwise
user_102.Week=user_102.Week.apply(lambda x: int(x))
user_102.Year=user_102.Year.apply(lambda x: int(x))

#drop nan from date columns
ser_102=user_102.dropna(subset=['d'])

#groupby sales and week and year
tswg = user_102[['s','Week','Year']].groupby(['Week','Year']).sum()


#indexing and dataframe
tswg =pd.DataFrame(tswg )
tswg =tswg  .reset_index()
tswg =tswg .sort_values('Year')


featuresw=tswg .loc[:,["Week","Year"]].values
featuresw=pd.DataFrame(featuresw, columns=["week","year"])


labelsw=tswg .loc[:,"s"].values
labelsw=pd.DataFrame(labelsw, columns=["sales"])

#splitting dataset in training and testing dataset
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
features_train,features_test,labels_train,labels_test=train_test_split(featuresw,labelsw,test_size=0.01,random_state=0)

#features scaling
#from sklearn.preprocessing import StandardScaler
#sc=StandardScaler()
#features_train=sc.fit_transform(features_train)
#features_test=sc.transform(features_test)

#create model
#Fitting Multiple Linear Regression to the Training set
from sklearn.ensemble import RandomForestRegressor

clfw= RandomForestRegressor(n_estimators=300,random_state=0)  
clfw.fit(features_train,labels_train.values.reshape(-1,))
predclfw = clfw.predict(features_test)  



pred_featuresw=np.array([[11,2019],[12,2019],[13,2019],[14,2019],[15,2019],[16,2019],[17,2019],[18,2019]])
pred_featuresw=pd.DataFrame(pred_featuresw, columns=["w","y"])
pred_resultw=clfw.predict(pred_featuresw).astype("int64")
pred_resultw=pd.DataFrame(pred_resultw, columns=["s"])

predw=pd.concat([pred_featuresw, pred_resultw], axis=1)
predw=predw.sort_values('y')




"""
#create model
#Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
lrw=LinearRegression(copy_X=True, fit_intercept=True, n_jobs=20, normalize=False)
lrw.fit(features_train,labels_train)

#prediction on features_test
# Predicting the Test set results
predlrw=lrw.predict(features_test)

df=pd.DataFrame(predlrw).astype('int64')


pred_featureslw=np.array([[16,2019],[17,2019],[18,2019],[19,2019],[20,2019],[24,2019],[25,2019],[27,2019]])
pred_featureslw=pd.DataFrame(pred_featureslw, columns=["w","y"])
pred_resultlw=lrw.predict(pred_featureslw).astype("int64")
pred_resultlw=pd.DataFrame(pred_resultlw, columns=["s"])

predlw=pd.concat([pred_featureslw, pred_resultlw], axis=1)
predlw=predlw.sort_values('y')

"""

# Save your model

import numpy as np
import pickle
#serializing our model to a file called model.pkl
pickle.dump(clfw, open("modelweeks.pkl","wb"))
#loading a model from a file called model.pkl

model_colweeks = list(featuresw.columns)
print(model_colweeks )
pickle.dump(model_colweeks, open('model_columnsw.pkl',"wb"))
print("Models columns dumped!")

#serializing our model to a file called model.pkl
pickle.dump(lr, open("modeldays.pkl","wb"))
#loading a model from a file called model.pkl

model_coldays = list(featuresx.columns)
print(model_coldays)
pickle.dump(model_coldays, open('model_columnsd.pkl',"wb"))
print("Models columns dumped!")

#serializing our model to a file called model.pkl
pickle.dump(clf, open("modelmonths.pkl","wb"))
#loading a model from a file called model.pkl

model_colmonths = list(featuresm.columns)
print(model_colmonths)
pickle.dump(model_colmonths, open('model_columnsm.pkl',"wb"))
print("Models columns dumped!")

