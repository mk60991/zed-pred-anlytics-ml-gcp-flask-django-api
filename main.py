# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 18:16:52 2019

@author: hp
"""



from flask import Flask, request, jsonify
from sklearn.externals import joblib
import traceback
import pandas as pd
import numpy as np
from flask import request
from datetime import datetime

from flask_cors import CORS, cross_origin
from wtforms import TextField,TextAreaField, SubmitField
from wtforms.validators import Required
 
import sys
import os
import datetime
import calendar

import pickle

import json
# Preparing the Classifier
#monthwise

cur_dir = os.path.dirname('__file__')


#total sales
#monthwise
regressor = pickle.load(open(os.path.join(cur_dir,
            'pkl_objects/modelmonth_s.pkl'), 'rb'))
model_colm = pickle.load(open(os.path.join(cur_dir,
            'pkl_objects/model_columns_sm.pkl'),'rb')) 

#actual
print("mmmmmmmmmmmmmmmm")
actual = pickle.load(open(os.path.join(cur_dir,
            'pkl_objects/modelmonth_sact.pkl'), 'rb'))


#customer visits
#monthwise

#monthwise
clfvm = pickle.load(open(os.path.join(cur_dir,
			'pkl_objects/modelvmonths.pkl'), 'rb'))
model_colvm = pickle.load(open(os.path.join(cur_dir,
			'pkl_objects/model_columns_cvm.pkl'),'rb')) 

#actual cust visits
act_cv = pickle.load(open(os.path.join(cur_dir,
            'pkl_objects/modelcvmonths_act.pkl'), 'rb'))


#expenses monthwise
regressor_exp = pickle.load(open(os.path.join(cur_dir,
                        'pkl_objects/model_mexp.pkl'), 'rb'))
model_colexp = pickle.load(open(os.path.join(cur_dir,
                        'pkl_objects/model_columns_expm.pkl'),'rb'))


#actual month expenses
act_expm = pickle.load(open(os.path.join(cur_dir,
            'pkl_objects/model_expm_act.pkl'), 'rb'))



#total sales


#daywise
clf = pickle.load(open(os.path.join(cur_dir,
			'pkl_objects/modeldays.pkl'), 'rb'))
model_cold = pickle.load(open(os.path.join(cur_dir,
			'pkl_objects/model_columnsd.pkl'),'rb')) 

#weekwise
clfweek = pickle.load(open(os.path.join(cur_dir,
			'pkl_objects/modelweeks.pkl'), 'rb'))
model_colw = pickle.load(open(os.path.join(cur_dir,
			'pkl_objects/model_columnsw.pkl'),'rb')) 



#customer visits
#daywise
clfvd = pickle.load(open(os.path.join(cur_dir,
			'pkl_objects/modelvdays.pkl'), 'rb'))
model_colvd = pickle.load(open(os.path.join(cur_dir,
			'pkl_objects/model_columnsvd.pkl'),'rb')) 


#weekwise
clfvw = pickle.load(open(os.path.join(cur_dir,
			'pkl_objects/modelvweeks.pkl'), 'rb'))
model_colvw = pickle.load(open(os.path.join(cur_dir,
			'pkl_objects/model_columnsvw.pkl'),'rb')) 













# Your API definition
app = Flask(__name__)

#for localhost
cors = CORS(app, resources={r"/": {"origins": "http://localhost:5000"}})

#monthwise
@app.route('/month', methods=['POST'])
@cross_origin(origin='localhost',headers=['Content- Type','Authorization'])


def predmonth():
    
    #print(request)
    #if regressor:
    try:
        
        
        json1= request.json
        query1 = pd.get_dummies(pd.DataFrame(json1))
        query1 = query1.reindex(columns=model_colm, fill_value=0)
        #print(query1)
        copy_query1 = query1.copy(deep=True)
        copy_query1['month'] = copy_query1['month'].astype(str) + '月'
  

        print("JJJJJJJJJJJJJJJJJJ")
        print(query1)
        print("SSSSSSSSSSSSSSSSSS")
        
        
        #mname = query1['m'].apply(lambda x: calendar.month_abbr[x])
        mname = copy_query1['month']
        #print(mname)
        #print(query1)
        dic=[]
        mname=pd.DataFrame(mname)
        print(mname)
        for element in json1:
            month_no=element['month']
            rows=actual.loc[actual['Month'] == month_no]
            print("zxzxxxxxxxxxxxxx")
            print(rows)
            for ik in rows['s']:
                dic.append(ik)
               
                
        print(dic)
        
        
        #mname=pd.DataFrame(mname)
        #print(mname)
         
        prediction1 = (regressor.predict(query1).astype('int64'))
        

        #print(prediction1)
        #print(prediction)
        prediction1=pd.DataFrame(prediction1, columns=["sales_pred"])
        print("predddddddddddddddddddddddd")
        print(prediction1)
        
        actual_data=pd.DataFrame(dic, columns=["sales_act"])
        print("actualllllllllllllllllllllll")
        print(actual_data)
        
        
        #df_out = pd.merge(actual_data,prediction1,how = 'left',left_index = True, right_index = True)
        
        df_out=pd.concat([actual_data,prediction1],axis=1)
        print("concattttttttttttttttttt")
        print(df_out)
        
        analytics=[]
        for i in range(0,12):
            if i<=7:
                print("ddddddddddddddddddddddd")
                
                x=df_out["sales_act"][i]
                analytics.append(x)
                print("ddddddddddddddd")
                
            else:
                print("eeeeeeeeeeeeeee")
                x=df_out["sales_pred"][i]
                print("sdddddddddddddddd")
                analytics.append(x)
                print("eeeeeeeeeeeeeeeeeee")
        
        print("daruuuuuuuuuuuuuuuuu")
        print(analytics)
        
        df_act_pred=pd.DataFrame(analytics,columns=["sales"])
        
        df_act_pred['sales'] = df_act_pred['sales'].apply(lambda x: x/1000)

        con=pd.concat([mname,df_act_pred], axis=1)
        print("IIIIIIIIIII")
        print(con)
        print("NNNNNNNNNN")
        df=pd.DataFrame(con)
        #df.set_index('m')['0'].to_dict()
        df.set_index('month')['sales'].to_dict()

        count = df.shape[0]
        #print(count)
   
        #print("LLLLLLLLLLOOO") 
        #print(df)
        #print("KKKKKKKKKKKKK")
 

        months = df['month'].tolist()
        sales = df['sales'].tolist()

        print("PPPPPPPPPP")
        print(months)
        print(sales)
        print("KKKKKKKKKKKKK")

        list_of_dicts = []
        D={}

        #add a key and setup for a sub-dictionary

        for i in range(count):
            D[i] = {}
            D[i]['x']=months[i]
            D[i]['value']=sales[i]
            list_of_dicts.append(D[i])

        print("BBBBBBBBBBBBBB")
        print(list_of_dicts)
        print("LLLLLLLLLLLLLL")

        # convert into JSON:
        json_dict = json.dumps(list_of_dicts,ensure_ascii=False)

        # the result is a JSON string:
        print("LLLLLLLLLLOOO")
        print(json_dict)
        print("KKKKKKKKKKKKK")


        #dict(zip(df.m, df.s))
        
        #from collections import defaultdict
        #mydict = defaultdict(list)
        #for k, v in zip(df.m.values,df.s.values):
        #    mydict[k].append(v)
        #    mydict[k]="x"
        #    mydict[v]="v"

            
        #print("LLLLLLLLLLOOO") 
        #print(mydict)
        #print(k)
        #print(v)
        #print("LLLLLLLLLL")
        
        #df.groupby('name')[['value1','value2']].apply(lambda g: g.values.tolist()).to_dict()

            
        
        #return jsonify({'prediction': str(prediction1)})
        #t = "cheese"
        
        
        
        #return(t)
        return(json_dict)
        
    except:

        return jsonify({'trace': traceback.format_exc()})



#monthwise
@app.route('/monthv', methods=['POST'])
@cross_origin(origin='localhost',headers=['Content- Type','Authorization'])


def predvmonths():
    
    #print(request)
    try:
        json1= request.json
        query1 = pd.get_dummies(pd.DataFrame(json1))
        query1 = query1.reindex(columns=model_colvm, fill_value=0)
        #print(query1)
        copy_query1 = query1.copy(deep=True)
        copy_query1['month'] = copy_query1['month'].astype(str) + '月'
        

        print("JJJJJJJJJJJJJJJJJJ")
        print(query1)
        print("SSSSSSSSSSSSSSSSSS")
        
        
        #mname = query1['m'].apply(lambda x: calendar.month_abbr[x])
        mname = copy_query1['month']
        #print(mname)
        #print(query1)
        
        mname=pd.DataFrame(mname)
        print(mname)
         
        
        dic=[]
       
        for element in json1:
            month_no=element['month']
            print(element)
            print(month_no)
            rows=act_cv.loc[act_cv['Month'] == month_no]
            for ik in rows['visits']:
                print("ppppppppppp")
                print(ik)
                dic.append(ik)
               
                
        print(dic)
        
        prediction1 = (clfvm.predict(query1))
        print(prediction1)
        #print(prediction)
        prediction1=pd.DataFrame(prediction1, columns=["visits_pred"])
        
        
        actual_data=pd.DataFrame(dic, columns=["visits_act"])
        print("actualllllllllllllllllllllll")
        print(actual_data)
        
        
        #df_out = pd.merge(actual_data,prediction1,how = 'left',left_index = True, right_index = True)
        
        df_out=pd.concat([actual_data,prediction1],axis=1)
        print("concattttttttttttttttttt")
        print(df_out)
        
        analytics=[]
        for i in range(0,12):
            if i<=7:
                print("ddddddddddddddddddddddd")
                
                x=df_out["visits_act"][i]
                analytics.append(x)
                print("ddddddddddddddd")
                
            else:
                print("eeeeeeeeeeeeeee")
                x=df_out["visits_pred"][i]
                print("sdddddddddddddddd")
                analytics.append(x)
                print("eeeeeeeeeeeeeeeeeee")
        
        print("daruuuuuuuuuuuuuuuuu")
        print(analytics)
        
        df_act_pred=pd.DataFrame(analytics,columns=["visits"])
        
        
        #prediction1['visits'] = prediction1['s'].apply(lambda x: x/1000)

        con=pd.concat([mname,df_act_pred], axis=1)
        """print("IIIIIIIIIII")
        print(con)
        print("NNNNNNNNNN")"""
        df=pd.DataFrame(con)
        #df.set_index('m')['0'].to_dict()
        df.set_index('month')['visits'].to_dict()

        count = df.shape[0]
        #print(count)
   
        #print("LLLLLLLLLLOOO") 
        #print(df)
        #print("KKKKKKKKKKKKK")
 

        months = df['month'].tolist()
        sales = df['visits'].tolist()

        print("PPPPPPPPPP")
        print(months)
        print(sales)
        print("KKKKKKKKKKKKK")

        list_of_dicts = []
        D={}

        #add a key and setup for a sub-dictionary

        for i in range(count):
            D[i] = {}
            D[i]['x']=months[i]
            D[i]['value']=sales[i]
            list_of_dicts.append(D[i])

        """print("BBBBBBBBBBBBBB")
        print(list_of_dicts)
        print("LLLLLLLLLLLLLL")"""

        # convert into JSON:
        json_dict = json.dumps(list_of_dicts,ensure_ascii=False)

        # the result is a JSON string:
        """print("LLLLLLLLLLOOO")
        print(json_dict)
        print("KKKKKKKKKKKKK")"""


        #dict(zip(df.m, df.s))
        
        #from collections import defaultdict
        #mydict = defaultdict(list)
        #for k, v in zip(df.m.values,df.s.values):
        #    mydict[k].append(v)
        #    mydict[k]="x"
        #    mydict[v]="v"

            
        #print("LLLLLLLLLLOOO") 
        #print(mydict)
        #print(k)
        #print(v)
        #print("LLLLLLLLLL")
        
        #df.groupby('name')[['value1','value2']].apply(lambda g: g.values.tolist()).to_dict()

            
        
        #return jsonify({'prediction': str(prediction1)})
        #t = "cheese"
        #return(t)
        return(json_dict)
        
    except:

        return jsonify({'trace': traceback.format_exc()})




#else:
#    print ('Train the model first')
#    return ('No model here to use')


#monthwise expenses

@app.route('/monthexp', methods=['POST'])
@cross_origin(origin='localhost',headers=['Content- Type','Authorization'])


def predmonthexp():
    #print(request)
    try:
        print("HELLLO")
        json1= request.json
        length = len(json1)
        
        query1 = pd.get_dummies(pd.DataFrame(json1))
        query1 = query1.reindex(columns=model_colexp, fill_value=0)
        #print(query1)
        copy_query1 = query1.copy(deep=True)
        copy_query1['month'] = copy_query1['month'].astype(str) + '月'
                
                
        print("JJJJJJJJJJJJJJJJJJ")
        print(query1)
        print("SSSSSSSSSSSSSSSSSS")
                
                
        #mname = query1['m'].apply(lambda x: calendar.month_abbr[x])
        mname = copy_query1['month']
        #print(mname)

        
        mname=pd.DataFrame(mname)
        print(mname)
        
        dic=[]
       
        for element in json1:
            month_no=element['month']
            print(element)
            print(month_no)
            rows=act_expm.loc[act_expm['Month'] == month_no]
            for ik in rows['expenses']:
                print("ppppppppppp")
                print(ik)
                dic.append(ik)
               
        
        prediction1 = (regressor_exp.predict(query1).astype('int64'))

        print("JJJJJJJJJJ")
        print(prediction1)
        print("KkKKKKKK")
        #print(prediction)
        prediction1=pd.DataFrame(prediction1, columns=["expenses_pred"])
        
        
        actual_data=pd.DataFrame(dic, columns=["expenses_act"])
        print("actualllllllllllllllllllllll")
        print(actual_data)
        
        
        df_out=pd.concat([actual_data,prediction1],axis=1)
        print("concattttttttttttttttttt")
        print(df_out)
        
        analytics=[]
        for i in range(0,12):
            print("-------------------------------------------------------------")
            if i==0:
                print("eeeeeeeeeeeeeee0")
                x=df_out["expenses_pred"][i]
                print("sddddddddddddddd0")
                analytics.append(x)
                print("eeeeeeeeeeeeeeeeee0")
                
            elif i in range(1,6):
                print("ddddddddddddddddddddddd16")
                
                x=df_out["expenses_act"][i-1]
                analytics.append(x)
                print("ddddddddddddddd16")
                
            elif i==6:
                print("eeeeeeeeeeeeeee6")
                x=df_out["expenses_pred"][i]
                print("sdddddddddddddddd6")
                analytics.append(x)
                print("eeeeeeeeeeeeeeeeeee6")
                
            elif i==7:
                print("ddddddddddddddddddddddd7")
                
                x=df_out["expenses_act"][i-2]
                analytics.append(x)
                print("ddddddddddddddd7")
                
            
            
            else:
                print("eeeeeeeeeeeeeee")
                x=df_out["expenses_pred"][i]
                print("sdddddddddddddddd")
                analytics.append(x)
                print("eeeeeeeeeeeeeeeeeee")
        
        print("daruuuuuuuuuuuuuuuuu")
        print(analytics)
        
        df_act_pred=pd.DataFrame(analytics,columns=["expenses"])
        
        #prediction1['expenses'] = prediction1['expenses'].apply(lambda x: x/1000)
        
        con=pd.concat([mname, df_act_pred], axis=1)
        print("IIIIIIIIIII")
        print(con)
        print("NNNNNNNNNN")
        df=pd.DataFrame(con)
        #df.set_index('m')['0'].to_dict()
        df.set_index('month')['expenses'].to_dict()
        
        count = df.shape[0]
        #print(count)
        
        #print("LLLLLLLLLLOOO")
        #print(df)
        #print("KKKKKKKKKKKKK")
        
        
        #months = df['month'].tolist()
        expenses = df['expenses'].tolist()
        
        
        ###################################################################
        
        
        
        query2 = pd.get_dummies(pd.DataFrame(json1))
        query2 = query2.reindex(columns=model_colm, fill_value=0)
        
        #print("JJJJJJJJJJJJJJJJJJ")
        #print(query2)
        #print("SSSSSSSSSSSSSSSSSS")
        

        copy_query2 = query2.copy(deep=True)
        copy_query2['month'] = copy_query2['month'].astype(str) + '月'
        
        
        #print("JJJJJJJJJJJJJJJJJJ")
        #print(query2)
        #print("SSSSSSSSSSSSSSSSSS")
        
        
        #mname = query1['m'].apply(lambda x: calendar.month_abbr[x])
        mname = copy_query2['month']
        mname=pd.DataFrame(mname)
        dic2=[]
       
        for element2 in json1:
            month_no2=element2['month']
            rows2=actual.loc[actual['Month'] == month_no2]
            print("zxzxxxxxxxxxxxxx")
            print(rows2)
            for ik2 in rows2['s']:
                dic2.append(ik2)
               
                
        print(dic2)
        
        
        #mname=pd.DataFrame(mname)
        #print(mname)
         
        prediction2 = (regressor.predict(query2).astype('int64'))

        #print(prediction1)
        #print(prediction)
        prediction2=pd.DataFrame(prediction2, columns=["sales_pred"])
        print("predddddddddddddddddddddddd")
        print(prediction2)
        
        actual_data2=pd.DataFrame(dic2, columns=["sales_act"])
        print("actualllllllllllllllllllllll")
        print(actual_data2)
        
        
        #df_out = pd.merge(actual_data,prediction1,how = 'left',left_index = True, right_index = True)
        
        df_out2=pd.concat([actual_data2,prediction2],axis=1)
        print("concattttttttttttttttttt")
        print(df_out2)
        
        analytics2=[]
        for i2 in range(0,12):
            if i2<=7:
                print("ddddddddddddddddddddddd")
                
                x2=df_out2["sales_act"][i2]
                analytics2.append(x2)
                print("ddddddddddddddd")
                
         
            else:
                print("eeeeeeeeeeeeeee")
                x2=df_out2["sales_pred"][i2]
                print("sdddddddddddddddd")
                analytics2.append(x2)
                print("eeeeeeeeeeeeeeeeeee")
        
        print("daruuuuuuuuuuuuuuuuu")
        print(analytics2)
        
        df_act_pred2=pd.DataFrame(analytics2,columns=["sales"])
        
        
        #prediction2['sales'] = prediction2['sales'].apply(lambda x: x/1000)
        
        con=pd.concat([mname,df_act_pred2], axis=1)
        print("IIIIIIIIIII")
        print(con)
        print("NNNNNNNNNN")
        df=pd.DataFrame(con)
        #df.set_index('m')['0'].to_dict()
        df.set_index('month')['sales'].to_dict()
        
        count = df.shape[0]
        #print(count)
        months = df['month'].tolist() ########################
        sales = df['sales'].tolist() ##########################
        
        sales_npar = np.asarray(sales)
        expenses_npar = np.asarray(expenses)
        profit_npar = sales_npar - expenses_npar
        profit = profit_npar.tolist()
        
        #profit = sales - expenses
        print("NNNNNNNNNNNNN")
        print(sales)
        print(expenses)
        print(profit)
        print("KKKKKKKKKKKKK")
        
        
        
        
        ################################################################
        
        join = list(zip(months,sales,expenses,profit))  ###################
        join_json_string = json.dumps(join,ensure_ascii=False)
        

        
        '''
        print("PPPPPPPPPP")
        print(months)
        print(sales)
        print("KKKKKKKKKKKKK")
        
        list_of_dicts = []
        D={}
        
        #add a key and setup for a sub-dictionary
        
        for i in range(count):
            D[i] = {}
            D[i]['x']=months[i]
            D[i]['value']=sales[i]
            list_of_dicts.append(D[i])
    
        print("BBBBBBBBBBBBBB")
        print(list_of_dicts)
        print("LLLLLLLLLLLLLL")
        
        # convert into JSON:
        json_dict = json.dumps(list_of_dicts,ensure_ascii=False)
        
        # the result is a JSON string:
        print("LLLLLLLLLLOOO")
        print(json_dict)
        print("KKKKKKKKKKKKK")

        '''
        #t = "cheese"
        #return(t)
        #return(json_dict)
        return(join_json_string)
    
    except:
        
        return jsonify({'trace': traceback.format_exc()})

#monthwise column
@app.route('/monthexpcolumn', methods=['POST'])
@cross_origin(origin='localhost',headers=['Content- Type','Authorization'])

def predmonth_expcolumn():
    #print(request)
 
    try:
        print("HELLLO")
        json1= request.json
        length = len(json1)
        
        query1 = pd.get_dummies(pd.DataFrame(json1))
        query1 = query1.reindex(columns=model_colexp, fill_value=0)
        #print(query1)
        copy_query1 = query1.copy(deep=True)
        copy_query1['month'] = copy_query1['month'].astype(str) + '月'
        
        
        print("JJJJJJJJJJJJJJJJJJ")
        print(query1)
        print("SSSSSSSSSSSSSSSSSS")
        
        
        #mname = query1['m'].apply(lambda x: calendar.month_abbr[x])
        mname = copy_query1['month']
        #print(mname)
        
        
        mname=pd.DataFrame(mname)
        print(mname)
        
        dic=[]
       
        for element in json1:
            month_no=element['month']
            print(element)
            print(month_no)
            rows=act_expm.loc[act_expm['Month'] == month_no]
            for ik in rows['expenses']:
                print("ppppppppppp")
                print(ik)
                dic.append(ik)
               
        
        prediction1 = (regressor_exp.predict(query1).astype('int64'))

        print("JJJJJJJJJJ")
        print(prediction1)
        print("KkKKKKKK")
        #print(prediction)
        prediction1=pd.DataFrame(prediction1, columns=["expenses_pred"])
        
        
        actual_data=pd.DataFrame(dic, columns=["expenses_act"])
        print("actualllllllllllllllllllllll")
        print(actual_data)
        
        
        df_out=pd.concat([actual_data,prediction1],axis=1)
        print("concattttttttttttttttttt")
        print(df_out)
        
        analytics=[]
        for i in range(0,12):
            print("-------------------------------------------------------------")
            if i==0:
                print("eeeeeeeeeeeeeee0")
                x=df_out["expenses_pred"][i]
                print("sddddddddddddddd0")
                analytics.append(x)
                print("eeeeeeeeeeeeeeeeee0")
                
            elif i in range(1,6):
                print("ddddddddddddddddddddddd16")
                
                x=df_out["expenses_act"][i-1]
                analytics.append(x)
                print("ddddddddddddddd16")
                
            elif i==6:
                print("eeeeeeeeeeeeeee6")
                x=df_out["expenses_pred"][i]
                print("sdddddddddddddddd6")
                analytics.append(x)
                print("eeeeeeeeeeeeeeeeeee6")
                
            elif i==7:
                print("ddddddddddddddddddddddd7")
                
                x=df_out["expenses_act"][i-2]
                analytics.append(x)
                print("ddddddddddddddd7")
                
            
            
            else:
                print("eeeeeeeeeeeeeee")
                x=df_out["expenses_pred"][i]
                print("sdddddddddddddddd")
                analytics.append(x)
                print("eeeeeeeeeeeeeeeeeee")
        
        print("daruuuuuuuuuuuuuuuuu")
        print(analytics)
        
        df_act_pred=pd.DataFrame(analytics,columns=["expenses"])
        
        #prediction1['expenses'] = prediction1['expenses'].apply(lambda x: x/1000)
        
        con=pd.concat([mname, df_act_pred], axis=1)
        print("IIIIIIIIIII")
        print(con)
        print("NNNNNNNNNN")
        df=pd.DataFrame(con)
        #df.set_index('m')['0'].to_dict()
        df.set_index('month')['expenses'].to_dict()
        
        count = df.shape[0]
        #print(count)
        
        #print("LLLLLLLLLLOOO")
        #print(df)
        #print("KKKKKKKKKKKKK")
        
        
        #months = df['month'].tolist()
        expenses = df['expenses'].tolist()
        
        
        ###################################################################
        
        
        
        query2 = pd.get_dummies(pd.DataFrame(json1))
        query2 = query2.reindex(columns=model_colm, fill_value=0)
        
        #print("JJJJJJJJJJJJJJJJJJ")
        #print(query2)
        #print("SSSSSSSSSSSSSSSSSS")
        

        copy_query2 = query2.copy(deep=True)
        copy_query2['month'] = copy_query2['month'].astype(str) + '月'
        
        
        #print("JJJJJJJJJJJJJJJJJJ")
        #print(query2)
        #print("SSSSSSSSSSSSSSSSSS")
        
        
        #mname = query1['m'].apply(lambda x: calendar.month_abbr[x])
        mname = copy_query2['month']
        mname=pd.DataFrame(mname)
        dic2=[]
       
        for element2 in json1:
            month_no2=element2['month']
            rows2=actual.loc[actual['Month'] == month_no2]
            print("zxzxxxxxxxxxxxxx")
            print(rows2)
            for ik2 in rows2['s']:
                dic2.append(ik2)
               
                
        print(dic2)
        
        
        #mname=pd.DataFrame(mname)
        #print(mname)
         
        prediction2 = (regressor.predict(query2).astype('int64'))

        #print(prediction1)
        #print(prediction)
        prediction2=pd.DataFrame(prediction2, columns=["sales_pred"])
        print("predddddddddddddddddddddddd")
        print(prediction2)
        
        actual_data2=pd.DataFrame(dic2, columns=["sales_act"])
        print("actualllllllllllllllllllllll")
        print(actual_data2)
        
        
        #df_out = pd.merge(actual_data,prediction1,how = 'left',left_index = True, right_index = True)
        
        df_out2=pd.concat([actual_data2,prediction2],axis=1)
        print("concattttttttttttttttttt")
        print(df_out2)
        
        analytics2=[]
        for i2 in range(0,12):
            if i2<=7:
                print("ddddddddddddddddddddddd")
                
                x2=df_out2["sales_act"][i2]
                analytics2.append(x2)
                print("ddddddddddddddd")
                
         
            else:
                print("eeeeeeeeeeeeeee")
                x2=df_out2["sales_pred"][i2]
                print("sdddddddddddddddd")
                analytics2.append(x2)
                print("eeeeeeeeeeeeeeeeeee")
        
        print("daruuuuuuuuuuuuuuuuu")
        print(analytics2)
        
        df_act_pred2=pd.DataFrame(analytics2,columns=["sales"])
        
        
        #prediction2['sales'] = prediction2['sales'].apply(lambda x: x/1000)
        
        con=pd.concat([mname,df_act_pred2], axis=1)
        print("IIIIIIIIIII")
        print(con)
        print("NNNNNNNNNN")
        df=pd.DataFrame(con)
        #df.set_index('m')['0'].to_dict()
        df.set_index('month')['sales'].to_dict()
        
        count = df.shape[0]
        #print(count)
        months = df['month'].tolist() ########################
        sales = df['sales'].tolist() ##########################
        
        sales_npar = np.asarray(sales)
        expenses_npar = np.asarray(expenses)
        profit_npar = sales_npar - expenses_npar
        profit = profit_npar.tolist()
        
        #list_of_dicts = []
        #D = {}
        #D['x']="Saving"
        #D['value']=profit[0]
        #list_of_dicts.append(D)
        
        #E = {}
        #E['x']="Expenses"
        #E['value']=expenses[0]
        #list_of_dicts.append(E)
        
        list_output = [['売上', sales[0]],['費用', expenses[0]]]

        print("NNNNNNNNNNNNN")
        #print(sales[0])
        print(expenses[0])
        print(profit[0])
        print("KKKKKKKKKKKKK")
        
        
        ################################################################
        
        out_json_string = json.dumps(list_output,ensure_ascii=False)
        #out_json_string = json.dumps(list_of_dicts,ensure_ascii=Fa1lse)
        
        
        #t = "cheese"
        #return(t)
        #return(json_dict)
        return(out_json_string)
    
    except:
        
        return jsonify({'trace': traceback.format_exc()})

    
    
    
"""
if __name__ == '__main__':
    app.run()     
    
"""  
    
    
    
    
    
    
        
#daywise
@app.route('/day', methods=['POST'])
@cross_origin(origin='localhost',headers=['Content- Type','Authorization'])

#for gcp cloud
#cors = CORS(app, resources={r"/": {"origins": "https://jts-board.appspot.com/"}})

#@app.route('/', methods=['POST'])
#@cross_origin(origin='*',headers=['Content- Type','Authorization'])

def predict():
    #print(request)
    if clf:
        try:
            print("LLLLLLLLLLLLLLL")
            jsond = request.json
            queryd = pd.get_dummies(pd.DataFrame(jsond))
            queryd = queryd.reindex(columns=model_cold, fill_value=0)
            
            predictiond = list(clf.predict(queryd).astype("int64"))
            print(predictiond)
            


            print(queryd)
            copy_queryd = queryd.copy(deep=True)
            copy_queryd.columns = ['day','month','year']

            date_time = pd.to_datetime(copy_queryd[['day', 'month', 'year']])
            #date_time.columns = ['timestamp']
 
            date_time_df=pd.DataFrame(date_time, columns=["timestamp"])
            date_time_df['day'] = date_time_df['timestamp'].dt.dayofweek
            dnum=pd.DataFrame(date_time_df['day'])
            dname = dnum['day'].apply(lambda x: calendar.day_abbr[x])
            #df['weekday'] = df['Timestamp'].apply(lambda x: x.weekday())
            
            print("PPPPPPPPPPPPPP")
            #print(copy_queryd)
            #print(date_time_df)
            #print(dname)
            print("JJJJJJJJJJJJJJ")

            #print(queryd['m'])
            #mname = queryd['d'].apply(lambda x: calendar.day_abbr[x])
            #print(mname)
            #print(queryxid1)

            #mname=pd.DataFrame(queryd['d'])
            #mname=pd.DataFrame(mname)
        
            predictiond=pd.DataFrame(predictiond, columns=["sales"])
            predictiond['sales'] = predictiond['sales'].apply(lambda x: x/1000)

            print(predictiond)
            con=pd.concat([dname,predictiond], axis=1)##################
            print(con)
            df=pd.DataFrame(con)###############


            df.set_index('day')['sales'].to_dict()
            print(df)

            count = df.shape[0]
            #print(count)
   
            #print("LLLLLLLLLLOOO") 
            #print(df)
            #print("KKKKKKKKKKKKK")
 

            days = df['day'].tolist()
            sales = df['sales'].tolist()

            #print("LLLLLLLLLLOOO")
            #print(months[1])
            #print(sales[1])
            #print("KKKKKKKKKKKKK")

            list_of_dicts = []
            D={}

            #add a key and setup for a sub-dictionary

            for i in range(count):
                D[i] = {}
                D[i]['x']=days[i]
                D[i]['value']=sales[i]
                list_of_dicts.append(D[i])


            #print(list_of_dicts)

            # convert into JSON:
            json_dict = json.dumps(list_of_dicts)

            # the result is a JSON string:
            print("LLLLLLLLLLOOO")
            print(json_dict)
            print("KKKKKKKKKKKKK")

    









            a = "cheese"        
            return json_dict    
            #return jsonify({'prediction': str(predictiond)})

        except:

            return jsonify({'trace': traceback.format_exc()})
    else:
        print ('Train the model first')
        return ('No model here to use')



    #weekwise
@app.route('/week', methods=['POST'])
@cross_origin(origin='localhost',headers=['Content- Type','Authorization'])


def predweek():
    #print(request)
    if clfweek:
        try:
            print("WWWWWWWWWWWWWWWWWWWWWWWWW")
            jsonw= request.json
            queryw = pd.get_dummies(pd.DataFrame(jsonw))
            queryw = queryw.reindex(columns=model_colw, fill_value=0)


            print(queryw)
            print("OOOOOOOOOOOOOOOOO")
            #print(query1['m'])
            #mname = queryw['w'].apply(lambda x: calendar.day_abbr[x])########
            #print(mname)
            #print(query1)
            mname=pd.DataFrame(queryw['week'])#########
            print(mname)
            print("WWWWWWWWWWWWWWWWWWWWWw")

            
            predictionw = (clfweek.predict(queryw).astype("int64"))
            
            #print("PPPPPPPPPPPPPP")
            #print(predictionw)
            #print("JJJJJJJJJJJJJJ")

            #print(prediction1)
            #print(prediction)
            predictionw=pd.DataFrame(predictionw, columns=["sales"])##########
            print(predictionw)
            predictionw['sales'] = predictionw['sales'].apply(lambda x: x/1000)

            con=pd.concat([mname,predictionw], axis=1)##################
            print(con)
            df=pd.DataFrame(con)################


            df.set_index('week')['sales'].to_dict()
            #print(df)



            count = df.shape[0]
            #print(count)

            #print("LLLLLLLLLLOOO") 
            #print(df)
            #print("KKKKKKKKKKKKK")


            weeks = df['week'].tolist()
            sales = df['sales'].tolist()

            #print("LLLLLLLLLLOOO")
            #print(months[1])
            #print(sales[1])
            #print("KKKKKKKKKKKKK")

            list_of_dicts = []
            D={}

            #add a key and setup for a sub-dictionary

            for i in range(count):
                D[i] = {}
                D[i]['x']=weeks[i]
                D[i]['value']=sales[i]
                list_of_dicts.append(D[i])


            #print(list_of_dicts)

            # convert into JSON:
            json_dict = json.dumps(list_of_dicts)

            # the result is a JSON string:
            print("LLLLLLLLLLOOO")
            print(json_dict)
            print("KKKKKKKKKKKKK")

            return json_dict
            #return jsonify({'prediction weekwise': str(predictionw).splitlines()})

        except:

            return jsonify({'trace': traceback.format_exc()})
    else:
        print ('Train the model first')
        return ('No model here to use')
        
    
  #customer visits  
#daywise
@app.route('/vdays', methods=['POST'])
@cross_origin(origin='localhost',headers=['Content- Type','Authorization'])

#for gcp cloud
#cors = CORS(app, resources={r"/": {"origins": "https://jts-board.appspot.com/"}})

#@app.route('/', methods=['POST'])
#@cross_origin(origin='*',headers=['Content- Type','Authorization'])

def predictvdays():
    #print(request)
    if clfvd:
        try:
           # print("LLLLLLLLLLLLLLL")
            jsond = request.json
            queryd = pd.get_dummies(pd.DataFrame(jsond))
            queryd = queryd.reindex(columns=model_colvd, fill_value=0)
            
            predictiond = list(clfvd.predict(queryd))
            #print(predictiond)
            


            #print(queryd)
            copy_queryd = queryd.copy(deep=True)
            copy_queryd.columns = ['day','month','year']

            date_time = pd.to_datetime(copy_queryd[['day', 'month', 'year']])
            #date_time.columns = ['timestamp']
 
            date_time_df=pd.DataFrame(date_time, columns=["timestamp"])
            date_time_df['day'] = date_time_df['timestamp'].dt.dayofweek
            dnum=pd.DataFrame(date_time_df['day'])
            dname = dnum['day'].apply(lambda x: calendar.day_abbr[x])
            #df['weekday'] = df['Timestamp'].apply(lambda x: x.weekday())
            
            #print("PPPPPPPPPPPPPP")
            #print(copy_queryd)
            #print(date_time_df)
            #print(dname)
            #print("JJJJJJJJJJJJJJ")

            #print(queryd['m'])
            #mname = queryd['d'].apply(lambda x: calendar.day_abbr[x])
            #print(mname)
            #print(queryxid1)

            #mname=pd.DataFrame(queryd['d'])
            #mname=pd.DataFrame(mname)
        
            predictiond=pd.DataFrame(predictiond, columns=["visits"])
            #predictiond[''] = predictiond['s'].apply(lambda x: x/1000)

            #print(predictiond)
            con=pd.concat([dname,predictiond], axis=1)##################
            #print(con)
            df=pd.DataFrame(con)###############


            df.set_index('day')['visits'].to_dict()
            #print(df)

            count = df.shape[0]
            #print(count)
   
            #print("LLLLLLLLLLOOO") 
            #print(df)
            #print("KKKKKKKKKKKKK")
 

            days = df['day'].tolist()
            sales = df['visits'].tolist()

            #print("LLLLLLLLLLOOO")
            #print(months[1])
            #print(sales[1])
            #print("KKKKKKKKKKKKK")

            list_of_dicts = []
            D={}

            #add a key and setup for a sub-dictionary

            for i in range(count):
                D[i] = {}
                D[i]['x']=days[i]
                D[i]['value']=sales[i]
                list_of_dicts.append(D[i])


            #print(list_of_dicts)

            # convert into JSON:
            json_dict = json.dumps(list_of_dicts)

            # the result is a JSON string:
            """
            print("LLLLLLLLLLOOO")
            print(json_dict)
            print("KKKKKKKKKKKKK")
            """

    









            a = "cheese"        
            return json_dict    
            #return jsonify({'prediction': str(predictiond)})

        except:

            return jsonify({'trace': traceback.format_exc()})
    else:
        print ('Train the model first')
        return ('No model here to use')
    
    
    
#monthwise
        



    #weekwise
@app.route('/vweeks', methods=['POST'])
@cross_origin(origin='localhost',headers=['Content- Type','Authorization'])


def predvweeks():
    #print(request)
    if clfvw:
        try:
            print("WWWWWWWWWWWWWWWWWWWWWWWWW")
            jsonw= request.json
            queryw = pd.get_dummies(pd.DataFrame(jsonw))
            queryw = queryw.reindex(columns=model_colvw, fill_value=0)


            print(queryw)
            print("OOOOOOOOOOOOOOOOO")
            #print(query1['m'])
            #mname = queryw['w'].apply(lambda x: calendar.day_abbr[x])########
            #print(mname)
            #print(query1)
            mname=pd.DataFrame(queryw['week'])#########
            print(mname)
            print("WWWWWWWWWWWWWWWWWWWWWw")

            
            predictionw = (clfvw.predict(queryw).astype("int64"))
            
            #print("PPPPPPPPPPPPPP")
            #print(predictionw)
            #print("JJJJJJJJJJJJJJ")

            #print(prediction1)
            #print(prediction)
            predictionw=pd.DataFrame(predictionw, columns=["visits"])##########
            print(predictionw)
            predictionw['visits'] = predictionw['visits'].apply(lambda x: x/1000)

            con=pd.concat([mname,predictionw], axis=1)##################
            print(con)
            df=pd.DataFrame(con)################


            df.set_index('week')['visits'].to_dict()
            #print(df)



            count = df.shape[0]
            #print(count)

            #print("LLLLLLLLLLOOO") 
            #print(df)
            #print("KKKKKKKKKKKKK")


            weeks = df['week'].tolist()
            sales = df['visits'].tolist()

            #print("LLLLLLLLLLOOO")
            #print(months[1])
            #print(sales[1])
            #print("KKKKKKKKKKKKK")

            list_of_dicts = []
            D={}

            #add a key and setup for a sub-dictionary

            for i in range(count):
                D[i] = {}
                D[i]['x']=weeks[i]
                D[i]['value']=sales[i]
                list_of_dicts.append(D[i])


            #print(list_of_dicts)

            # convert into JSON:
            json_dict = json.dumps(list_of_dicts)

            # the result is a JSON string:
            print("LLLLLLLLLLOOO")
            print(json_dict)
            print("KKKKKKKKKKKKK")

            return json_dict
            #return jsonify({'prediction weekwise': str(predictionw).splitlines()})

        except:

            return jsonify({'trace': traceback.format_exc()})
    else:
        print ('Train the model first')
        return ('No model here to use')


#monthwise expenses



#monthwise donut
@app.route('/monthexpdonut', methods=['POST'])
@cross_origin(origin='localhost',headers=['Content- Type','Authorization'])

def predmonth_expdonut():
    #print(request)

    try:
        print("HELLLO")
        json1= request.json
        length = len(json1)
        
        query1 = pd.get_dummies(pd.DataFrame(json1))
        query1 = query1.reindex(columns=model_colexp, fill_value=0)
        #print(query1)
        copy_query1 = query1.copy(deep=True)
        copy_query1['month'] = copy_query1['month'].astype(str) + '月'
        
        
        print("JJJJJJJJJJJJJJJJJJ")
        print(query1)
        print("SSSSSSSSSSSSSSSSSS")
        
        
        #mname = query1['m'].apply(lambda x: calendar.month_abbr[x])
        mname = copy_query1['month']
        #print(mname)
        
        
        mname=pd.DataFrame(mname)
        print(mname)
        
        dic=[]
       
        for element in json1:
            month_no=element['month']
            print(element)
            print(month_no)
            rows=act_expm.loc[act_expm['Month'] == month_no]
            for ik in rows['expenses']:
                print("ppppppppppp")
                print(ik)
                dic.append(ik)
               
        
        prediction1 = (regressor_exp.predict(query1).astype('int64'))

        print("JJJJJJJJJJ")
        print(prediction1)
        print("KkKKKKKK")
        #print(prediction)
        prediction1=pd.DataFrame(prediction1, columns=["expenses_pred"])
        
        
        actual_data=pd.DataFrame(dic, columns=["expenses_act"])
        print("actualllllllllllllllllllllll")
        print(actual_data)
        
        
        df_out=pd.concat([actual_data,prediction1],axis=1)
        print("concattttttttttttttttttt")
        print(df_out)
        
        analytics=[]
        for i in range(0,12):
            print("-------------------------------------------------------------")
            if i==0:
                print("eeeeeeeeeeeeeee0")
                x=df_out["expenses_pred"][i]
                print("sddddddddddddddd0")
                analytics.append(x)
                print("eeeeeeeeeeeeeeeeee0")
                
            elif i in range(1,6):
                print("ddddddddddddddddddddddd16")
                
                x=df_out["expenses_act"][i-1]
                analytics.append(x)
                print("ddddddddddddddd16")
                
            elif i==6:
                print("eeeeeeeeeeeeeee6")
                x=df_out["expenses_pred"][i]
                print("sdddddddddddddddd6")
                analytics.append(x)
                print("eeeeeeeeeeeeeeeeeee6")
                
            elif i==7:
                print("ddddddddddddddddddddddd7")
                
                x=df_out["expenses_act"][i-2]
                analytics.append(x)
                print("ddddddddddddddd7")
                
            
            
            else:
                print("eeeeeeeeeeeeeee")
                x=df_out["expenses_pred"][i]
                print("sdddddddddddddddd")
                analytics.append(x)
                print("eeeeeeeeeeeeeeeeeee")
        
        print("daruuuuuuuuuuuuuuuuu")
        print(analytics)
        
        df_act_pred=pd.DataFrame(analytics,columns=["expenses"])
        
        #prediction1['expenses'] = prediction1['expenses'].apply(lambda x: x/1000)
        
        con=pd.concat([mname, df_act_pred], axis=1)
        print("IIIIIIIIIII")
        print(con)
        print("NNNNNNNNNN")
        df=pd.DataFrame(con)
        #df.set_index('m')['0'].to_dict()
        df.set_index('month')['expenses'].to_dict()
        
        count = df.shape[0]
        #print(count)
        
        #print("LLLLLLLLLLOOO")
        #print(df)
        #print("KKKKKKKKKKKKK")
        
        
        #months = df['month'].tolist()
        expenses = df['expenses'].tolist()
        
        
        ###################################################################
        
        
        
        query2 = pd.get_dummies(pd.DataFrame(json1))
        query2 = query2.reindex(columns=model_colm, fill_value=0)
        
        #print("JJJJJJJJJJJJJJJJJJ")
        #print(query2)
        #print("SSSSSSSSSSSSSSSSSS")
        

        copy_query2 = query2.copy(deep=True)
        copy_query2['month'] = copy_query2['month'].astype(str) + '月'
        
        
        #print("JJJJJJJJJJJJJJJJJJ")
        #print(query2)
        #print("SSSSSSSSSSSSSSSSSS")
        
        
        #mname = query1['m'].apply(lambda x: calendar.month_abbr[x])
        mname = copy_query2['month']
        mname=pd.DataFrame(mname)
        dic2=[]
       
        for element2 in json1:
            month_no2=element2['month']
            rows2=actual.loc[actual['Month'] == month_no2]
            print("zxzxxxxxxxxxxxxx")
            print(rows2)
            for ik2 in rows2['s']:
                dic2.append(ik2)
               
                
        print(dic2)
        
        
        #mname=pd.DataFrame(mname)
        #print(mname)
         
        prediction2 = (regressor.predict(query2).astype('int64'))

        #print(prediction1)
        #print(prediction)
        prediction2=pd.DataFrame(prediction2, columns=["sales_pred"])
        print("predddddddddddddddddddddddd")
        print(prediction2)
        
        actual_data2=pd.DataFrame(dic2, columns=["sales_act"])
        print("actualllllllllllllllllllllll")
        print(actual_data2)
        
        
        #df_out = pd.merge(actual_data,prediction1,how = 'left',left_index = True, right_index = True)
        
        df_out2=pd.concat([actual_data2,prediction2],axis=1)
        print("concattttttttttttttttttt")
        print(df_out2)
        
        analytics2=[]
        for i2 in range(0,12):
            if i2<=7:
                print("ddddddddddddddddddddddd")
                
                x2=df_out2["sales_act"][i2]
                analytics2.append(x2)
                print("ddddddddddddddd")
                
         
            else:
                print("eeeeeeeeeeeeeee")
                x2=df_out2["sales_pred"][i2]
                print("sdddddddddddddddd")
                analytics2.append(x2)
                print("eeeeeeeeeeeeeeeeeee")
        
        print("daruuuuuuuuuuuuuuuuu")
        print(analytics2)
        
        df_act_pred2=pd.DataFrame(analytics2,columns=["sales"])
        
        
        #prediction2['sales'] = prediction2['sales'].apply(lambda x: x/1000)
        
        con=pd.concat([mname,df_act_pred2], axis=1)
        print("IIIIIIIIIII")
        print(con)
        print("NNNNNNNNNN")
        df=pd.DataFrame(con)
        #df.set_index('m')['0'].to_dict()
        df.set_index('month')['sales'].to_dict()
        
        count = df.shape[0]
        #print(count)
        months = df['month'].tolist() ########################
        sales = df['sales'].tolist() ##########################
        
        sales_npar = np.asarray(sales)
        expenses_npar = np.asarray(expenses)
        profit_npar = sales_npar - expenses_npar
        profit = profit_npar.tolist()
        
        list_of_dicts = []
        D = {}
        D['x']="利益"
        D['value']=profit[0]
        list_of_dicts.append(D)
        
        E = {}
        E['x']="費用"
        E['value']=expenses[0]
        list_of_dicts.append(E)
        
        #list_of_dicts.append(D[i])
        
        #list_output = [['Expenses', expenses[0]],['Saving', profit[0]]]
        #profit = sales - expenses
        print("NNNNNNNNNNNNN")
        #print(sales[0])
        print(expenses[0])
        print(profit[0])
        print(list_of_dicts)
        print("KKKKKKKKKKKKK")
        
        
        ################################################################
        
        #out_json_string = json.dumps(list_output,ensure_ascii=False)
        out_json_string = json.dumps(list_of_dicts,ensure_ascii=False)
        
        
        #t = "cheese"
        #return(t)
        #return(json_dict)
        return(out_json_string)
    
    except:
        
        return jsonify({'trace': traceback.format_exc()})
   





if __name__ == '__main__':
    app.run()     


