# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import os
import numpy as np
import pandas as pd
from time import time
import csv

os.chdir('C:/Users/Pc-stock2/Desktop/carte memoire')


#training set is too big (6GB) so create a HDF5 data format
data_type={'id':str, 'click':int, 'hour':str, 'C1':int, 
'banner_pos':int, 'site_id':str, 'site_domain':str, 'site_category':str,
'app_id':str, 'app_domain':str, 'app_category':str, 'device_id':str,'device_ip':str,
'device_model':str, 'device_type':int,'device_conn_type':int, 'C14':int, 'C15':int,
'C16':int, 'C17':int, 'C18':int, 'C19': int, 'C20':int, 'C21':int}

min_presetsize={'id':20, 'click':2, 'hour':9, 'C1':5, 
'banner_pos':2, 'site_id':9, 'site_domain':9, 'site_category':9,
'app_id':9, 'app_domain':9, 'app_category':9, 'device_id':9,'device_ip':9,
'device_model':9, 'device_type':2,'device_conn_type':2, 'C14':6, 'C15':4,
'C16':3, 'C17':5, 'C18':2, 'C19': 3, 'C20':3, 'C21':3}

store=pd.HDFStore('train.h5')
for df in pd.read_csv('train.csv',dtype=data_type,chunksize=500000):
    
    for i in range(len(df)):
    if feature7[i] not in site_id:
        site_id[feature7[i]]=site_id_counter
        site_id_counter+=1
    if feature8[i] not in site_domain:
        site_domain[feature8[i]]=site_domain_counter
        site_domain_counter+=1
    if feature9[i] not in site_category:
        site_category[feature9[i]]=site_category_counter
        site_category_counter+=1
    if feature10[i] not in app_id:
        app_id[feature10[i]]=app_id_counter
        app_id_counter+=1
    if feature11[i] not in app_domain:
        app_domain[feature11[i]]=app_domain_counter
        app_domain_counter+=1
    if feature12[i] not in app_category:
        app_category[feature12[i]]=app_category_counter
        app_category_counter+=1
    if feature13[i] not in device_id:
        device_id[feature13[i]]=device_id_counter
        device_id_counter+=1
    if feature14[i] not in device_ip:
        device_ip[feature14[i]]=device_ip_counter
        device_ip_counter+=1
    if feature15[i] not in device_model:
        device_model[feature15[i]]=device_model_counter
        device_model_counter+=1    
    
    
    
    
    
    
    
    
    store.append('train',df,data_columns=True,min_itemsize = min_presetsize)

store.close()


store=pd.HDFStore('train.h5')



t0 = time()
for row in store.select('train',chunksize=1,start=0,stop=100000):
    row['C1'].values
t1 = time()
t1-t0

C



test=pd.read_csv('train.csv', dtype=data_type, nrows=100)

train=test[['C1','C14','C15','C16']]
test=test['click']

train.to_hdf('train2.h5','train')
train=pd.HDFStore('train2.h5')

test.to_hdf('test2.h5','test')
test=pd.HDFStore('test2.h5')


from sklearn.linear_model import SGDClassifier
clf = SGDClassifier(loss="log").fit(train, test)

X = [[0., 0.], [1., 1.]]
y = [0, 1]



nb_rows=100000
#output
output=pd.read_csv('train.csv',usecols=[1],squeeze=1, nrows=nb_rows)
#features
##feature 1 will be divided into features 1-4
feature1=pd.read_csv('train.csv',usecols=[2],squeeze=1, nrows=nb_rows)
##features 5-25 
feature5=pd.read_csv('train.csv',usecols=[3],squeeze=1, nrows=nb_rows)
feature6=pd.read_csv('train.csv',usecols=[4],squeeze=1, nrows=nb_rows)
feature7=pd.read_csv('train.csv',usecols=[5],squeeze=1, nrows=nb_rows)
feature8=pd.read_csv('train.csv',usecols=[6],squeeze=1, nrows=nb_rows)
feature9=pd.read_csv('train.csv',usecols=[7],squeeze=1, nrows=nb_rows)
feature10=pd.read_csv('train.csv',usecols=[8],squeeze=1, nrows=nb_rows)
feature11=pd.read_csv('train.csv',usecols=[9],squeeze=1, nrows=nb_rows)
feature12=pd.read_csv('train.csv',usecols=[10],squeeze=1, nrows=nb_rows)
feature13=pd.read_csv('train.csv',usecols=[11],squeeze=1, nrows=nb_rows)
feature14=pd.read_csv('train.csv',usecols=[12],squeeze=1, nrows=nb_rows)
feature15=pd.read_csv('train.csv',usecols=[13],squeeze=1, nrows=nb_rows)
feature16=pd.read_csv('train.csv',usecols=[14],squeeze=1, nrows=nb_rows)
feature17=pd.read_csv('train.csv',usecols=[15],squeeze=1, nrows=nb_rows)
feature18=pd.read_csv('train.csv',usecols=[16],squeeze=1, nrows=nb_rows)
feature19=pd.read_csv('train.csv',usecols=[17],squeeze=1, nrows=nb_rows)
feature20=pd.read_csv('train.csv',usecols=[18],squeeze=1, nrows=nb_rows)
feature21=pd.read_csv('train.csv',usecols=[19],squeeze=1, nrows=nb_rows)
feature22=pd.read_csv('train.csv',usecols=[20],squeeze=1, nrows=nb_rows)
feature23=pd.read_csv('train.csv',usecols=[21],squeeze=1, nrows=nb_rows)
feature24=pd.read_csv('train.csv',usecols=[22],squeeze=1, nrows=nb_rows)
feature25=pd.read_csv('train.csv',usecols=[23],squeeze=1, nrows=nb_rows)




train=pd.read_csv('train.csv', nrows=100)
train=pd.read_csv('train.csv', header=0,nrows=10)

for sample in train:
    print sample
    



feature1_4=train[]


a=train['hour'].str.split(a)



#initialize the dictionaries and counters
site_id={}
site_domain={}
site_category={}
app_id={}
app_domain={} 
app_category={}
device_id={}
device_ip={}
device_model={}

site_id_counter=0
site_domain_counter=0
site_category_counter=0
app_id_counter=0
app_domain_counter=0
app_category_counter=0
device_id_counter=0
device_ip_counter=0
device_model_counter=0


#fill the dictionaries
for i in range(nb_rows):
    if feature7[i] not in site_id:
        site_id[feature7[i]]=site_id_counter
        site_id_counter+=1
    if feature8[i] not in site_domain:
        site_domain[feature8[i]]=site_domain_counter
        site_domain_counter+=1
    if feature9[i] not in site_category:
        site_category[feature9[i]]=site_category_counter
        site_category_counter+=1
    if feature10[i] not in app_id:
        app_id[feature10[i]]=app_id_counter
        app_id_counter+=1
    if feature11[i] not in app_domain:
        app_domain[feature11[i]]=app_domain_counter
        app_domain_counter+=1
    if feature12[i] not in app_category:
        app_category[feature12[i]]=app_category_counter
        app_category_counter+=1
    if feature13[i] not in device_id:
        device_id[feature13[i]]=device_id_counter
        device_id_counter+=1
    if feature14[i] not in device_ip:
        device_ip[feature14[i]]=device_ip_counter
        device_ip_counter+=1
    if feature15[i] not in device_model:
        device_model[feature15[i]]=device_model_counter
        device_model_counter+=1





for df in pd.read_csv('train.csv',dtype=data_type,chunksize=1000):
    
    df=test
    
    #create individual columns for year, month, day and time
    df['year']=df.hour.str[0:2].astype(int)
    df['month']=df.hour.str[2:4].astype(int)
    df['day']=df.hour.str[4:6].astype(int)
    df['time']=df.hour.str[6:8].astype(int)
    
    #replace category values
    df.replace(
                {
                'site_id':site_id,
                'site_domain':site_domain,
                'site_category':site_category,
                'app_id':app_id,
                'app_domain':app_domain,
                'app_category':app_category,
                'device_id':device_id,
                'device_ip':device_ip,
                'device_model':device_model
                },
            inplace=True
            )
    
    #output
    y=df.ix[:,1].values
    x=df.ix[:,2:len(df.columns)].values
    
    clf = SGDClassifier(loss="log").fit(x, y)






#set initial values
theta=np.array([0.01]*26)
alpha=0.001


from time import time



cost=[0]*20
cost_temp=[0]*1000
  
t0 = time()
for i in range(nb_rows):
    
    #1) create the feature vector
    x = np.array([
        #start with x0
        1,
        #divide YYMMDDHH into four seperate features
        int(str(feature1[i])[0:2]),
        int(str(feature1[i])[2:4]),
        int(str(feature1[i])[4:6]),
        int(str(feature1[i])[6:8]),
        #add the next two features
        feature5[i],
        feature6[i],
        #add values associated with site_id, site_domain, ...
        site_id[feature7[i]],
        site_domain[feature8[i]],
        site_category[feature9[i]],
        app_id[feature10[i]],
        app_domain[feature11[i]],
        app_category[feature12[i]],
        device_id[feature13[i]],
        device_ip[feature14[i]],
        device_model[feature15[i]],
        #add the final values
        feature16[i],
        feature17[i],
        feature18[i],
        feature19[i],
        feature20[i],
        feature21[i],
        feature22[i],
        feature23[i],
        feature24[i],
        feature25[i]])
    
    ##2) make the prediction
    pred_y = 1/(1+np.exp(-np.dot(theta,x)))
    
    ##
    cost_temp[i%1000]=-(output[i]*np.log(pred_y) + (1-output[i])*(1-np.log(pred_y)))
    if (i%5000==0): cost[i/5000]=np.mean(cost_temp)

    ##3) update the theta vector
    theta += alpha*(pred_y-output[i])*x
time()-t0 
    

    
    
    









