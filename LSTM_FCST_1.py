# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 19:43:09 2019

@author: mainak.kundu
"""

import os
import time
import math
import pandas as pd
import sys
import keras as keras
from keras.models import Sequential
from keras.layers import Embedding
from keras.layers import Dense
from keras.layers import LSTM
import ast 
from multiprocessing import Pool
from contextlib import closing
import numpy as np 
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

p1 = str.replace(sys.argv[1],"#","//") ## input 
p2 = str.replace(sys.argv[2],"#","//") ## output
#p9 = str.replace(sys.argv[3],'#',"//") ## hyperdata 
p3 = sys.argv[3].split(",")
p4 = sys.argv[4].split(",")
p5 = sys.argv[5].split(",")
p6 = sys.argv[6]
p7 = sys.argv[7]
p8 = sys.argv[8]
print(p3)
print(p4)
#exit()
print(p1)
print(p2)
print(p3)
print(p4)
print(p5)



def normalizeData(data,cols):
    
    for c in cols:
        data[c] = SCALER.fit_transform(data[c].values.reshape(-1,1))
    return data

def findLags(df,var):
    data = list(df[var])
    maxi,curr = 0,0
    for d in data:
        if d==1:
            curr +=1
        else:
            if curr > maxi:
                maxi = curr
            curr = 0
   
    return min(2,maxi)


def createLag(dfh,dfl, lagVar, normalize = False):
    newVar = []
    lags =  findLags(dfh,lagVar)
   
    for i in range(lags):
        dfh[lagVar + '_LAG_%s'%i] = dfh[lagVar].shift(i+1)
        dfh[lagVar + '_LAG_%s'%i].fillna(0, inplace=True)
        dfl[lagVar + '_LAG_%s'%i] = dfl[lagVar].shift(i+1)
        dfl[lagVar + '_LAG_%s'%i].fillna(0, inplace=True)
        newVar.append(lagVar + '_LAG_%s'%i)
        if normalize:
            df[lagVar + '_LAG_%s'%i] = SCALER.fit_transform(df[lagVar + '_LAG_%s'%i] .values.reshape(-1,1))
    
    return (dfh,dfl,newVar)




'''
LITRETURE DEFAULT
'''
def LitretureParams():
    epoch = 10
    DROP_OUT = 0.15
    RCRNT_DROP_OUT =  0.15
    IN_NEURON = 25
    L1_NEURON = 15
    return epoch, DROP_OUT,RCRNT_DROP_OUT,IN_NEURON,L1_NEURON
    


'''
SYSTEM DEFAULT 
'''

def SystemDefaults(terr):
    epoch = terr['epoch'].value_counts().idxmax()
    DROP_OUT =  terr['DROP_OUT'].value_counts().idxmax()
    RCRNT_DROP_OUT = terr['RCRNT_DROP_OUT'].value_counts().idxmax()
    IN_NEURON = terr['IN_NEURON'].value_counts().idxmax()
    L1_NEURON = terr['L1_NEURON'].value_counts().idxmax()
    return epoch, DROP_OUT,RCRNT_DROP_OUT,IN_NEURON,L1_NEURON


'''
RESOLVE OUTPUT PATH IN  R ---PENDING ACTION 
'''



if __name__== '__main__':
    print('---Main Program starts----')
    usr = p6
    cncpt = p7
    dept = p8
    logs = open(usr+'_'+cncpt+'_'+dept+'_'+'LSTM_PLL.txt','w')
    sys.stdout = logs
    print('--Log file initiated----')
    #INPUT_PATH          = p1
    HISTDATA_PATH        = p1
    #OUTPUT_PATH         = p2
    LEADATA_PATH         = p2
    BATCH_SIZE           = 1
    LAG_VAR_LIST         = p3
    VAR                  = p4     
    SCALER               = MinMaxScaler(feature_range=(0, 1)) # for the normalization of the continuos data
    NORMALIZE_VAR_LIST   = p5
    histDataFilePath  =  HISTDATA_PATH  
    leadDataFilePath  =  LEADATA_PATH
    histDataset       = pd.read_csv(histDataFilePath)
    leadDataset       = pd.read_csv(leadDataFilePath)
    histDataset['TRDNG_WK_END_DT'] = pd.to_datetime(histDataset['TRDNG_WK_END_DT'])
    leadDataset['TRDNG_WK_END_DT'] = pd.to_datetime(leadDataset['TRDNG_WK_END_DT'])
    print('----HADS & LADS in environment-----')
    print('====================================')
    
    print('---Look for Hyperparameter file----')
    p = r'/shared/sasdata03/ACOE_DEV/ordering_solution/ACoE_Ordering/Master_Ordering_Code/DEV/ver_2.0_testR/R_Code' ### That is hardcoded path
    
    import logging
    hyper = None
    try:
        #csvfile = open('test.xlsx',newline='')
        hyper = pd.read_csv(p+'/'+cncpt+'_LSTM_TUNNED_HYPERPARAMETER.csv')
        histDataset1 = pd.merge(histDataset,hyper,on=['STND_TRRTRY_NM','KEY'],how='left')
    except IOError:
        logging.exception('')
        print("Oops! No such Hyperparameter file")
        histDataset1 = histDataset
        histDataset1['epoch'] = None 
        histDataset1['batch_size'] = None 
        histDataset1['DROP_OUT'] = None 
        histDataset1['RCRNT_DROP_OUT'] = None 
        histDataset1['IN_NEURON'] = None 
        histDataset1['L1_NEURON'] = None     
            

    
    resultF    = pd.DataFrame()
    terr = [v for v in histDataset['STND_TRRTRY_NM'].unique()]
    for ter in terr:
        terrData   = histDataset1[histDataset1.STND_TRRTRY_NM==ter]
        optionList = list(set(terrData.KEY))
        print(len(optionList))
        for ind,option in enumerate(optionList):
            result   = pd.DataFrame()
            print('In progress' +'\n'+ter + '-----'+option )
            histData = histDataset1[(histDataset1.STND_TRRTRY_NM==ter)& (histDataset1.KEY==option)]
            leadData = leadDataset[(leadDataset.STND_TRRTRY_NM==ter)& (leadDataset.KEY==option)]
            h = histData
            h['Flg'] = h['epoch']+h['batch_size']+h['DROP_OUT']+h['RCRNT_DROP_OUT']+h['IN_NEURON']+h['L1_NEURON']
            for i in h['Flg']:
                if i > 0:
                    epoch = histData['epoch'].iloc[ind]
                    DROP_OUT = histData['DROP_OUT'].iloc[ind]
                    RCRNT_DROP_OUT = histData['RCRNT_DROP_OUT'].iloc[ind]
                    IN_NEURON = histData['IN_NEURON'].iloc[ind]
                    L1_NEURON = histData['L1_NEURON'].iloc[ind]
                elif i == 0:
                    epoch, DROP_OUT,RCRNT_DROP_OUT,IN_NEURON,L1_NEURON = LitretureParams()
                elif math.isnan(i):
                    epoch, DROP_OUT,RCRNT_DROP_OUT,IN_NEURON,L1_NEURON = LitretureParams()
                    
            print('---Parameters for LSTM model----')
            print(IN_NEURON,L1_NEURON,DROP_OUT,RCRNT_DROP_OUT,epoch)
            if len(histData) >= 52:
                histData = normalizeData(histData,NORMALIZE_VAR_LIST)
                leadData = normalizeData(leadData,NORMALIZE_VAR_LIST)
                histDataY      = histData.RTL_QTY.values
                histDataTrainY = histDataY
                histDataTrainY = SCALER.fit_transform(histDataTrainY.reshape(-1,1))
                for var in LAG_VAR_LIST:
                    histData,leadData,tempNewVar = createLag(histData,leadData,var)
                    NNVar = VAR+tempNewVar
                histDataX       = histData.loc[:, histData.columns.str.contains('|'.join(NNVar))]
                print(histDataX.columns)
                histDataX       = histDataX.values
                histDataTrainX  = histDataX.reshape((histDataX.shape[0], 1, histDataX.shape[1]))
                print(histDataTrainX.shape)
                leadDataX       = leadData.loc[:, leadData.columns.str.contains('|'.join(NNVar))]
                print(leadDataX.columns)
                leadDataX       = leadDataX.values
                leadDataX       = leadDataX.reshape((leadDataX.shape[0], 1, leadDataX.shape[1]))
                print(leadDataX.shape)
                print(type(histDataTrainX.shape[1]))
                print(type(histDataTrainX.shape[2]))
                print('----Model Building Start----')
                
                keras.backend.clear_session()
                model = Sequential()
                model.add(LSTM(int(IN_NEURON),stateful=True,batch_size=int(BATCH_SIZE),input_shape=(int(histDataTrainX.shape[1]), int(histDataTrainX.shape[2])),dropout=DROP_OUT,recurrent_dropout =RCRNT_DROP_OUT,return_sequences=True))
                model.add(LSTM(int(L1_NEURON),stateful=True,batch_size=int(BATCH_SIZE),dropout =DROP_OUT,recurrent_dropout=RCRNT_DROP_OUT))
                print('GOT')
                model.add(Dense(1))
                model.compile(loss='mae', optimizer='adam')
                history = model.fit(histDataTrainX, histDataTrainY, epochs=int(epoch), batch_size=int(BATCH_SIZE), verbose=0, shuffle=False)
                leadDataYhat = model.predict(leadDataX,batch_size=int(BATCH_SIZE))
                leadDataYhat = SCALER.inverse_transform(leadDataYhat)
                histDataYhat = model.predict(histDataTrainX,batch_size=int(BATCH_SIZE))
                histDataYhat = SCALER.inverse_transform(histDataYhat)
                histData['LSTM_FCST'] = histDataYhat
                leadData['LSTM_FCST'] = leadDataYhat
                histData['LSTM_FLG'] = 1
                leadData['LSTM_FLG'] = 1
            else:
                histData['LSTM_FCST'] = 0
                leadData['LSTM_FCST'] = 0
                histData['LSTM_FLG'] = 0 
                leadData['LSTM_FLG'] = 0
            result = histData.append(leadData)
            resultF = pd.concat([resultF,result])
            cols = ['STND_TRRTRY_NM','KEY','TRDNG_WK_END_DT','LSTM_FCST','LSTM_FLG']
            resultF = resultF[cols]
            print(resultF.shape)
            #print(resultF.head(2))
            resultF.to_csv(usr+'_'+cncpt+'_'+dept+'_'+'LSTM_PYOUT.txt')
            print('-------------------DONE LSTM FORECASTING------------')       
                    
                
                    
                
                