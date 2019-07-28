import os
import time
import math
import pandas as pd
import sys
from matplotlib import pylab as plt
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
#%matplotlib inline 
from matplotlib import pyplot 
import seaborn as sns 
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
plt.rcParams['figure.figsize'] = (15.0, 8.0)
#from bokeh.charts import TimeSeries, output_file, show
from bokeh.io import output_notebook
from bokeh.plotting import figure, show, output_file, output_notebook
from bokeh.palettes import Spectral11, colorblind, Inferno, BuGn, brewer
from bokeh.models import HoverTool, value, LabelSet, Legend, ColumnDataSource,LinearColorMapper,BasicTicker, PrintfTickFormatter, ColorBar
import datetime
#output_notebook()
#from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
import numpy
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from keras.optimizers import SGD
from keras.constraints import maxnorm
from sklearn.model_selection import train_test_split
import math

def ChooseTerr(hads,t1):
    hads_2 = hads[hads['STND_TRRTRY_NM']==t1]
    #lads_2 = lads[(lads['STND_TRRTRY_NM']==t1)]
    print(set(hads_2['STND_TRRTRY_NM']))
    #print(set(lads_2['STND_TRRTRY_NM']))
    return hads_2

def transformation(bahrain):
    '''
    Transform the data and make it ready for consuming Time -Based clustering 
    '''
    terr = bahrain['STND_TRRTRY_NM'].unique()
    terr = str(terr)
    d = bahrain.pivot_table(index='TRDNG_WK_END_DT',values='RTL_QTY',columns='KEY_OPTN')
    d = d.describe()
    d = d.reset_index()
    d_T = d.T
    d_T = d_T.drop(d_T.index[0])
    d_T = d_T.reset_index()
    d_T.rename(columns = {1:'MEAN',2:'STD_DEV'}, inplace = True)
    d_T = d_T[['KEY_OPTN','MEAN','STD_DEV']]
    d = bahrain.pivot_table(index='TRDNG_WK_END_DT',values='RTL_QTY',columns='KEY_OPTN')
    df_sp = pd.DataFrame(d.isnull().sum())
    df_sp.rename(columns = {0:'SPARSITY'}, inplace = True)
    n_shape = df_sp.shape[0]
    df_sp['SPARCITY_%'] = df_sp['SPARSITY']/n_shape
    print(d_T.shape,df_sp.shape)
    df_sp = df_sp.reset_index()
    df = pd.concat([d_T,df_sp['SPARCITY_%']], axis=1)
    df['TERR'] = terr
    return df

'''
DOING A CLUSTER ON BASIS OF TIME SERIES COMPONENTS
'''
def prepare_encodings(df):
    mean = []
    std  = []
    spar = []
    for i in df['MEAN']:
        if i <= np.median(df['MEAN']):
             mean.append('L')
        elif i > np.percentile(df['MEAN'],75):
            mean.append('H')
        else:
            mean.append('M')
    df['MEAN_ENCD'] = pd.Series(mean)
    print(df.shape)

    for i in df['STD_DEV']:
        if i <= np.median(df['STD_DEV']):
            std.append('L')
        elif i > np.percentile(df['STD_DEV'],75):
            std.append('H')
        else:
            std.append('M')
    df['STD_DEV_ENCD'] = pd.Series(std)
    print(df.shape)

    for i in df['SPARCITY_%']:
        if i > np.median(df['SPARCITY_%']):
            spar.append('H')
        else:
            spar.append('L')
    
    df['SPAR_EN'] = pd.Series(spar)
    print(df.shape)
    df['concat'] = df['MEAN_ENCD']+df['STD_DEV_ENCD']+df['SPAR_EN']
    return df 

'''
DOING RARE LABEL ENCODING AND PREPARE THE FINAL CLUSTER DATA 
'''

def prepare_cluster(df,variable):
    # find frequent labels / discrete numbers
    temp = df.groupby([variable])[variable].count()/np.float(len(df))
    print(temp.head())
    frequent_cat = [x for x in temp.loc[temp>0.13].index.values]
    print(frequent_cat)
    df[variable] = np.where(df[variable].isin(frequent_cat), df[variable], 'RARE')


'''
Extract the Longest Series  
'''

def longest_series_tagging(df,riyadh_tr):
    tmp1=df.merge(riyadh_tr,on=['STND_TRRTRY_NM','KEY_OPTN'],how='inner')
    tmp2=tmp1.groupby(['STND_TRRTRY_NM','KEY_OPTN','concat']).agg({'concat':{'count_cl':'count'}}).reset_index()
    tmp2.columns=tmp2.columns.droplevel(level=0)
    tmp2.columns=(['STND_TRRTRY_NM','KEY_OPTN','concat','count_cl'])
    cls_list = [v for v in tmp2['concat'].unique()]
    cls_brac = []
    v = pd.DataFrame()
    for i in  cls_list:
        print (i)
        MX1=tmp2.loc[tmp2['concat']==i]
        cls_brac.append(MX1)
        OP=MX1.loc[MX1['count_cl'].idxmax()]
        v = v.append(OP)
    return v

def final_df_prep(df,riyad):
    a = df.merge(riyad,on=['STND_TRRTRY_NM','KEY_OPTN'],how='inner')
    print(a.shape)
    return a

'''
TIME BASED SPLITTING
'''
def time_based_split(riyad_fl_df):
    riyad_fl_df.sort_values(['TRDNG_WK_END_DT'],ascending=[True], inplace=True)
    row_shape_test = math.ceil(riyad_fl_df.shape[0]*0.1)
    row_shape_train = math.floor(riyad_fl_df.shape[0]*0.9)
    train = riyad_fl_df.head(row_shape_train)
    test = riyad_fl_df.tail(row_shape_test)
    return train,test

def train_test_splt(riyad_fl_df):
    train_df = pd.DataFrame()
    test_df = pd.DataFrame()
    riyad_fl_df['TRDNG_WK_END_DT'] =pd.to_datetime(riyad_fl_df['TRDNG_WK_END_DT'])
    cls_list = [v for v in riyad_fl_df['concat'].unique()]
    tmp2=riyad_fl_df.sort_values(['concat','TRDNG_WK_END_DT'],ascending=[True,True])
    for i in cls_list:
        tmp = tmp2[tmp2['concat'] == i]
        #train, test = train_test_split(tmp, test_size=0.2,shuffle=False,tmp['TRDNG_WK_END_DT'])
        train, test = time_based_split(tmp)
        train_df = train_df.append(train)
        test_df = test_df.append(test)
    return train_df,test_df


'''    
DATA PREP 
'''
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

def createLag(dfh,dfl,lagVar, normalize = False):
    newVar = []
    lags =  findLags(dfh,lagVar)
   
    for i in range(lags):
        dfh[lagVar + '_LAG_%s'%i] = dfh[lagVar].shift(i+1)
        dfh[lagVar + '_LAG_%s'%i].fillna(0, inplace=True)
        dfl[lagVar + '_LAG_%s'%i] = dfl[lagVar].shift(i+1)
        dfl[lagVar + '_LAG_%s'%i].fillna(0, inplace=True)
        #dfv[lagVar + '_LAG_%s'%i] = dfv[lagVar].shift(i+1)
        #dfv[lagVar + '_LAG_%s'%i].fillna(0, inplace=True)
        newVar.append(lagVar + '_LAG_%s'%i)
        if normalize:
            df[lagVar + '_LAG_%s'%i] = SCALER.fit_transform(df[lagVar + '_LAG_%s'%i] .values.reshape(-1,1))
    
    return (dfh,dfl,newVar)

def modelLSTM_new(row):

    #Row specifications
    epoch_fix=int(row["epoch"])
    BATCH_SIZE=int(row["batch_size"])
    DROP_OUT =float(row["DROP_OUT"])
    RCRNT_DROP_OUT = float(row["RCRNT_DROP_OUT"])
    IN_NEURON = int(row["IN_NEURON"])
    L1_NEURON = int(row["L1_NEURON"])
    territory=str(row["Territory"])
    cluster=str(row["Cluster"])

    #Dataframe for a perticular territory and cluster
    partitionedDataFrame_train=final_train_df[(final_train_df.STND_TRRTRY_NM==territory) & (final_train_df.concat==cluster)]
    partitionedDataFrame_test=final_test_df[(final_test_df.STND_TRRTRY_NM==territory) & (final_test_df.concat==cluster)]
    
    #Model data preparation
    X_train,y_train = partitionedDataFrame_train.drop(['RTL_QTY'],axis=1),partitionedDataFrame_train['RTL_QTY']
    X_test,y_test = partitionedDataFrame_test.drop(['RTL_QTY'],axis=1),partitionedDataFrame_test['RTL_QTY']  
    X_train =X_train[VAR]
    X_test = X_test[VAR]
    X_train = normalizeData(X_train,NORMALIZE_VAR_LIST)
    X_test = normalizeData(X_test,NORMALIZE_VAR_LIST)
    y_train = y_train.values
    y_test = y_test.values
    y_train_S = SCALER.fit_transform(y_train.reshape(-1,1))
    y_test_S = SCALER.fit_transform(y_test.reshape(-1,1))
    for var in LAG_VAR_LIST:
        X_train,X_test,tempNewVar = createLag(X_train,X_test,var)
        NNVar = VAR+tempNewVar
    X_train_v = X_train.values
    X_test_v = X_test.values
    X_train_reshaped = X_train_v.reshape(X_train.shape[0], 1, X_train.shape[1])
    X_test_reshaped = X_test_v.reshape(X_test.shape[0], 1, X_test.shape[1])
    input_shape = (X_train_reshaped.shape[1],X_train_reshaped.shape[2])

    #Model specification
    keras.backend.clear_session()
    model = Sequential()
    model.add(LSTM(IN_NEURON,stateful=True,batch_size=BATCH_SIZE,input_shape=input_shape,dropout=DROP_OUT,recurrent_dropout = RCRNT_DROP_OUT,return_sequences=True))
    model.add(LSTM(L1_NEURON,stateful=True,batch_size=BATCH_SIZE,dropout = DROP_OUT,recurrent_dropout=RCRNT_DROP_OUT))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam',metrics=['accuracy'])

    #Model train
    model.fit(X_train_reshaped,y_train, epochs=epoch_fix, batch_size=BATCH_SIZE, verbose=0, shuffle=False)
    y_test_pred = model.predict(X_test_reshaped,batch_size=BATCH_SIZE)
    y_test_pred = SCALER.inverse_transform(y_test_pred)
    mse=np.sqrt(mean_squared_error(y_test,y_test_pred))
    return mse

def parallel_call(terr):
    clusterList = list(set(final_train_df["concat"][final_train_df.STND_TRRTRY_NM==terr]))
    print(clusterList)
    consolidatedResults=pd.DataFrame()
    for clusterLopp in clusterList:    
        print('------Randomized Grid Search starting ------------')
        modelValidateDf=pd.DataFrame()
        modelValidateDf["batch_size"]=[1]
        dictForColumnsToTune={}
        dictForColumnsToTune["epoch"]=(1,50,1)
        dictForColumnsToTune["DROP_OUT"]=(0.1,0.3,0.05)
        dictForColumnsToTune["RCRNT_DROP_OUT"]=(0.1,0.3,0.05)
        dictForColumnsToTune["IN_NEURON"]=(1,50,1)
        dictForColumnsToTune["L1_NEURON"]=(1,50,1)
        for i in dictForColumnsToTune.keys():
            tempDf=pd.DataFrame()
            tempDf[i]=numpy.arange(dictForColumnsToTune[i][0],dictForColumnsToTune[i][1] ,dictForColumnsToTune[i][2] )
            tempDf['key'] = 0
            modelValidateDf['key'] = 0
            tempDf=tempDf.merge(modelValidateDf, how='outer')
            tempDf=tempDf.drop("key",axis=1)
            modelValidateDf=tempDf
        print('-----Randomized Grid Search-------')
        modelValidateDf=modelValidateDf[["epoch","batch_size","DROP_OUT","RCRNT_DROP_OUT","IN_NEURON","L1_NEURON"]]
        modelValidateDf["Cluster"]=clusterLopp
        modelValidateDf["Territory"]=terr
        sampleModelValidateDf=modelValidateDf.sample(30)

        sampleModelValidateDf['MSE'] = sampleModelValidateDf.apply(modelLSTM_new, axis=1)
        sampleModelValidateDf.to_csv('HYPERPARAMETER'+'_'+terr+'_'+clusterLopp+'.csv')
        consolidatedResults.append(sampleModelValidateDf)
    consolidatedResults
    return(consolidatedResults)


if __name__== '__main__':
    NORMALIZE_VAR_LIST = ["DISC_PER","DSP_STR"]
    LAG_VAR_LIST = ["FLG_SEOSS","FLG_WEOSS","FLG_SPOSS","FLG_RAMADAN","FLG_EID2","FLG_MSOS"]
    VAR = ["FLG_SEOSS","FLG_WEOSS","FLG_SPOSS","FLG_RAMADAN","FLG_EID2","DISC_PER","DSP_STR","FLG_NATIONAL_DAY","FLG_BTS","FLG_MSOS","FLG_MSOW","FLG_MSOP","OOS_PER_A","OOS_PER_B","OOS_PER_C"]
    SCALER = MinMaxScaler(feature_range=(0, 1))
    path = '/home/mainakk/HYPERPARAMETER_TUNNING/hyper_parameter_output'
    os.chdir(path)
    df = pd.read_csv('TRK_BS_CLTB_FCST_HADS_19.csv')
    df['TRDNG_WK_END_DT'] =pd.to_datetime(df.TRDNG_WK_END_DT)
    df= df.sort_values('TRDNG_WK_END_DT')
    TERRY = df['STND_TRRTRY_NM'].unique()
    df_dict={}
    for i in TERRY:
        df_dict[i] = ChooseTerr(df,t1=i)

    bahrain = pd.DataFrame(df_dict['Bahrain'])
    dammam = pd.DataFrame(df_dict['Dammam - KSA'])
    egpyt = pd.DataFrame(df_dict['Egypt'])
    jeddah = pd.DataFrame(df_dict['Jeddah - KSA'])
    kuwait = pd.DataFrame(df_dict['Kuwait'])
    lebenon = pd.DataFrame(df_dict['Lebanon'])
    oman = pd.DataFrame(df_dict['Oman'])
    quatar = pd.DataFrame(df_dict['Qatar'])
    riyadh = pd.DataFrame(df_dict['Riyadh - KSA'])
    thailand = pd.DataFrame(df_dict['Thailand'])
    uae = pd.DataFrame(df_dict['United Arab Emirates'])

    bahrain_tr = transformation(bahrain)    
    dammam_tr = transformation(dammam)    
    egpyt_tr = transformation(egpyt)    
    jeddah_tr = transformation(jeddah)   
    kuwait_tr  = transformation(kuwait)   
    lebenon_tr = transformation(lebenon)
    oman_tr = transformation(oman)
    quatar_tr = transformation(quatar)  
    riyadh_tr = transformation(riyadh)   
    thailand_tr = transformation(thailand)
    uae_tr = transformation(uae)

    dammam_tr = prepare_encodings(dammam_tr) 
    bahrain_tr = prepare_encodings(bahrain_tr)    
    dammam_tr = prepare_encodings(dammam_tr)    
    egpyt_tr = prepare_encodings(egpyt_tr)    
    jeddah_tr = prepare_encodings(jeddah_tr)   
    kuwait_tr  = prepare_encodings(kuwait_tr)   
    lebenon_tr = prepare_encodings(lebenon_tr)
    oman_tr  = prepare_encodings(oman_tr)
    quatar_tr = prepare_encodings(quatar_tr)  
    riyadh_tr = prepare_encodings(riyadh_tr)   
    thailand_tr = prepare_encodings(thailand_tr)
    uae_tr = prepare_encodings(uae_tr)

    prepare_cluster(dammam_tr,'concat') 
    prepare_cluster(bahrain_tr,'concat')    
    prepare_cluster(dammam_tr,'concat')    
    prepare_cluster(egpyt_tr,'concat')    
    prepare_cluster(jeddah_tr,'concat')   
    prepare_cluster(kuwait_tr,'concat')   
    prepare_cluster(lebenon_tr,'concat')
    prepare_cluster(oman_tr,'concat')
    prepare_cluster(quatar_tr,'concat')  
    prepare_cluster(riyadh_tr,'concat')   
    prepare_cluster(thailand_tr,'concat')
    prepare_cluster(uae_tr,'concat')
    
    dammam_tr['STND_TRRTRY_NM'] = 'Dammam - KSA'
    bahrain_tr['STND_TRRTRY_NM'] = 'Bahrain'
    egpyt_tr['STND_TRRTRY_NM'] = 'Egypt'
    jeddah_tr['STND_TRRTRY_NM'] = 'Jeddah - KSA'
    kuwait_tr['STND_TRRTRY_NM'] = 'Kuwait'
    lebenon_tr['STND_TRRTRY_NM'] = 'Lebanon'
    oman_tr['STND_TRRTRY_NM'] = 'Oman'
    quatar_tr['STND_TRRTRY_NM'] ='Qatar'
    riyadh_tr['STND_TRRTRY_NM'] = 'Riyadh - KSA'
    uae_tr['STND_TRRTRY_NM'] = 'United Arab Emirates'
    thailand_tr['STND_TRRTRY_NM'] = 'Thailand'

    omn = longest_series_tagging(df,oman_tr)
    dmmn = longest_series_tagging(df,dammam_tr)
    bhrn = longest_series_tagging(df,bahrain_tr)
    egpt = longest_series_tagging(df,egpyt_tr)
    jed = longest_series_tagging(df,jeddah_tr)
    kuwat = longest_series_tagging(df,kuwait_tr)
    lebnon = longest_series_tagging(df,lebenon_tr)
    qtr = longest_series_tagging(df,quatar_tr)
    riyad = longest_series_tagging(df,riyadh_tr)
    uaee = longest_series_tagging(df,uae_tr)
    thai = longest_series_tagging(df,thailand_tr)

    omn_fl_df = final_df_prep(df,omn)
    dmmn_fl_df = final_df_prep(df,dmmn)
    bhrn_fl_df = final_df_prep(df,bhrn)
    egpt_fl_df = final_df_prep(df,egpt)
    jed_fl_df = final_df_prep(df,jed)
    kuwat_fl_df = final_df_prep(df,kuwat)
    lebnon_fl_df = final_df_prep(df,lebnon)
    qtr_fl_df = final_df_prep(df,qtr)
    riyad_fl_df = final_df_prep(df,riyad)
    uae_fl_df = final_df_prep(df,uaee)
    thai_fl_df = final_df_prep(df,thai) 

    train_riyad,test_riyad = train_test_splt(riyad_fl_df)
    train_oman,test_oman = train_test_splt(omn_fl_df)
    train_dmn,test_dmn = train_test_splt(dmmn_fl_df)
    train_bhrn,test_bhrn = train_test_splt(bhrn_fl_df)
    train_egpyt,test_egpyt = train_test_splt(egpt_fl_df)
    train_jeddah,test_jeddah = train_test_splt(jed_fl_df)
    train_kwt,test_kwt = train_test_splt(kuwat_fl_df)
    train_lebn,test_lebn = train_test_splt(lebnon_fl_df)
    train_qtr, test_qtr = train_test_splt(qtr_fl_df)
    train_uae,test_uae = train_test_splt(uae_fl_df)
    train_thai,test_thai = train_test_splt(thai_fl_df)

    final_train_df = train_riyad.append(train_oman).append(train_dmn).append(train_bhrn).append(train_egpyt).append(train_jeddah).append(train_kwt).append(train_lebn).append(train_qtr).append(train_uae).append(train_thai)
    final_test_df = test_riyad.append(test_oman).append(test_dmn).append(test_bhrn).append(test_egpyt).append(test_jeddah).append(test_kwt).append(test_lebn).append(test_qtr).append(test_uae).append(test_thai)
    print(final_train_df.shape,final_test_df.shape)
    print(len(final_train_df['STND_TRRTRY_NM'].value_counts()))
    print(len(final_test_df['STND_TRRTRY_NM'].value_counts()))

    terrList = list(set(final_train_df.STND_TRRTRY_NM))
    start = time.clock()
    print('-----Start time of Model building %s:'%(start))
    with closing(Pool(processes=len(terrList))) as pool:
        resultDFList = pool.map(parallel_call,terrList)
        pool.terminate()
    t = time.clock() - start
    st2 = time.clock()
    print('-----End time of Model building %s:'%(st2))
    print('Model took %s seconds'%(t))
    print('---------------LSTM HYPERPARAMETER TUNNING DONE -------------')
    resultDF = pd.concat(resultDFList)
    print(resultDF.head(5))
    resultDF.to_csv('PARALLEL_HYPERPARAMETERS.csv')

    #----------------------------- End -----------------------------------------------------------------
 
    