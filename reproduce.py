import numpy as np
import pandas as pd  
import tushare as ts
import talib as ta

from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier,XGBRegressor
from catboost import CatBoostClassifier,CatBoostRegressor
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.model_selection import train_test_split,KFold,cross_val_score
from sklearn.metrics import accuracy_score
import shap
from sklearn.feature_selection import SelectKBest,f_regression
from sklearn import preprocessing

import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px

get_ipython().run_line_magic('matplotlib', 'inline')

from pylab import mpl
mpl.rcParams['font.sans-serif']=['SimHei']
mpl.rcParams['axes.unicode_minus']=False


#data
#以上证指数数据为例
def get_data(code='sh',start='2000-01-01',end='2021-03-02'):
    df=ts.get_k_data('sh',start='2005')
    df.index=pd.to_datetime(df.date)
    df=df[['open','high','low','close','volume']]
    return df

df=get_data()
df_train,df_test=df.loc[:'2017'],df.loc['2018':]

def trade_signal(data,short=10,long=60,tr_id=False):
    data['SMA1'] = data.close.rolling(short).mean()
    data['SMA2'] = data.close.rolling(long).mean() 
    data['signal'] = np.where(data['SMA1'] >data['SMA2'], 1.0, 0.0) 
    if(tr_id is not True):
        display(data['signal'].value_counts())
    
df_tr1 = df_train.copy(deep=True)  
df_te1 = df_test.copy(deep=True) 
trade_signal(df_tr1)  #
trade_signal(df_te1,tr_id=True)  


plt.figure(figsize=(14,12), dpi=80)
ax1 = plt.subplot(211)
plt.plot(df_tr1.close,color='b')
plt.title('上证指数走势',size=15)
plt.xlabel('')
ax2 = plt.subplot(212)
plt.plot(df_tr1.signal,color='r')
plt.title('交易信号',size=15)
plt.xlabel('')
plt.show()


df_tr1[['SMA1','SMA2','signal']].iloc[-250:].plot(figsize=(14,6),secondary_y=['signal'])
plt.show()



#删除均线变量
df_tr1=df_tr1.drop(['SMA1','SMA2'], axis=1)
df_te1=df_te1.drop(['SMA1','SMA2'], axis=1)



cmap = sns.diverging_palette(220, 10, as_cmap=True)
def corrMat(df,target='demand',figsize=(9,0.5),ret_id=False):
    
    corr_mat = df.corr().round(2);shape = corr_mat.shape[0]
    corr_mat = corr_mat.transpose()
    corr = corr_mat.loc[:, df.columns == target].transpose().copy()
    
    if(ret_id is False):
        f, ax = plt.subplots(figsize=figsize)
        sns.heatmap(corr,vmin=-0.3,vmax=0.3,center=0, 
                     cmap=cmap,square=False,lw=2,annot=True,cbar=False)
        plt.title(f'Feature Correlation to {target}')
    
    if(ret_id):
        return corr


corrMat(df_tr1,'signal',figsize=(7,0.5))


#复制
df_tr2=df_tr1.copy(deep=True)
df_te2=df_te1.copy(deep=True)


#技术指标

def indicators(data):
    data['MA13']=ta.MA(data.close,timeperiod=13)
    data['MA34']=ta.MA(data.close,timeperiod=34)
    data['MA89']=ta.MA(data.close,timeperiod=89)
    data['EMA10']=ta.EMA(data.close,timeperiod=10)
    data['EMA30']=ta.EMA(data.close,timeperiod=30)
    data['EMA200']=ta.EMA(data.close,timeperiod=200)
    data['MOM10']=ta.MOM(data.close,timeperiod=10)
    data['MOM30']=ta.MOM(data.close,timeperiod=30)
    data['RSI10']=ta.RSI(data.close,timeperiod=10)
    data['RSI30']=ta.RSI(data.close,timeperiod=30)
    data['RS200']=ta.RSI(data.close,timeperiod=200)
    data['K10'],data['D10']=ta.STOCH(data.high,data.low,data.close, fastk_period=10)
    data['K30'],data['D30']=ta.STOCH(data.high,data.low,data.close, fastk_period=30)
    data['K20'],data['D200']=ta.STOCH(data.high,data.low,data.close, fastk_period=200)



indicators(df_tr2)
indicators(df_te2)


corrMat(df_tr2,'signal',figsize=(15,0.5))

#删除缺失值
df_tr2 = df_tr2.dropna() 
df_te2 = df_te2.dropna()


#对数据进行拟合和评估

models = []
#轻量级模型 
#线性监督模型
models.append(('LR', LogisticRegression(n_jobs=-1)))
models.append(('TREE', DecisionTreeClassifier())) 
#非监督模型
models.append(('LDA', LinearDiscriminantAnalysis())) 
models.append(('KNN', KNeighborsClassifier())) 
models.append(('NB', GaussianNB())) 
#高级模型
models.append(('GBM', GradientBoostingClassifier(n_estimators=25)))
models.append(('XGB',XGBClassifier(n_estimators=25,use_label_encoder=False)))
models.append(('CAT',CatBoostClassifier(silent=True,n_estimators=25)))
models.append(('RF', RandomForestClassifier(n_estimators=25)))


def modelEval(ldf,feature='signal',split_id=[None,None],eval_id=[True,True,True,True],
              n_fold=5,scoring='accuracy',cv_yrange=None,hm_vvals=[0.5,1.0,0.75]):
    

    if(split_id[0] is not None):
        train_df,eval_df = train_test_split(ldf,test_size=split_id[0],shuffle=False)
    elif(split_id[1] is not None):
        train_df = df.loc[:split_id[1]]; eval_df = df.loc[split_id[1]:] 
    else:
        print('Choose One Splitting Method Only')
        
    ''' Train/Test Feature Matrices + Target Variables Split'''
    y_train = train_df[feature]
    X_train = train_df.loc[:, train_df.columns != feature]
    y_eval = eval_df[feature]
    X_eval = eval_df.loc[:, eval_df.columns != feature]
    X_one = pd.concat([X_train,X_eval],axis=0)
    y_one = pd.concat([y_train,y_eval],axis=0)
 
    ''' Cross Validation, Training/Evaluation, one evaluation'''
    lst_res = []; names = []; lst_train = []; lst_eval = []; lst_one = []; lst_res_mean = []
    if(any(eval_id)):
        for name, model in models:  
            names.append(name)
            
            # Cross Validation Model on Training Se
            if(eval_id[0]):
                kfold = KFold(n_splits=n_fold, shuffle=True)
                cv_res = cross_val_score(model,X_train,y_train, cv=kfold, scoring=scoring)
                lst_res.append(cv_res)
                
            # Evaluate Fit Model on Training Data
            if(eval_id[1]):
                res = model.fit(X_train,y_train)
                train_res = accuracy_score(res.predict(X_train),y_train); lst_train.append(train_res)
            if(eval_id[2]):
                if(eval_id[1] is False):  # If training hasn't been called yet
                    res = model.fit(X_train,y_train)
                eval_res = accuracy_score(res.predict(X_eval),y_eval); lst_eval.append(eval_res)

            # Evaluate model on entire dataset
            if(eval_id[3]):
                res = model.fit(X_one,y_one)
                one_res = accuracy_score(res.predict(X_one),y_one); lst_one.append(one_res)
                
            ''' [out] Verbal Outputs '''
            lst_res_mean.append(cv_res.mean())
            fn1 = cv_res.mean()
            fn2 = cv_res.std();
            fn3 = train_res
            fn4 = eval_res
            fn5 = one_res
       
    s0 = pd.Series(np.array(lst_res_mean),index=names)
    s1 = pd.Series(np.array(lst_train),index=names)
    s2 = pd.Series(np.array(lst_eval),index=names)
    s3 = pd.Series(np.array(lst_one),index=names)
    pdf = pd.concat([s0,s1,s2,s3],axis=1)
    pdf.columns = ['cv_average','train','test','all']
        
    ''' Visual Ouputs '''
    sns.set(style="whitegrid")
    fig,ax = plt.subplots(1,2,figsize=(15,4))
    ax[0].set_title(f'{n_fold} Cross Validation Results')
    sns.boxplot(data=lst_res, ax=ax[0], orient="v",width=0.3)
    ax[0].set_xticklabels(names)
    sns.stripplot(data=lst_res,ax=ax[0], orient='v',color=".3",linewidth=1)
    ax[0].set_xticklabels(names)
    ax[0].xaxis.grid(True)
    ax[0].set(xlabel="")
    if(cv_yrange is not None):
        ax[0].set_ylim(cv_yrange)
    sns.despine(trim=True, left=True)
    sns.heatmap(pdf,vmin=hm_vvals[0],vmax=hm_vvals[1],center=hm_vvals[2],
            ax=ax[1],square=False,lw=2,annot=True,fmt='.3f',cmap='Blues')
    ax[1].set_title('Accuracy Scores')
    plt.show()


#使用原始特征值
modelEval(df_tr1,split_id=[0.2,None])
#增加技术指标特征值

modelEval(df_tr2,split_id=[0.2,None],cv_yrange=(0.8,1.0),hm_vvals=[0.8,1.0,0.9])

#模型效率优化

def feature_importance(ldf,feature='signal',n_est=100):
    # Input dataframe containing feature & target variable
    X = ldf.copy()
    y = ldf[feature].copy()
    del X[feature]

    # CORRELATION
    imp = corrMat(ldf,feature,figsize=(15,0.5),ret_id=True)
    del imp[feature]
    s1 = imp.squeeze(axis=0);s1 = abs(s1)
    s1.name = 'Correlation'      

    # SHAP
    model = CatBoostRegressor(silent=True,n_estimators=n_est).fit(X,y)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    shap_sum = np.abs(shap_values).mean(axis=0)
    s2 = pd.Series(shap_sum,index=X.columns,name='Cat_SHAP').T

    # RANDOMFOREST
    model = RandomForestRegressor(n_est,random_state=0, n_jobs=-1)
    fit = model.fit(X,y)
    rf_fi = pd.DataFrame(model.feature_importances_,index=X.columns,                                  
            columns=['RandForest']).sort_values('RandForest',ascending=False)
    s3 = rf_fi.T.squeeze(axis=0)

    # XGB
    model=XGBRegressor(n_estimators=n_est,learning_rate=0.5,verbosity = 0)
    model.fit(X,y)
    data = model.feature_importances_
    s4 = pd.Series(data,index=X.columns,name='XGB').T

    # KBEST
    model = SelectKBest(k=5, score_func=f_regression)
    fit = model.fit(X,y)
    data = fit.scores_
    s5 = pd.Series(data,index=X.columns,name='K_best')

    # Combine Scores
    df0 = pd.concat([s1,s2,s3,s4,s5],axis=1)
    df0.rename(columns={'target':'lin corr'})
    x = df0.values 
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df = pd.DataFrame(x_scaled,index=df0.index,columns=df0.columns)
    df = df.rename_axis('Feature Importance via', axis=1)
    df = df.rename_axis('Feature', axis=0)
    pd.options.plotting.backend = "plotly"
    fig = df.plot(kind='bar',title='Scaled Feature Importance')
    fig.show()



feature_importance(df_tr2)


df_tr2_FI = df_tr2.drop(columns=['open','high','low','close','EMA10'])


modelEval(df_tr2_FI,split_id=[0.2,None],cv_yrange=(0.8,1.0),hm_vvals=[0.8,1.0,0.9])

