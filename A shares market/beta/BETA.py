BETA

'''

该代码基于《Bali-Empirical Asset Pricing》编写，原书中使用的是CRSP美股数据，
而本代码则使用CSMAR数据，主要用于计算和分析中国A股市场中beta的特征。
具体内容包括不同时间跨度与计算方法下的beta展示、beta的描述性统计、股票组合的beta分析以及Fama-MacBeth回归分析。
代码参考了WHU-Fintech Workshop的相关代码。在后续部分，我会对每一行代码进行详细注释，帮助理解，也帮助我再一次复习不同beta计算逻辑、组合分析和FM回归的方法。

'''
# 导入必要模块
import numpy as np
import pandas as pd
import datetime as dt
import os
import statsmodels.formula.api as smf
from scipy.stats import pearsonr
from scipy.stats import spearmanr
import statsmodels.api as sm
from scipy.stats.mstats import winsorize


# 导入数据
os.chdir(r'/Users/mac/Desktop/Asset Pricing/A shares market/beta/data')
ret = pd.read_csv(os.path.join('TRD_Month.csv'))
daily_data = pd.read_csv(os.path.join('TRD_daily.csv'))
ff5 = pd.read_csv(os.path.join('fivefactor_monthly.csv')) #数据来源：央财因子数据库
mktcap = pd.read_csv(os.path.join('fivefactor_daily.csv'))


#定义获取收益和因子数据的年份与月份的函数，便于后续处理
def get_month(table,key):
    table[key] = pd.to_datetime(table[key])
    table['year'] = table[key].dt.year
    table['month'] = table[key].dt.month
    return table

def get_month2(table,key):
    table[key] = pd.to_datetime(table[key],format = '%Y%m')
    table['year'] = table[key].dt.year
    table['month'] = table[key].dt.month
    return table


#计算月度超额收益
'''
**1997-2023月度数据，97年开始的原因是因为，我国A股市场96年实行涨跌停板机制，为了避免影响故从97年开始
**全A股收益率（包含上证A，深A，科创板，创业板）

Trdmnt:股票收益时间，CSMAR字段
trdmn:无风险收益时间，CSMAR字段
'''
ret1 = get_month(ret,'Trdmnt') #获取股票月度收益时间
ff5 = get_month2(ff5,'trdmn') #从因子数据库 ff5 中获取无风险收益的时间

#筛选1997-2023的股票
def time(df,t1,t2):
    df = df[(df['year']>=t1)&(df['year']<=t2)]
    return df
ret1 = time(ret1,1997,2023)
ff5 = time(ff5,1997,2023)
ff = ff5['mkt_rf']
ff.index = ff5['trdmn']

#筛选A股，因为原数据中包含了除主板、创业板和科创板以外的股票，因此需要筛选剔除
def filt(df,x1,x2,x3,x4):
    df = df[(df['Markettype'] == x1)|(df['Markettype'] == x2)|(df['Markettype'] == x3)|(df['Markettype'] == x4)]
    return df

ret2 = filt(ret1,1,4,16,32)

#合并收益率数据计算超额收益
def data(inx,col,value): 
    temp = ff5[['year','month','mkt_rf','rf']]
    df = pd.merge(ret2,temp,on = ['year','month'])   #两个表因为交易时间问题，时间并不一致所以通过年份和月份进行合并 
    df['rt'] = df['Mretnd'] - df['rf']
    month_data = pd.pivot(df,index=inx,columns=col,values=value)
    month_data['month_num'] = (month_data.index.year-1997)*12+month_data.index.month   
    return month_data

month_data = data('Trdmnt','Stkcd','rt')

# 修改column名称便于理解和后续处理

## 修改daily_data中收益列的名称，Dretwd改为rt
daily_data = daily_data.rename(columns={'Dretwd': 'rt'})

#修改daily_data中的日期列的名称，Trddt改为date
daily_data = daily_data.rename(columns={'Trddt': 'date'})

#修改mktcap的时间列名，trddy修改为date
mktcap = mktcap.rename(columns={'trddy': 'date'})

#修改daily_data代码列名，Stkcd改为code
daily_data = daily_data.rename(columns={'Stkcd': 'code'})

##筛选日度数据
daily_data = get_month(daily_data,'date')
daily_data = time(daily_data,1997,2023)
daily_data = filt(daily_data,1,4,16,32)
daily_data['month_num'] = (daily_data['year']-1997)*12 + daily_data['month']

##合并收益率数据与无风险利率
mktcap['date'] =  pd.to_datetime(mktcap['date'])
mktcap = mktcap[['date','mkt_rf','rf']]
daily_data = pd.merge(daily_data,mktcap,on='date')
daily_data.head(10)
daily_data['rt'] = daily_data['rt']-daily_data['rf'] 
full_data = pd.pivot(daily_data,index='date',columns='code',values='rt')
full_data['month_num'] = (full_data.index.year-1997)*12+full_data.index.month
mktcap['mkt'] = mktcap['mkt_rf'] + mktcap['rf']
mktcap.index = mktcap['date']



def beta_calculator(data,factor,span,low_limit):
    '''
    用来计算beta的表格函数，输出是某一种计算方式的beta的表格。
    
    输入参数
    ----------
    data是以month_num为columns，code为index，rt为value
    span是每次回归跨度月份数，一年为12
    low_limit是计算beta的最低样本数（天数），一个月为10，三个月为50等
    输出
    -------
    index为股票代码，columns为月份编号，value为对应规则算出beta 的df
    '''
    X = pd.DataFrame()
    for i in range(max(data['month_num'])-span+1):
        same_time_data = data[(data['month_num']>i)&(data['month_num']<=i+span)]
        same_time = []
        code_list = list(same_time_data.columns[:-1])
        for code in code_list:
            temp_data = same_time_data[code]
            temp_data.name = 'rt'
            reg_data = pd.concat([temp_data,factor],axis=1,join='inner')
            if reg_data['rt'].notna().sum() >= low_limit:
                model = smf.ols('rt~mkt_rf',reg_data,missing='drop').fit()
                beta = model.params[1]
            else:
                beta = np.nan
            same_time.append(beta)
        same_time = pd.Series(same_time,index = code_list,name = i+span)
        X = pd.concat([X,same_time],axis=1)
    return X


# 计算不同时间窗口的 beta
beta_1m = beta_calculator(full_data,mktcap,1,10)
beta_3m = beta_calculator(full_data,mktcap,3,50)
beta_6m = beta_calculator(full_data,mktcap,6,100)
beta_12m = beta_calculator(full_data,mktcap,12,200)
beta_24m = beta_calculator(full_data,mktcap,24,450)
beta_1y = beta_calculator(month_data,ff,12,10)
beta_2y = beta_calculator(month_data,ff,24,20)
beta_3y = beta_calculator(month_data,ff,36,24)
beta_5y = beta_calculator(month_data,ff,60,24)

# 输出beta为csv，方便之后读取（因为算力限制无法直接进行后续计算）
beta_1m.to_csv('beta_1m.csv')
beta_3m.to_csv('beta_3m.csv')
beta_6m.to_csv('beta_6m.csv')
beta_12m.to_csv('beta_12m.csv')
beta_24m.to_csv('beta_24m.csv')
beta_1y.to_csv('beta_1y.csv')
beta_2y.to_csv('beta_2y.csv')
beta_3y.to_csv('beta_3y.csv')
beta_5y.to_csv('beta_5y.csv')

# 修改索引名称，否则会出现index与column混淆的情况
mktcap.index.name = 'date1' 


def beta_calculator_sw(data, factor):
    '''
    用来计算beta_sw的表格函数，输出是beta_sw计算方式的beta的表格。
    
    输入参数
    ----------
    data是有每只股票每月日度收益和市场超额收益信息的表格，变量命名month_num,code,rt,mkt
    span是每次回归跨度月份数，一年为12
    low_limit是计算beta的最低样本数（天数），一个月为15，三个月为40等
    
    输出
    -------
    index为股票代码，columns为月份编号，value为对应规则算出beta 的df
    '''
    # 创建一个空的DataFrame来存储beta值
    X = pd.DataFrame()

    # 计算市场因子的自相关系数
    rou = pearsonr(factor['mkt'][1:], factor['mkt'][:-1])[0]

    # 为市场超额收益创建不同的时间偏移量
    mktcap1 = mktcap2 = mktcap3 = factor[['date', 'mkt']]
    mktcap1['date'] = mktcap1['date'] + dt.timedelta(days=1)
    mktcap3['date'] = mktcap3['date'] + dt.timedelta(days=-1)

    # 对每个时间窗口进行循环
    for i in range(1, max(data['month_num']) - 12):
        same_time_data = data[(data['month_num'] > i) & (data['month_num'] <= i + 12)]
        same_time = []
        code_list = list(set(same_time_data['code']))

        # 对每只股票分别计算beta值
        for code in code_list:
            temp_data = same_time_data[same_time_data['code'] == code]
            reg_data1 = pd.merge(temp_data, mktcap1, on='date')
            reg_data2 = pd.merge(temp_data, mktcap2, on='date')
            reg_data3 = pd.merge(temp_data, mktcap3, on='date')

            reg_data1 = reg_data1[['rt', 'mkt']]
            reg_data2 = reg_data2[['rt', 'mkt']]
            reg_data3 = reg_data3[['rt', 'mkt']]

            # 检查数据是否足够用来回归
            if reg_data2['rt'].notna().sum() >= 200:
                model1 = smf.ols('rt~mkt', reg_data1, missing='drop').fit()
                model2 = smf.ols('rt~mkt', reg_data2, missing='drop').fit()
                model3 = smf.ols('rt~mkt', reg_data3, missing='drop').fit()

                beta1 = model1.params[1]
                beta2 = model2.params[1]
                beta3 = model3.params[1]

                beta = (beta1 + beta2 + beta3) / (1 + 2 * rou)
            else:
                beta = np.nan

            same_time.append(beta)

        # 将beta值按股票代码索引并加入结果
        same_time = pd.Series(same_time, index=code_list, name=i + 12)
        X = pd.concat([X, same_time], axis=1)

    return X
full_data1 = daily_data[['code','rt','date','month_num']]
beta_sw = beta_calculator_sw(full_data1,mktcap)
# beta_sw.columns = beta_sw.columns+1
beta_sw.to_csv('beta_sw.csv')


# mkt只保留mkt_rf，用于beta_d的回归计算
mktcap1 = mktcap[['mkt_rf']]
def beta_calculator_d(data, factor):
    '''
    用来计算beta_d的表格函数，输出是beta_d计算方式的beta的表格。
    
    输入参数
    ----------
    data是以month_num为columns，code为index，rt为value
    span是每次回归跨度月份数，一年为12
    low_limit是计算beta的最低样本数（天数），一个月为15，三个月为40等
    
    输出
    -------
    index为股票代码，columns为月份编号，value为对应规则算出beta 的df
    '''
    X = pd.DataFrame()
    mkt = pd.DataFrame()

    for k in range(6):
        x1 = factor.shift(k)
        x2 = factor.shift(-k)
        if k == 0:
            mkt = pd.concat([mkt, x1], axis=1)
        else:
            mkt = pd.concat([mkt, x1], axis=1)
            mkt = pd.concat([mkt, x2], axis=1)

    mkt.columns = ['mkt1', 'mkt2', 'mkt3', 'mkt4', 'mkt5', 'mkt6', 'mkt7', 'mkt8', 'mkt9', 'mkt10', 'mkt11']
    mkt = mkt.dropna()

    for i in range(1, max(data['month_num']) - 12):
        same_time_data = data[(data['month_num'] > i) & (data['month_num'] <= i + 12)]
        same_time = []
        code_list = same_time_data.columns[:-1]

        for code in code_list: #按照股票代码进行循环，计算每只股票的beta
            temp_data = same_time_data[code]
            temp_data.name = 'rt'
            reg_data = pd.concat([temp_data, mkt], axis=1, join='inner')

            if reg_data['rt'].notna().sum() >= 200:
                model = smf.ols('rt ~ mkt6 + mkt5 + mkt7 + mkt4 + mkt8 + mkt3 + mkt9 + mkt2 + mkt10 + mkt1 + mkt11', reg_data, missing='drop').fit()
                beta = sum(model.params[1:])
            else:
                beta = np.nan

            same_time.append(beta)

        same_time = pd.Series(same_time, index=code_list, name=i + 12)
        X = pd.concat([X, same_time], axis=1)
        print(i) #根据循环显示进度

    return X

beta_d = beta_calculator_d(full_data,mktcap1)
beta_d.to_csv('beta_d.csv')


# 计算描述性统计并生成表格
## 导入之前计算好的数据
beta_1m = pd.read_csv(os.path.join('beta_1m.csv'),index_col=0)
beta_3m = pd.read_csv(os.path.join('beta_3m.csv'),index_col=0)
beta_6m = pd.read_csv(os.path.join('beta_6m.csv'),index_col=0)
beta_12m = pd.read_csv(os.path.join('beta_12m.csv'),index_col=0)
beta_24m = pd.read_csv(os.path.join('beta_24m.csv'),index_col=0)
beta_1y = pd.read_csv(os.path.join('beta_1y.csv'),index_col=0)
beta_2y = pd.read_csv(os.path.join('beta_2y.csv'),index_col=0)
beta_3y = pd.read_csv(os.path.join('beta_3y.csv'),index_col=0)
beta_5y = pd.read_csv(os.path.join('beta_5y.csv'),index_col=0)
beta_sw = pd.read_csv(os.path.join('beta_sw.csv'),index_col=0)
beta_d = pd.read_csv(os.path.join('beta_d.csv'),index_col=0)

# 定义计算描述性统计的函数
def beta_statistic(list_of_beta,name_of_beta):
    X = pd.DataFrame()
    for i in range(len(list_of_beta)):
        x = list_of_beta[i]
        new = pd.Series([x.mean().mean(),x.std().mean(),x.skew().mean(),x.kurt().mean(),x.min().mean(),x.quantile(.05).mean(),x.quantile(.25).mean(),x.median().mean(),x.quantile(.75).mean(),x.quantile(.95).mean(),x.max().mean(),x.count().mean()],
                         index = ['Mean','SD','Skew','Kurt','Min','5%','25%','Median','75%','95%','Max','n'],name = name_of_beta[i])
        X = pd.concat([X,new],axis=1)
    X = X.T
    X = X.applymap(lambda x:round(x, 2))
    return X

df_list = [beta_1m,beta_3m,beta_6m,beta_12m,beta_24m,beta_1y,beta_2y,beta_3y,beta_5y,beta_sw,beta_d]

def drop():
    beta = []
    for i in df_list:
        if i.columns[-1] == '276':
            i = i.drop('276',axis = 1)
        beta.append(i)
    return beta

beta_list = drop()

beta_name_list = ['beta_1m','beta_3m','beta_6m','beta_12m','beta_24m','beta_1y','beta_2y','beta_3y','beta_5y','beta_sw','beta_d']
table1 = beta_statistic(beta_list,beta_name_list)
table1


# 相关性分析
## 定义计算person相关性的函数
def personcorr_calculator(dataname1,dataname2):
    X = []
    if len(dataname1.columns)>=len(dataname2.columns):
        month_list = dataname2.columns
    else:
        month_list = dataname1.columns
    for y in month_list:
        x1 = dataname1[y]
        x2 = dataname2[y]
        x = pd.concat([x1,x2],axis=1)
        x = x.dropna(axis=0,how='any')
        person_corr = x.corr('pearson').iloc[0,1]
        X.append(person_corr)
    X = pd.Series(X)
    x = X.mean()
    return x

## 定义计算spearman相关性的函数
def spearman_calculator(dataname1,dataname2):
    X = []
    if len(dataname1.columns)>=len(dataname2.columns):
        month_list = dataname2.columns
    else:
        month_list = dataname1.columns
    for y in month_list:
        x1 = dataname1[y]
        x2 = dataname2[y]
        x = pd.concat([x1,x2],axis=1)
        x = x.dropna(axis=0,how='any')
        spearman_corr = x.corr(method = 'spearman').iloc[0,1]
        X.append(spearman_corr)
    X = pd.Series(X)
    x = X.mean()
    return x

def beta_in_list(list_of_beta,name_of_beta):
    ##beta的list顺序和名字顺序要完全对应
    X = pd.DataFrame([],index = name_of_beta,columns = name_of_beta)
    for i in range(len(list_of_beta)):
        for j in range(len(list_of_beta)):
            if i<=j:
                X.iloc[i,j] = spearman_calculator(list_of_beta[i],list_of_beta[j])
            else:
                X.iloc[i,j] = personcorr_calculator(list_of_beta[i],list_of_beta[j])
    X = X.applymap(lambda x:round(x, 2))
    return X

table2 = beta_in_list(beta_list,beta_name_list)            
table2


## 持续性分析
def Persistence_calculator(df):
    temp = winsorize(df,limits=(0.005, 0.005))
    temp1 = pd.DataFrame(temp)
    corr = temp1.corr()
    delay_list = [1,3,6,12,24,36,48,60,120]
    X = pd.DataFrame([],index = df.columns,columns = delay_list)
    for x in range(len(df.columns)):
        for y in range(9):
            if x+delay_list[y] < df.shape[1]:
                X.iloc[x,y] = corr.iloc[x,x+delay_list[y]]
    stats = X.mean()
    return stats

def beta_autocorr(list_of_beta,name_of_beta):
    X = pd.DataFrame()
    for i in list_of_beta:
        x = Persistence_calculator(i)
        X = pd.concat([X,x],axis=1)
    del_list = [1,2,3,4,3,4,5,7,3,3]
    for j in range(10):
        k = del_list[j]
        X.iloc[:k,j+1] = np.nan
    X.columns = name_of_beta
    X = X.applymap(lambda x:round(x, 2))
    return X
      
table3 = beta_autocorr(beta_list,beta_name_list)
table3


# 等权重组合分析
## 导入数据
monthly_rt = data('Trdmnt','Stkcd','rt').T
monthly_mkt = data('Trdmnt','Stkcd','Msmvttl').T  
monthly_rt.columns = beta_1m.columns
monthly_mkt.columns = beta_1m.columns

ff1 = ff4[['year','month','mkt_rf']]
ff1['month_num'] = (ff1['year']-1997)*12+ff1['month']
ff1 = ff1[['month_num','mkt_rf']].set_index('month_num')


def capm_reg_equal(beta,rt,mkt_rf):
    mkt_rf.index = rt.columns
    beta_list = pd.DataFrame()
    rt_list = pd.DataFrame()
    for i in beta.columns:
        temp_beta = beta[i]
        temp_rt = rt[i]
        x = pd.concat([temp_beta,temp_rt],axis=1)
        x.columns = ['beta','rt']
        x['group'] = pd.qcut(temp_beta,10,labels=False)
        x = x.dropna()
        beta_avg_i = x.groupby('group')['beta'].mean()
        rt_avg_i = x.groupby('group')['rt'].mean()
        beta_list = pd.concat([beta_list,beta_avg_i],axis=1)
        rt_list = pd.concat([rt_list,rt_avg_i],axis=1)
    beta_list.columns = beta.columns
    rt_list.columns = beta.columns
    beta_list = beta_list.T
    rt_list = rt_list.T
    beta_list.columns = range(1,11)
    rt_list.columns = range(1,11)
    beta_list['10-1'] = beta_list[10] - beta_list[1]
    rt_list['10-1'] = rt_list[10] - rt_list[1]
    alpha = []
    alpha_t = []
    for j in rt_list.columns:
        reg_list = pd.concat([rt_list[j],mkt_rf],axis=1,join='inner')
        reg_list.columns = ['rt','mkt_rf']
        reg_list = reg_list.dropna()
        model = smf.ols('rt~mkt_rf',reg_list).fit(cov_type='HAC',cov_kwds={'maxlags':6})
        alpha.append(model.params[0])
        alpha_t.append(model.tvalues[0])
    beta_avg = beta_list.mean()
    rt_avg = rt_list.mean()
    alpha = pd.Series(alpha,index = beta_avg.index)
    alpha_t = pd.Series(alpha_t,index = beta_avg.index)
    X = pd.concat([beta_avg,rt_avg,alpha,alpha_t],axis=1).T
    X.index = ['beta','excess return','alpha','alpha_t']
    return X

def table4(list_of_beta,rt,mkt_rf):
    X = pd.DataFrame()
    for i in list_of_beta:
        x = capm_reg_equal(i,rt,mkt_rf)
        X = pd.concat([X,x])
    X = X.applymap(lambda x:round(x, 2))
    return X

table4 = table4(beta_list,monthly_rt,ff1)  
table4


## 市值加权
def capm_reg_mktweight(beta,rt,mkt,mkt_rf):
    mkt_rf.index = rt.columns
    beta_list = pd.DataFrame()
    rt_list = pd.DataFrame()
    for i in beta.columns:
        temp_beta = beta[i]
        temp_rt = rt[i]
        temp_mkt = mkt[i]
        x = pd.concat([temp_beta,temp_rt,temp_mkt],axis=1)
        x.columns = ['beta','rt','mktcap']
        x['rt*mkt'] = x['rt']*x['mktcap']
        x = x.dropna()
        x['group'] = pd.qcut(temp_beta,10,labels=False)
        beta_avg_i = x.groupby('group')['beta'].mean()
        rt_avg_i = x.groupby('group')['rt*mkt'].sum()/x.groupby('group')['mktcap'].sum()
        beta_list = pd.concat([beta_list,beta_avg_i],axis=1)
        rt_list = pd.concat([rt_list,rt_avg_i],axis=1)
    beta_list.columns = beta.columns
    rt_list.columns = beta.columns
    beta_list = beta_list.T
    rt_list = rt_list.T
    beta_list.columns = range(1,11)
    rt_list.columns = range(1,11)
    beta_list['10-1'] = beta_list[10] - beta_list[1]
    rt_list['10-1'] = rt_list[10] - rt_list[1]
    alpha = []
    alpha_t = []
    for j in rt_list.columns:
        reg_list = pd.concat([rt_list[j],mkt_rf],axis=1,join='inner')
        reg_list.columns = ['rt','mkt']
        model = smf.ols('rt~mkt',reg_list).fit(cov_type='HAC',cov_kwds={'maxlags':6})
        alpha.append(model.params[0])
        alpha_t.append(model.tvalues[0])
    beta_avg = beta_list.mean()
    rt_avg = rt_list.mean()
    alpha = pd.Series(alpha,index = beta_avg.index)
    alpha_t = pd.Series(alpha_t,index = beta_avg.index)
    X = pd.concat([beta_avg,rt_avg,alpha,alpha_t],axis=1).T
    X.index = ['beta','excess return','alpha','alpha_t']
    return X     

capm_reg_mktweight(beta_12m,monthly_rt,monthly_mkt,ff1)

def table5(list_of_beta,rt,mkt,mkt_rf):
    X = pd.DataFrame()
    for i in list_of_beta:
        x = capm_reg_mktweight(i,rt,mkt,mkt_rf)
        X = pd.concat([X,x])
    X = X.applymap(lambda x:round(x, 2))
    return X   

table5 = table5(beta_list,monthly_rt,monthly_mkt,ff1)
table5


## Fama-Macbeth回归
def FM_regression1(df1,df2):
    coefs = []
    adj_R = []
    number = []

    for i in df2.columns:       
        temp_beta = df2[i]
        temp_rt = df1[i]
        x = pd.concat([temp_beta,temp_rt],axis=1)
        x.columns = ['beta','rt']
        temp = x.dropna()
        number.append(len(temp)) #样本量
        temp['beta'] = winsorize(temp['beta'], limits=(0.005, 0.005))
        Y = temp['rt']
        X = temp['beta']
        model = sm.OLS(Y.values,sm.add_constant(X).values).fit(cov_type='HAC', cov_kwds={'maxlags': 6})
        coefs.append(model.params)
        adj_R.append(model.rsquared_adj)
    result = pd.DataFrame(coefs)
    result['adj_R'] = adj_R
    result['n'] = number
    
    return result

## NW滞后6期调整
def NWtest_1sample(a, lags=6):
    adj_a = np.array(a)
    # 对常数回归
    model = sm.OLS(adj_a, [1] * len(adj_a)).fit(cov_type='HAC', cov_kwds={'maxlags': lags})

    return adj_a.mean(), float(model.tvalues)


## 生成FM回归分析表
def table6(beta,name):
    temp = pd.DataFrame()
    for i in beta_list:
        data = FM_regression1(reg,i)
        value1 = data.iloc[:, :-2].apply(NWtest_1sample)
        value1 = np.array([list(x) for x in value1.values]).reshape(-1)
        
        value2 = data.iloc[:, -2:].mean().values
        value = pd.Series(list(value1) + list(value2))
        temp = pd.concat([temp,value],axis = 1) 
    temp.index = ['intercept','','beta','','Adj_R2','n']
    temp.columns = name
    return temp

table6 = table6(beta_list,beta_name_list)
table61.iloc[:-1,:] = table6[:-1,:].apply(lambda x:round(x, 3))
table61.iloc[-1,:] = table61.iloc[-1,:].apply(lambda x: int(x)) 
table61