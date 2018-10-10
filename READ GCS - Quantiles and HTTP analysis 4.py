
#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May 15 10:56:48 2018

@author: swe03
"""



# In[3]:


import numpy as np
import pandas as pd

import pandas.io.gbq as pdg

from pandasql import PandaSQL 
pdsql = PandaSQL()

from datetime import datetime, timedelta

desired_width = 500
pd.set_option('display.width',desired_width)
pd.set_option('display.max_rows', 500)


#%%
## Concatenate a set of csv files into one file

unq_rel_cnts = pd.DataFrame()

for i in range(12,26):  ## This range is 3.12 to 3.25.  The 26th is there as the non-inclusive end boundary
    s = str(i)
    urc_single_file  = pd.DataFrame()
    
    fileString = '/Users/swe03/RawData/urc_3_'+ s +'.csv' 
   
    urc_single_file = pd.read_csv(fileString,sep=',')
    
    unq_rel_cnts = pd.concat([unq_rel_cnts, urc_single_file],ignore_index=True)
    
#%%
unq_rel_cnts.sort_values(by='distinct_freq', ascending=False, inplace=True)

#%%
## method='first' uses the sorted order from the dstAd_cnt above to assign a sequential nbr.
## That is, a set number of unique dstAd's and, for each unique dstAd, the Count of the number of times it was contacted from the src_ip
unq_rel_cnts['rank'] = unq_rel_cnts['distinct_freq'].rank(method='first') 
unq_rel_cnts[0:10]


# In[126]:

## Create the arbitrary quantiles(i.e., qcut).  Use .values,10 for Deciles
## This will create quantiles for the dstAd Counts(i.e., dstAd_cnt) associated with each unique dstAd
## NOTE:  using the labels array will replace the bin ranges for the set of single bin int value(e.g., 1, 2, 3, .... 10 for deciles)
unq_rel_cnts['rank_int'] = pd.qcut(unq_rel_cnts['rank'].values,10,labels=[1,2,3,4,5,6,7,8,9,10])
unq_rel_cnts[0:10]
unq_rel_cnts.info()

# In[128]:

## Need to convert the rank_int from data type Category to String
unq_rel_cnts['rank_int'] = unq_rel_cnts.rank_int.astype(int) 
unq_rel_cnts['rank'] = unq_rel_cnts['rank'].astype(int)
unq_rel_cnts.info()
unq_rel_cnts[5000:5010]

#%%
## Count is the number of Unique dstAd's IN EACH INTERVAL and, thus, Count column will SUM UP 
## to the largest rank value which is the largest rank
## The Univariate Statistics(e.g., mean, std, etc.) describes the set of the dstAd_cnt's within each quantile.
## 
unq_rel_cnts.groupby('rank_int')['distinct_freq'].describe()

#unq_rel_cnts.info()

#%%
## Test
def urc():
    unq_rel_cnts = unq_rel_cnts.loc[(unq_rel_cnts['rank_int']==8) &  
                                    (unq_rel_cnts['date_hour']=='2018-03-12 04')]
    #unq_rel_cnts.info()
    view_hour = unq_rel_cnts.groupby(['date_hour'])['distinct_freq'].agg('sum')
    view_hour = view_hour.to_frame()
    view_hour.rename(columns={"distinct_freq":"distinct_freq_sum"},inplace=True)
    view_hour.reset_index(inplace=True)
    view_hour.info()
    return(view_hour)
    
#view_hour = urc()
    
#%%
## SQL test
def vh_test():
    view_hour = pdsql("""Select date_hour
                  from unq_rel_cnts
                  where rank_int = 8
                  and   date_hour = '2018-03-12 04' 
                  """, locals())
    return(view_hour)
    
#view_hour = vh_test()
    
# In[141]:
##  

view_hour = pdsql("""Select date_hour, sum(distinct_freq) as distinct_freq_sum
              from unq_rel_cnts
              where rank_int == 8
              and substr(date_hour,1,10) >= '2018-03-12' 
              group by date_hour
              order by date_hour""", locals()) 
view_hour.info()

#%%

import matplotlib.pyplot as pplt

from matplotlib.pylab import rcParams

## the first param (i.e., width) was 15 and the x-axis was unreadable
rcParams['figure.figsize'] = 75, 6

from datetime import datetime, timedelta

desired_width = 250
pd.set_option('display.width',desired_width)
pd.set_option('display.max_rows', 500)

pplt.plot(view_hour['distinct_freq_sum'] , c = 'b', label = 'count')

pplt.xticks(range(len(view_hour['date_hour'])),view_hour['date_hour'])

pplt.xticks(rotation=80)
pplt.title('Rank Interval = 5, dstAd_cnt_sums for 24 hours')

pplt.show()

#%%
import statsmodels.tsa as tsa

from statsmodels.tsa.stattools import ccf 
from statsmodels.graphics.api import qqplot
from statsmodels.tsa.base import datetools
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.arima_model import ARMA

from statsmodels.tsa.stattools import acf  
from statsmodels.tsa.stattools import pacf
from statsmodels.tsa.seasonal import seasonal_decompose

import statsmodels.api as sm  

from scipy.stats.stats import pearsonr
from scipy import stats
from sklearn import linear_model

from sklearn.metrics import mean_squared_error
from math import sqrt


#%%
## view_hour['datetime'].dtype displays dtype('<M8[ns]') but info() displays datetime64[ns]
view_hour['datetime'] = pd.to_datetime(view_hour['date_hour'])   
view_hour.info()
view_hour.head(5)

#%%
view_hour.reset_index(inplace=True)
view_hour = view_hour.set_index('datetime')
view_hour.sort_index(inplace=True)
view_hour.info()

#%%
## Write out the csv file to the local directory
#view_hour.to_csv('/Users/swe03/view_hour.txt', index=True)

#%%
view_hour.describe()

#%%

decomposition = seasonal_decompose(view_hour['distinct_freq_sum'].values,freq=24 )  
  
fig = decomposition.plot()  
fig.set_size_inches(50, 8)

#%%
# Graph Autocorrelation and Partial Autocorrelation data
fig, axes = pplt.subplots(1, 2, figsize=(15,4))

fig = sm.graphics.tsa.plot_acf(view_hour['distinct_freq_sum'], lags=24, ax=axes[0])
fig = sm.graphics.tsa.plot_pacf(view_hour['distinct_freq_sum'], lags=24, ax=axes[1])

#%%
## Specify the SARIMAX model
## Default for the CI is 95%.  Set in the Alpha parameter for conf_int function

def sarimax(): 
    mod = sm.tsa.statespace.SARIMAX(view_hour['distinct_freq_sum'], order=(4,0,0),seasonal_order=(3,1,0,24))
    #mod1 = sm.tsa.statespace.SARIMAX(ent_calc_df['In_degree_entropy'], order=(8,0,0), seasonal_order=(2,0,0,6))
    #mod2 = sm.tsa.statespace.SARIMAX(ent_calc_df['In_degree_entropy'], trend='n', order=(1,0,0), seasonal_order=(2,0,0,24))
    results = mod.fit()
    print(results.summary()) 
    return (results,mod)

#results, mod = sarimax()
    

#%%
endog1 = view_hour['distinct_freq_sum']
endog2 = np.log(view_hour['distinct_freq_sum'])

def uncomp1():
    mod = sm.tsa.UnobservedComponents(endog1, 'llevel', freq_seasonal=[
    {'period':24, 'harmonics': 6}, {'period': 24 * 7, 'harmonics': 6}], ar=5)
    results = mod.fit(maxiter=100)
    results = mod.fit(results.params, method='nm', maxiter=1000)
    return (results,mod)


def uncomp2():
    mod = sm.tsa.UnobservedComponents(endog2, 'llevel', freq_seasonal=[{'period':24, 'harmonics': 6}, {'period': 24 * 7, 'harmonics': 6}])
    results = mod.fit(maxiter=100)
    results = mod.fit(results.params, method='nm', maxiter=1000)
    return (results,mod)





results,mod = uncomp2()

#%%
print(results.summary())


#%%c
endog2.plot()
#endog1.plot()

#%%
fig = results.plot_diagnostics()

#%%
results.resid.values

#%%
## Check to see if Residuals are still correlated
## Values are between 0 and 4.  Values close to 2: no Serial Correlation. Close to 0: Pos Corr. Close to 4: Neg Corr.
sm.stats.durbin_watson(results.resid.values)

#%%
## Check to see if the distribution of Residuals are Normally Distributed (i.e., This is desirable)
## Null Ho is that the data is associated with a Normal Distribution
##If the p-val is very small, it means it is unlikely that the data came from a normal distribution
resid1 = results.resid
stats.normaltest(resid1)

#%%
fig = pplt.figure(figsize=(12,8))
ax = fig.add_subplot(111)
fig = qqplot(resid1, line='q', ax=ax, fit=True)

#%%
# Graph the acf and pacf for the Residuals (Stacked)
#fig = pplt.figure(figsize=(12,8))
#ax1 = fig.add_subplot(211)
#fig = sm.graphics.tsa.plot_acf(resid1.values.squeeze(), lags=24, ax=ax1)
#ax2 = fig.add_subplot(212)
#fig = sm.graphics.tsa.plot_pacf(resid1, lags=24, ax=ax2)

#%%
# # Graph the acf and pacf for the Residuals (Side by Side)
fig, axes = pplt.subplots(1, 2, figsize=(15,4))
fig = sm.graphics.tsa.plot_acf(resid1, lags=24, ax=axes[0])
fig = sm.graphics.tsa.plot_pacf(resid1, lags=24, ax=axes[1])

#%%
results.plot_diagnostics(figsize=(13, 8));

#%%
results.plot_components(figsize=(20, 30))


#%%
## This will produce 10 sequential exponentiated forecast values
## Forecast will create the requested number of values for the future
## Predict will, regardless of requested number, list all model predicted values
## Conf Int will list each upper and lower c.i. for each model component

print("Forecast: ", np.exp(results.forecast(10)))
print("Predict: ", np.exp(results.predict()))
print("Conf Int: ", np.exp(results.conf_int()))
#results.forecast(5)

#%%
## This exponentiation worked
## Without a value in the .forecast() the returned value will only be the first forecasted value
results_o = np.exp(results.forecast())
## The forecast function will just return a series and, thus, no attribute for plot_components. Thus, the plot will fail.
#results_o.plot_components(figsize=(13, 20));

## As stated above, with .forecast() the value of results_o is the first, single, forecasted value.
results_o

#%%
# Box-Pierce Q statistic tests the Null Ho that *ALL* correlations up to Lag K are equal to Zero.  
## Thus, the desired output is to accept the Null Ho (i.e., no correlations) indicated by high Prob(>Q) values
# This is not the same as the correlogram above.
r,q,p = sm.tsa.acf(resid1.values.squeeze(), qstat=True)
data = np.c_[range(1,41), r[1:], q, p]
table = pd.DataFrame(data, columns=['lag', "AC", "Q", "Prob(>Q)"])
print(table.set_index('lag'))
  
#%%
# This is just a SARIMAX Results Wrapper.  It collects the model parameters for later use.
res = mod.filter(results.params)

#%%
### In-sample prediction and out-of-sample forecasting
###########################
### The get_prediction function works the same as above res.predict() except that exogenous variables can be 
###   included as a parameter in this function as well as more statistical functions in the predict object model 
###   like predicted_mean and predicted_results
###########################
predict = res.get_prediction()
#predict.predicted_mean

#%%
## Create the Confidence Intervals for the entire range
## Note that using the full range will create initial C.I.'s that are large (i.e., not enough history for tighter C.I.'s)
##   This will change the scale of the graph and is not useful for visualization below
predict_ci = predict.conf_int()

## This gives a warning but still completes
## Warning is associated with the initial 0 to INF CI values
ci = np.exp(predict_ci)  
ci.info()
float(ci['lower distinct_freq_sum'])  ## cannot convert the series to type float
#%%
## Create the Predicted Means for Graph #1 below
## The Predicted Means are both In-Sample Predictions as well as Out-Of-Sample Forecast
## Can just use the last datetime of the training sample and get the next forecasted hour but any range with the training data will give the 
##  same forecasted results
#predict_dy.predicted_mean.astype(int)
#type(predicted_mean_1)  ## Series
predicted_mean_1 = predict.predicted_mean.loc['2018-03-12 00:00:00':'2018-03-26 03:00:00']   # include 8-14 02:00 since there is a spike

predicted_mean = np.exp(predicted_mean_1)
predicted_mean
#%%
### Create the Confidence Intervals for Graph #1 below
### This is a constrained view based on dates and is used below in the graphing
### Note:  The Start date can be any date in the Training Set data.  Any date will apparently not change the predicted values and
###        and, here, the CI
predict = res.get_prediction(start='2018-03-12 00:00:00', end='2018-03-26 10:00:00')
predict_ci_1 = predict.conf_int()

#%%
# RMSE accuracy measure: the "standard deviation" of the error (i.e., expressed in the data units)
test = view_hour.loc[(view_hour['date_hour'] > '2018-03-12 00')] 
test1 = test.loc[(test['date_hour'] <= '2018-03-26 03')]
test2 = test1['distinct_freq_sum']
predict1 = predicted_mean_1[lambda x: x!= 0]  
mse = mean_squared_error(test2, predict1)
rmse = sqrt(mse)
print('RMSE: %.3f' % rmse)

#%%
fig, ax = pplt.subplots(figsize=(15, 5))

ax.plot(view_hour['distinct_freq_sum'])
#ax.plot(np.exp(results.forecast(5 * 7 * 24)));
ax.plot(np.exp(results.forecast(10)));

#%%

fig, ax = pplt.subplots(figsize=(40,8))
npre = 4
#ax.set(title='Count of Packets by Hour', xlabel='Hour', ylabel='Count')
fig.suptitle('Distinct Freq Sums by Hour', fontsize=30)
pplt.xlabel('', fontsize=28)   # The fontsize here is applied to the df variable name not the label in the function
pplt.ylabel('Frequency Sums', fontsize=26)

# Plot data points
view_hour.ix[:,'distinct_freq_sum'].plot(ax=ax, style='blue', label='Observed')  # 650 is '12.17.2016 02:00:00'

# Plot predictions
#predict.predicted_mean.ix[:].plot(ax=ax, style='r', label='Dynamic forecast', fontsize=30)
#ci = predict_ci_1.ix[1:]

predicted_mean.ix[:].plot(ax=ax, style='r', label='Dynamic forecast', fontsize=30)


#ax.fill_between(ci.index, ci.ix[:,0], ci.ix[:,1], color='black', alpha=0.2)
#ax.fill_between(ci.index, ci.ix[8:,0], ci.ix[8:,1], color='black', alpha=0.2)

legend = ax.legend(loc='lower right', fontsize=30)

#%%
view_hour
predicted_mean
ci
