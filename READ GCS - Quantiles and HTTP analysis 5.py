
#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May 15 10:56:48 2018

@author: swe03
"""

#!pip install pandasql

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
    
    fileString = '/home/steve/Data/RawData/urc_3_'+ s +'.csv' 
   
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

unq_rel_cnts.info()

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

from fbprophet import Prophet


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
## Seasonal Decomposition
def diagnostics():
    decomposition = seasonal_decompose(view_hour['distinct_freq_sum'].values,freq=24 )  
      
    fig = decomposition.plot()  
    fig.set_size_inches(50, 8)
#diagnostics()
#%%
## Graph Autocorrelation and Partial Autocorrelation data
def acf_pacf():
    fig, axes = pplt.subplots(1, 2, figsize=(15,4))
    
    fig = sm.graphics.tsa.plot_acf(view_hour['distinct_freq_sum'], lags=24, ax=axes[0])
    fig = sm.graphics.tsa.plot_pacf(view_hour['distinct_freq_sum'], lags=24, ax=axes[1])
##acf_pacf()
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

#results,mod = uncomp2()


### Beginning of Prophet section
#%%
view_hour['y'] = np.log(view_hour['distinct_freq_sum'])
view_hour['ds'] = view_hour['date_hour']
view_hour.head(5)

#%%
m = Prophet()
m.add_seasonality(name='hourly', period=24, fourier_order=2)
m.fit(view_hour);

#%%
## Create a dataframe for the future dates
future = m.make_future_dataframe(periods=24,freq='H')
future.tail()

#%%
forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

#%%
## Note that the predict function will create a df that contains
##   many period features(e.g., trend, daily, hourly, weekly, seasonal
##   along with _upper and _lower ci's). Execute a .info() against
##   the dataframe to see all the elements.
## This creates a dataframe with just the 4 elements below
forecast1 = m.predict(future)
forecast1[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail() 

#%%
## This works !
print(np.exp(forecast['yhat'][0:5]))
print(forecast['yhat_lower'][0:5])
print(forecast['yhat_upper'][0:5])

#%%
## This works and retained the dataframe type
forecast2 = np.exp(forecast1[['yhat','yhat_lower','yhat_upper']])

## Now merge to bring the ds back into the df
## Without the "on" keyword the join key is implicitly the index
forecast2 = forecast2.join(forecast1['ds'], how='inner')

print(forecast1)

#%%
## This works
## This will create a plot that includes Forecasted, C.I.'s, and Actual values
m.plot(forecast2)

#%%
## I think it is unecessary to review exponentiated components 
## Plus the complexity of joining forecast2 with forecast1
m.plot_components(forecast1);

#%%
## It was necessary, in the fill_between, to use a datetime index associated with 
## the first parameter of the function.
## This necessitated converting the existing ds datetime element to an index
pplt.subplots(figsize=(30,10))
forecast2.set_index('ds',inplace=True)

pplt.plot(view_hour['distinct_freq_sum'], label='Original', color='black');
pplt.plot(forecast2.yhat, color='red', label='Forecast');
pplt.fill_between(forecast2.index, forecast2['yhat_upper'], forecast2['yhat_lower'], color='gray', alpha=0.25)
pplt.ylabel('Distinct Freq Sums');
pplt.xlabel('Hours');
#pplt.savefig('./img/prophet_forecast.png')
pplt.show()


#%%
# RMSE accuracy measure: the "standard deviation" of the error (i.e., expressed in the data units)

actual = view_hour['distinct_freq_sum']
predicted = forecast2[:340]['yhat']  
mse = mean_squared_error(predicted, actual)
rmse = sqrt(mse)
print('RMSE: %.3f' % rmse)

#%%
## These were just some descriptive views into the various dataframes

view_hour.info()
forecast2.info()
view_hour[335:]
forecast2[:].yhat_upper
forecast2[:].yhat_lower
#%%
fig, ax = pplt.subplots(figsize=(20,8))
npre = 4
#ax.set(title='Count of Packets by Hour', xlabel='Hour', ylabel='Count')
fig.suptitle('Distinct Freq Sums by Hour', fontsize=30)
pplt.xlabel('', fontsize=28)   # The fontsize here is applied to the df variable name not the label in the function
pplt.ylabel('Frequency Sums', fontsize=26)

# Plot data points
view_hour.ix[:,'distinct_freq_sum'].plot(ax=ax, style='blue', label='Observed')  # 650 is '12.17.2016 02:00:00'

# Plot predictions
forecast2.yhat.plot( style='r', label='Dynamic forecast', fontsize=30)
ci = forecast2[['yhat_lower','yhat_upper']]

ax.fill_between(forecast2.index, ci['yhat_lower'],ci['yhat_upper'],  color='black', alpha=0.2)

legend = ax.legend(loc='best', fontsize=15)

#%%
help(Prophet)
help(Prophet.fit)








##-------------------------------------------------------------------
## Beginning of UCM / Sarimax subsequent processing section
##-------------------------------------------------------------------

#%%
print(results.summary())

#%%c
#endog2.plot()
endog1.plot()

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
np.exp(results.forecast(10))
np.exp(results.predict(10))

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
## This is just a SARIMAX / UCM Results Wrapper.  
## It collects the model parameters for later use.
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

#ci = np.exp(predict_ci)  
#ci.info()
#float(ci['lower distinct_freq_sum'])  ## cannot convert the series to type float
#%%
## Create the Predicted Means for Graph #1 below
## The Predicted Means are both In-Sample Predictions as well as Out-Of-Sample Forecast
## Can just use the last datetime of the training sample and get the next forecasted hour but any range with the training data will give the 
##  same forecasted results
#predict_dy.predicted_mean.astype(int)
#type(predicted_mean_1)  ## Series
predicted_mean_1 = predict.predicted_mean.loc['2018-03-12 00:00:00':'2018-03-26 03:00:00']   # include 8-14 02:00 since there is a spike

predicted_mean = np.exp(predicted_mean_1)
predicted_mean[50:55]
#%%
### Create the Confidence Intervals for Graph #1 below
### This is a constrained view based on dates and is used below in the graphing
### Note:  The Start date can be any date in the Training Set data.  Any date will apparently not change the predicted values and
###        and, here, the CI

predict = res.get_prediction(start='2018-03-12 00:00:00', end='2018-03-26 03:00:00')
predict_ci_1 = predict.conf_int()
#predicted_mean_1 = predict.predicted_mean

print(predict_ci_1[0:50])
print(predicted_mean_1[0:50])

#%%
ci_exp1 = np.exp(predict_ci_1)
ci_exp1[50:55]
#%%
# RMSE accuracy measure: the "standard deviation" of the error (i.e., expressed in the data units)
#test = view_hour.loc[(view_hour['date_hour'] >= '2018-03-12 00')] 
#test1 = test.loc[(test['date_hour'] <= '2018-03-26 03')]
#test2 = test1['distinct_freq_sum']
#predict1 = predicted_mean   ##[lambda x: x!= 0]  
actual = view_hour['distinct_freq_sum']
actual0 = actual.loc[actual.index >= '2018-03-13 00:00:00']
predict = predicted_mean.astype(int)
predict0 = predict.loc[predict.index >= '2018-03-13 00:00:00']
mse = mean_squared_error(actual0, predict0)
rmse = sqrt(mse)
print('RMSE: %.3f' % rmse)

#%%
test2
predict1.info()
#%%
## This works
fig, ax = pplt.subplots(figsize=(15, 5))

ax.plot(view_hour['distinct_freq_sum'])
#ax.plot(np.exp(results.forecast(5 * 7 * 24)));
ax.plot(np.exp(results.forecast(10)));

#%%

fig, ax = pplt.subplots(figsize=(40,18))
npre = 4
#ax.set(title='Count of Packets by Hour', xlabel='Hour', ylabel='Count')
fig.suptitle('Distinct Freq Sums by Hour', fontsize=30)
pplt.xlabel('', fontsize=28)   # The fontsize here is applied to the df variable name not the label in the function
pplt.ylabel('Frequency Sums', fontsize=26)

# Plot data points
view_hour.ix[73:,'distinct_freq_sum'].plot(ax=ax, style='blue', label='Observed')  # 650 is '12.17.2016 02:00:00'

# Plot predictions
np.exp(predicted_mean_1.ix[73:]).plot(ax=ax, style='r', label='Dynamic forecast', fontsize=30)
ci = np.exp(predict_ci_1)

#predicted_mean.ix[:].plot(ax=ax, style='r', label='Dynamic forecast', fontsize=30)


ax.fill_between(ci.index[73:], ci.ix[73:,0], ci.ix[73:,1], color='black', alpha=0.2)
ax.fill_between(ci.index[73:], ci.ix[73:,0], ci.ix[73:,1], color='black', alpha=0.2)

legend = ax.legend(loc='upper right', fontsize=30)

#%%
print(predicted_mean_1[25:30])
print(ci.ix[10:20])
print(ci.index[10:20])
