
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

from google.cloud import storage
from google.cloud import bigquery

desired_width = 500
pd.set_option('display.width',desired_width)
pd.set_option('display.max_rows', 500)

#%%
## This works
def ReadFromBQ():
    client = bigquery.Client()
    query = (
        'SELECT Timestamp, eventtypeid, src_addr, src_port, dst_addr, dst_port \
        FROM `ipfix.ipfix_10_14_00` '
        'WHERE dst_port = "443" '
        'LIMIT 100')

    df = client.query(query).to_dataframe()
    
    return(df)
    
ReadFromBQ()

#%%
## This works
def ReadFromMWG_BQ1():
    client = bigquery.Client()
    query = (
        'SELECT * FROM `io1-datalake-views.security_pr.mcafee_wg` '
            'where srcAd = "10.32.146.49" '
            'and TIMESTAMP >= "2018-08-13 00:00:00" and TIMESTAMP < "2018-08-26 23:00:00" '
            'order by TIMESTAMP desc'
            '--and byteP2S > 500000000' 
            'LIMIT 100 ')

    df = client.query(query).to_dataframe()
    
    return(df)
    
ReadFromMWG_BQ1()

#%%
## This 
def ReadFromJuniper_BQ1():
    client = bigquery.Client()
    global juniper_df
    query = (
            'SELECT EXTRACT(HOUR from timestamp) as hour, destination_address, destination_port,\
                source_address, count(*) as dest_count \
            FROM `io1-datalake-views.security_pr.juniper_junos_firewall` '
            'where source_address = "165.130.144.82" '
            'and TIMESTAMP >= "2018-09-13 00:00:00" and TIMESTAMP < "2018-09-13 23:00:00" '
            'group by hour, source_address, destination_address, destination_port '
            'order by hour desc \
            LIMIT 100 ')

    juniper_df = client.query(query).to_dataframe()
    
    return()
    
ReadFromJuniper_BQ1()

#%%
## This works
def ReadFromGCS2(name):
    global mwg_data
    storage_client = storage.Client(project='network-sec-analytics')
    bucket = storage_client.get_bucket('swe-files')
    blob = bucket.get_blob(name)
    
    ## This creates a csv locally
    blob.download_to_filename('/home/steve/Data/RawData/'+ name +"'")
    
    
    ## Read a local file into a Dataframe
    mwg_data= pd.DataFrame() 
    mwg_data = pd.read_csv('/home/steve/Data/RawData/'+ name + "'",sep=',')  
    mwg_data.info()
    return()

ReadFromGCS2('mwg_26wks')

#%%
## This works
## Create a sample for auditing and then sample it to get a distribution of different hours
def audit():
    global p_audit
    p_audit=pd.DataFrame()
    p_audit = pd.read_csv('/home/steve/Data/RawData/5_10_23_2018',sep=',',nrows=50000)  
    pa_s = p_audit.sample(n=50, random_state=123)
    
    
    pa_s['date_hour'] = pa_s.datetime.str.slice(start=0,stop=13,step=None) 
    pa_s.sort_values(by=['date_hour','source_address','destination_address','destination_port','protocol_id'], ascending=False, inplace=True)
    
    unq_rel_cnts = pa_s.groupby(['date_hour','source_address','destination_address','destination_port','protocol_id']).size().reset_index(name='distinct_freq')
    unq_rel_cnts.sort_values(by=['date_hour','source_address','destination_address','destination_port','protocol_id'], ascending=True, inplace=True)
    return()
audit()

#%%
## Create a new variable date_hour as a TIMESTAMP and the sub string for just the date and hour
mwg_data['date_hour'] = mwg_data.timestamp.str.slice(start=0,stop=13,step=None) 
mwg_data.sort_values(by=['date_hour','srcAd','dstAd','connProto'], ascending=False, inplace=True)

## THIS IS A COUNT DISTINCT ON THE 5 TUPLES
unq_rel_cnts = mwg_data.groupby(['date_hour','srcAd','dstAd','connProto']).size().reset_index(name='distinct_freq')
unq_rel_cnts.sort_values(by=['date_hour','srcAd','dstAd','connProto'], ascending=True, inplace=True)

#%%
freq_mwg = pdsql("""Select  distinct_freq, count(distinct_freq) as unq_cnt
              from unq_rel_cnts
              group by distinct_freq
              order by distinct_freq desc""", locals()) 
freq_mwg



#%%
## method='first' uses the sorted order from the dstAd_cnt above to assign a sequential nbr.
## That is, a set number of unique dstAd's and, for each unique dstAd, the Count of the number of times it was contacted from the src_ip
unq_rel_cnts['rank'] = unq_rel_cnts['distinct_freq'].rank(method='first') 
unq_rel_cnts


#%%

## Create the arbitrary quantiles(i.e., qcut).  Use .values,10 for Deciles
## This will create quantiles for the dstAd Counts(i.e., dstAd_cnt) associated with each unique dstAd
## NOTE:  using the labels array will replace the bin ranges for the set of single bin int value(e.g., 1, 2, 3, .... 10 for deciles)
unq_rel_cnts['rank_interval'] = pd.qcut(unq_rel_cnts['rank'].values,10,labels=[1,2,3,4,5,6,7,8,9,10])
unq_rel_cnts[0:10]
unq_rel_cnts.info()

#%%

## Need to convert the rank_int from data type Category to String
unq_rel_cnts['rank_interval'] = unq_rel_cnts.rank_interval.astype(int) 
#unq_rel_cnts['rank'] = unq_rel_cnts['rank'].astype(int)
unq_rel_cnts.info()
unq_rel_cnts

#%%
## Count is the number of Unique dstAd's IN EACH INTERVAL and, thus, Count column will SUM UP 
## to the largest rank value which is the largest rank
## The Univariate Statistics(e.g., mean, std, etc.) describes the set of the dstAd_cnt's within each quantile.
## 
unq_rel_cnts.groupby('rank_interval')['distinct_freq'].describe()

#%%
unq_rel_cnts.sort_values(by=['date_hour','srcAd','dstAd','connProto'], ascending=False, inplace=True)

# In[141]:
##  

view_hour = pdsql("""Select date_hour, sum(distinct_freq) as distinct_freq_sum
              from unq_rel_cnts
              where rank_interval == 10
              group by date_hour
              order by date_hour""", locals()) 
view_hour.info()

#%%
view_hour


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

#######################################
### Beginning of Prophet section
#######################################
#%%
view_hour['y'] = np.log(view_hour['distinct_freq_sum'])
view_hour['ds'] = view_hour['date_hour']
view_hour.head(5)

#%%
## Prophet1
# set the uncertainty interval to 95% (the Prophet default is 80%)
m = Prophet()
m.add_seasonality(name='hourly', period=24, fourier_order=2)
m.fit(view_hour);


#%%
## Create a dataframe for the future dates
## The tail will only display the time periods without the forecasted values
future = m.make_future_dataframe(periods=24,freq='H')
future.tail()

#%%
## This is the data that is exponentiated below
forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

#%%
## This is the data that retains the log transform
## Note that the predict function will create a df that contains
##   many period features(e.g., trend, daily, hourly, weekly, seasonal
##   along with _upper and _lower ci's). Execute a .info() against
##   the dataframe to see all the elements.
## This creates a dataframe with just the 4 elements below
forecast1 = m.predict(future)
forecast1[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail() 


#%%
## This works !
## Expentiation of the Confidence Interval is invalid.
## For C.I. either keep everything as a log value (including the Hourly assessment)
##   or use the Delta Method. 
print(np.exp(forecast['yhat'][0:5]))
print(forecast['yhat_lower'][0:5])
print(forecast['yhat_upper'][0:5])


#print(forecast1[['ds','yhat','yhat_lower','yhat_upper']])

#%%
## This works and retained the dataframe type
forecast2 = np.exp(forecast1[['yhat','yhat_lower','yhat_upper']])

## Now merge to bring the ds back into the df
## Without the "on" keyword the join key is implicitly the index which is what we're doing here
forecast2 = forecast2.join(forecast1['ds'], how='inner')



#%%
## This works
## This will create a plot that includes Forecasted, C.I.'s, and Actual values
m.plot(forecast1)

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

## If using the view_hour data it will be REQUIRED to exponentiate the forecasts (i.e., forecast2)
pplt.plot(view_hour['distinct_freq_sum'], label='Original', color='black');

pplt.plot(forecast2.yhat, color='red', label='Forecast');
pplt.fill_between(forecast2.index, forecast2['yhat_upper'], forecast2['yhat_lower'], color='gray', alpha=0.25)
pplt.ylabel('Distinct Freq Sums');
pplt.xlabel('Hours');
#pplt.savefig('./img/prophet_forecast.png')
pplt.show()


#%%
# RMSE accuracy measure: the "standard deviation" of the error (i.e., expressed in the data units)

#actual = view_hour['distinct_freq_sum']  ## These are the raw input values
actual = view_hour['y']  ## These are the raw input values after the log transform
predicted = forecast1[:340]['yhat']  
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
#help(Prophet)
#help(Prophet.fit)








##-------------------------------------------------------------------
## This is old and is associated with the SARIMAX UCM post processing
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
