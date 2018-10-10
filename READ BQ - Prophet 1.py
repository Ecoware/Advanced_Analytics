
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
## `ConnectionModeling.Juniper_FIN1_2Days` '
## INVESTIGATE  the different occurences of ports as integer and string w a decimal (23.0 as str)
## Removed 'where ports = "23.0" '  since it created too much sparcity in the modeling data set
def ReadFromBQTable():
    client = bigquery.Client()
    query = (
        'SELECT hour, sum(rel_counts) as port_counts, sum(bytes_from_server) as bytes_from_server \
         FROM `ConnectionModeling.juniper_2wk` '
         'group by hour'
         )

    juniper_df = client.query(query).to_dataframe()
    
    return(juniper_df)
    
#juniper_df_port23 = ReadFromBQTable()
juniper_df_port23 = ReadFromBQTable()

#%%
def ReadFromBQTable2():
    client = bigquery.Client()
    query = (
        'SELECT hour, sum(rel_counts) as port_counts, sum(bytes_from_server) as bytes_from_server \
         FROM `ConnectionModeling.pci_df_2wk` '
         'group by hour'
         )

    juniper_df = client.query(query).to_dataframe()
    
    return(juniper_df)
    
#juniper_df_port23 = ReadFromBQTable()
juniper_df_port23 = ReadFromBQTable2()



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
    
#ReadFromBQ()

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
    
#ReadFromMWG_BQ1()

#%%
## This works
## Use this additional where clause if necessary  
##'and (destination_port = 80 or destination_port = 443) '
def ReadFromJuniper_BQ1():
    client = bigquery.Client()
    global juniper_df
    query = (
            'SELECT timestamp, destination_port as dstPt, source_address as srcAd, count(*) as dstPt_cnt \
            FROM `io1-datalake-views.security_pr.juniper_junos_firewall` '
            'where source_address = "165.130.144.82" '
            'and TIMESTAMP >= "2018-09-14 00:00:00" and TIMESTAMP <= "2018-09-14 23:59:59" '
            'group by timestamp, destination_port, source_address '
            'order by timestamp desc ' )
              ##LIMIT 100000 ')

    juniper_df = client.query(query).to_dataframe()
    
    return()
    
#ReadFromJuniper_BQ1()




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

#ReadFromGCS2('mwg_26wks')

#%%
## drop = True .... don't add old index as column
juniper_df_port23.sort_values(by=['hour'],ascending=[True],inplace=True)
juniper_df_port23.reset_index(inplace=True, drop=True)

#%%
##  For research sake, modified the questional set of low values

#juniper_df['rec_count'] = juniper_df['rec_count'].apply(lambda x: x+175 if x < 800 else x )
#juniper_df.sort_values(['hour'],ascending=[True])


#%%

import matplotlib.pyplot as pplt

from matplotlib.pylab import rcParams

## the first param (i.e., width) was 15 and the x-axis was unreadable
rcParams['figure.figsize'] = 75, 6

from datetime import datetime, timedelta

desired_width = 250
pd.set_option('display.width',desired_width)
pd.set_option('display.max_rows', 500)

#unq_rel_cnts80 = unq_rel_cnts.loc[(unq_rel_cnts['dstPt'] == 80),['date_hour','dstPt','dstPt_cnt']]

#pplt.plot(juniper_df_port23['port_counts'] , c = 'b', label = 'counts')
pplt.plot(juniper_df_port23['bytes_from_server'] , c = 'b', label = 'bytes_from_server')

pplt.xticks(range(len(juniper_df_port23['hour'])),juniper_df_port23['hour'])

pplt.xticks(rotation=80)
pplt.title('Port 23 Counts for 2 week')

pplt.show()
#pplt.savefig('juniper_port23.pdf')   ## or .pdf 


#%%
import statsmodels.tsa as tsa

from statsmodels.tsa.stattools import ccf 
from statsmodels.graphics.api import qqplot
from statsmodels.tsa.base import datetools
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.arima_model import ARMA
from pandas import DataFrame

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
## Write out the csv file to the local directory

#view_hour.to_csv('/Users/swe03/view_hour.txt', index=True)

#%%
## Some preprocessing to remove outliers and/or LOG transform
juniper_df_port23['bytes_from_server'] = juniper_df_port23['bytes_from_server'].apply(lambda x: x/1000 if x >= 1000000000 else x ) 
juniper_df_port23['bytes_from_server'] = np.log(juniper_df_port23['bytes_from_server'])
#%%
## Add recs to dataframe
s1 = pd.Series(['2018-09-24 00:00:00',],index=['date_hour','srcAd','dstAd','dstPt'])

#%%
## Correlation between Target variables and Regressors
stats.pearsonr(juniper_df_port23['port_counts'], juniper_df_port23['bytes_from_server'])


#%%
## Seasonal Decomposition
def diagnostics():
    #decomposition = seasonal_decompose(juniper_df_port23['port_counts'].values,freq=24 )  
    decomposition = seasonal_decompose(juniper_df_port23['bytes_from_server'].values,freq=24 ) 
    fig = decomposition.plot()  
    fig.set_size_inches(50, 8)
diagnostics()
#%%
## Graph Autocorrelation and Partial Autocorrelation data
def acf_pacf():
    fig, axes = pplt.subplots(1, 2, figsize=(15,4))
    
    #fig = sm.graphics.tsa.plot_acf(juniper_df_port23['port_counts'], lags=24, ax=axes[0])
    #fig = sm.graphics.tsa.plot_pacf(juniper_df_port23['port_counts'], lags=24, ax=axes[1])
    fig = sm.graphics.tsa.plot_acf(juniper_df_port23['bytes_from_server'], lags=24, ax=axes[0])
    fig = sm.graphics.tsa.plot_pacf(juniper_df_port23['bytes_from_server'], lags=24, ax=axes[1])
acf_pacf()




#%%
## ARIMA model to just look at residuals
model = ARIMA(juniper_df_port23['port_counts'], order=(1,0,0))
model_fit = model.fit(disp=0)
print(model_fit.summary())
# plot residual errors
residuals = DataFrame(model_fit.resid)
residuals.plot()
pplt.show()
residuals.plot(kind='kde')
pplt.show()
print(residuals.describe())

#%%
#### ARIMA Model 
##
X = juniper_df_port23['port_counts']
size = int(len(X) * 0.66)
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()
for t in range(len(test)):
	model = ARIMA(history, order=(1,0,7))
	model_fit = model.fit(disp=0)
	output = model_fit.forecast()
	yhat = output[0]
	predictions.append(yhat)
	obs = test.values[t]
	history.append(obs)
	#print('predicted=%f, expected=%f' % (yhat, obs))
error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)

#%%
## ARIMA Model and Forecast for the additional regressor(s) for the Prophet 
## method below

X = juniper_df_port23['bytes_from_server']
size = int(len(X) * 0.66)
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()
for t in range(len(test)):
	model = ARIMA(history, order=(2,0,0))
	model_fit = model.fit(disp=0)
	output = model_fit.forecast()
	yhat = output[0]
	predictions.append(yhat)
	obs = test.values[t]
	history.append(obs)

	#print('predicted=%f, expected=%f' % (yhat, obs))
error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)

## Added a Forecast object to have predicted values for the future periods for the 
## regressor bytes_from_server
forecast_regressor = model_fit.predict(start=0, end=340)
## Needed to convert array to dataframe
forecast_regressor = pd.DataFrame({'bytes_from_server':forecast_regressor[:]})
## Now Join the two. Default is to join on the indexes
add_regressor_df = forecast_regressor.join(future) 


# plot
from matplotlib.pylab import rcParams

## the first param (i.e., width) was 15 and the x-axis was unreadable
rcParams['figure.figsize'] = 75, 6
test.reset_index(drop=True,inplace=True)
pplt.plot(test)
pplt.plot(predictions, color='red')
pplt.show()


#######################################
### Beginning of Prophet section
#######################################

#%%
##juniper_df_port23 = pd.DataFrame()
juniper_df_port23['y'] = juniper_df_port23['port_counts']
#juniper_df_port23['y'] = np.log(juniper_df_port23['port_counts'])

juniper_df_port23['ds'] = juniper_df_port23['hour']
juniper_df_port23.head(25)

#%%
## This will convert the datetime element to an index.  
## This may be necessary for some of the non-Prophet plots 

#juniper_df2 = juniper_df
#juniper_df2.reset_index(inplace=True)
#juniper_df2 = juniper_df.set_index('ds')
#juniper_df2.sort_index(inplace=True)
#juniper_df2.info()

#%%


#%%
## Prophet1
# set the uncertainty interval to 95% (the Prophet default is 80%)
m = Prophet(interval_width=0.95)
m.add_seasonality(name='hourly', period=24, fourier_order=6)
m.add_regressor(name='bytes_from_server')
m.fit(juniper_df_port23);



#%%
## Create a dataframe for the future dates
## The tail will only display the time periods without the forecasted values
future = m.make_future_dataframe(periods=5,freq='H')
future.tail()

#%%
## This is the data that is exponentiated below
forecast = m.predict(add_regressor_df)
#forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

#%%
## This is the data that retains the log transform
## Note that the predict function will create a df that contains
##   many period features(e.g., trend, daily, hourly, weekly, seasonal
##   along with _upper and _lower ci's). Execute a .info() against
##   the dataframe to see all the elements.
## This creates a dataframe with just the 4 elements below

#forecast1 = m.predict(future)
#forecast1[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail() 


#%%
## This works !
## Expentiation of the Confidence Interval is invalid.
## For C.I. either keep everything as a log value (including the Hourly assessment)
##   or use the Delta Method. 

#print(np.exp(forecast['yhat'][0:5]))
#print(forecast['yhat_lower'][0:5])
#print(forecast['yhat_upper'][0:5])


#print(forecast1[['ds','yhat','yhat_lower','yhat_upper']])

#%%
## This works and retained the dataframe type
#forecast2 = np.exp(forecast1[['yhat','yhat_lower','yhat_upper']])

## Now merge to bring the ds back into the df
## Without the "on" keyword the join key is implicitly the index which is what we're doing here
#forecast2 = forecast2.join(forecast1['ds'], how='inner')



#%%
## This works
## This will create a plot that includes Forecasted, C.I.'s, and Actual values
m.plot(forecast)

#%%
## Save a copy of the plot
fig = m.plot(forecast)  
fig.savefig("forecast_raw.jpeg")

#%%
## I think it is unecessary to review exponentiated components 
## Plus the complexity of joining forecast2 with forecast1

m.plot_components(forecast);

#%%
## It was necessary, in the fill_between, to use a datetime index associated with 
## the first parameter of the function.
## This necessitated converting the existing ds datetime element to an index
pplt.subplots(figsize=(30,10))

## This may or may not be necessary depending on whether it was already done above
#forecast.set_index('ds',inplace=True)

## If using the view_hour data it will be REQUIRED to exponentiate the forecasts (i.e., forecast2)
pplt.plot(juniper_df_port23['rec_count'], label='Original', color='black');

pplt.plot(forecast.yhat, color='red', label='Forecast');
pplt.fill_between(forecast.index, forecast['yhat_upper'], forecast['yhat_lower'], color='gray', alpha=0.25)
pplt.ylabel('Record Counts');
pplt.xlabel('Hours');

pplt.savefig("/home/steve/forecast_raw_pplt.jpeg")
pplt.show()


#%%
# RMSE accuracy measure: the "standard deviation" of the error (i.e., expressed in the data units)

#actual = view_hour['distinct_freq_sum']  ## These are the raw input values
actual = juniper_df_port23['y']  ## These are the raw input values after the log transform
predicted = forecast[:336]['yhat']  
mse = mean_squared_error(predicted, actual)
rmse = sqrt(mse)
print('RMSE: %.3f' % rmse)

#%%
## These were just some descriptive views into the various dataframes

juniper_df.info()
forecast.info()
juniper_df[335:]
forecast[:].yhat_upper
forecast[:].yhat_lower
