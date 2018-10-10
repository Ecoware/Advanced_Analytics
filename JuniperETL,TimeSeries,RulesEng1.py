
#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Monday, Oct 1,  2018

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
## 
## `ConnectionModeling.Juniper_FIN1_2Days` '
##
def ReadFromBQTable():
    
    client = bigquery.Client()
    query = (
        'SELECT * '
        'FROM `ConnectionModeling.Voltage_1IP_2wks` '
        )

    voltage_df = client.query(query).to_dataframe()
    
    return(voltage_df)
    
voltage_df = ReadFromBQTable()


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
            'and TIMESTAMP >= "2018-08-13 00:00:00" and TIMESTAMP < "2018-08-13 00:10:00" '
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

#ReadFromGCS2('mwg_26wks')

#%%
## Create the date hour variable 
## string format of hour, from Big Query, is yyyy-mm-dd hh:mm:ss and the to_datetime changes it to 
## datetime64[ns]
juniper_df['hour'] = pd.to_datetime(juniper_df['hour'])

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

voltage_df.sort_values(by='hour',inplace=True)
voltage_df.reset_index(drop=True,inplace=True)

pplt.plot(voltage_df['rel_counts'] , c = 'b', label = 'relationship counts')

pplt.xticks(range(len(voltage_df['hour'])),voltage_df['hour'])

pplt.xticks(rotation=80)
pplt.title('Unique Relationship Counts for 24 hours')

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
## Write out the csv file to the local directory

#view_hour.to_csv('/Users/swe03/view_hour.txt', index=True)


#%%
## Seasonal Decomposition
def diagnostics():
    decomposition = seasonal_decompose(voltage_df['rel_counts'].values,freq=24 )  
      
    fig = decomposition.plot()  
    fig.set_size_inches(50, 8)
diagnostics()
#%%
## Graph Autocorrelation and Partial Autocorrelation data
def acf_pacf():
    fig, axes = pplt.subplots(1, 2, figsize=(15,4))
    
    fig = sm.graphics.tsa.plot_acf(voltage_df['rel_counts'], lags=24, ax=axes[0])
    fig = sm.graphics.tsa.plot_pacf(voltage_df['rel_counts'], lags=24, ax=axes[1])
acf_pacf()

#######################################
### Beginning of Prophet section
#######################################

#%%
##juniper_df = pd.DataFrame()
voltage_df['y'] = voltage_df['rel_counts']
voltage_df['ds'] = voltage_df['hour']
voltage_df.head(25)

#%%
## This will convert the datetime element to an index.  
## This may be necessary for some of the non-Prophet plots 
voltage_df2 = voltage_df
voltage_df2.reset_index(inplace=True)
voltage_df2 = voltage_df.set_index('ds')
voltage_df2.sort_index(inplace=True)
voltage_df2.info()

#%%
## Prophet1
# set the uncertainty interval to 95% (the Prophet default is 80%)
m = Prophet()
m.add_seasonality(name='hourly', period=24, fourier_order=2)
m.fit(voltage_df);


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
m.plot(forecast)

#%%
## Save a copy of the plot
fig = m.plot(forecast)  
fig.savefig("/home/steve/forecast_raw.jpeg")

#%%
## I think it is unecessary to review exponentiated components 
## Plus the complexity of joining forecast2 with forecast1
m.plot_components(forecast1);

#%%
## It was necessary, in the fill_between, to use a datetime index associated with 
## the first parameter of the function.
## This necessitated converting the existing ds datetime element to an index
pplt.subplots(figsize=(30,10))

## This may or may not be necessary depending on whether it was already done above
#forecast.set_index('ds',inplace=True)

## If using the view_hour data it will be REQUIRED to exponentiate the forecasts (i.e., forecast2)
pplt.plot(voltage_df['rel_counts'], label='Original', color='black');

pplt.plot(forecast.yhat, color='red', label='Forecast');
pplt.fill_between(forecast.index, forecast['yhat_upper'], forecast['yhat_lower'], color='gray', alpha=0.25)
pplt.ylabel('Record Counts');
pplt.xlabel('Hours');

pplt.savefig("/home/steve/forecast_raw_pplt.jpeg")
pplt.show()


#%%
# RMSE accuracy measure: the "standard deviation" of the error (i.e., expressed in the data units)

#actual = view_hour['distinct_freq_sum']  ## These are the raw input values
actual = voltage_df['y']  ## These are the raw input values after the log transform
predicted = forecast[:336]['yhat']  
mse = mean_squared_error(predicted, actual)
rmse = sqrt(mse)
print('RMSE: %.3f' % rmse)

#%%
## These were just some descriptive views into the various dataframes

voltage_df.info()
forecast.info()
voltage_df[335:]
forecast[:].yhat_upper
forecast[:].yhat_lower
