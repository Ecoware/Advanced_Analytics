
# coding: utf-8

# # Time Series Prototyping for Intrusion Detection 

# In[1]:

# use print only as a function
from __future__ import print_function
import sys
sys.version_info


# ## Connect to data and read into dataframe

# In[2]:

__author__ = 'swe03'

import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from decimal import *

get_ipython().magic('matplotlib inline')
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15, 6

from scipy import stats

import statsmodels.api as sm
import statsmodels.tsa as tsa

from statsmodels.graphics.api import qqplot
from statsmodels.tsa.base import datetools
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.arima_model import ARMA

from statsmodels.tsa.stattools import acf  
from statsmodels.tsa.stattools import pacf
from statsmodels.tsa.seasonal import seasonal_decompose

from datetime import datetime

desired_width = 250
pd.set_option('display.width',desired_width)


# In[10]:

# Read in the csv file stored in Data folder in Network Security 
# Note the use of both the parse_dates and data_parser functions to convert the string into a datetime64 dtype
dateparse = lambda x: pd.datetime.strptime(x, '%m/%d/%Y %H:%M')
calc_ent1 = pd.read_csv("H:\\Network Security/Data/ExtraHop/calc_ent1.csv",parse_dates=['date_time'],date_parser=dateparse )
calc_ent1.head(5)
#print('\n Data Type')
calc_ent1.dtypes    # "Object" dtype display means a String. Which was the type before the date parsing above in the read
#type(data_sample)
#dir(calc_ent2)


#df = pd.read_csv(infile, parse_dates={'datetime': ['date', 'time']}, date_parser=dateparse)


# In[11]:

## Never used this but still curious about it.
# reader = csv.DictReader(fil, delimiter=';')         #read the csv file
#    for row in reader:
#        date = datetime.strptime(row['Dispatch date'], '%Y-%m-%d %H:%M:%S') 


# ###### Install the SQL package if not already installed 

# In[12]:

#!pip install pandasql  


# In[13]:

# Convert the ts variable to a number of date-time values for Time Series index
# Use the functions usec_to_timestamp

from pandasql import PandaSQL 
pdsql = PandaSQL()
#type(pdsql)
#local = locals()


# today = datetime.date.today()
# print today
# print 'ctime:', today.ctime()
# print 'tuple:', today.timetuple()
# print 'ordinal:', today.toordinal()
# print 'Year:', today.year
# print 'Mon :', today.month
# print 'Day :', today.day

# In[14]:

calc_ent1['ln_ent1'] = np.log(calc_ent1['ent1'])
calc_ent1
#len(calc_ent1)  #2375  Correct


# In[15]:

# The first value has been audited and validated
calc_ent2 = pdsql("""SELECT i.date_time, sum(i.ent1 * i.ln_ent1) as hx_ent,  count(i.date_time) as hour_event_cnt
                  from calc_ent1 i
                  group by i.date_time
                  order by i.date_time""",locals())


# In[16]:

calc_ent2  ## 120 records (i.e., 24*5)


# In[17]:


calc_ent2['entropy'] = -(calc_ent2['hx_ent'] / np.log(calc_ent2['hour_event_cnt']))
calc_ent2


# In[33]:

calc_ent2.to_csv("H:\\Network Security/Data/ExtraHop/calc_ent2.csv", encoding='utf-8', columns=calc_ent2.columns.values.tolist()) 


# In[18]:

#dir(calc_ent2['entropy'])


# In[24]:

# Set the index for subsequent processing
#calc_ent2 = calc_ent2.set_index(pd.DatetimeIndex(calc_ent2['date_time'])
#calc_ent2


# In[19]:

## Frequency of attributes_bytes ranges (can't yet figure out how to make bins rather than accepting default)
#dir(port_df.plot.hist)
#my_plot = port_df_totals['attributes_bytes'].plot.hist()
#my_plot = port_df_mean['attributes_bytes'].plot.hist()
my_plot = calc_ent2['entropy'].plot.hist()


# In[25]:

import matplotlib.pyplot as pplt
#my_plot=pplt.plot(port_df_mean['attributes_bytes'])
my_plot=pplt.plot(calc_ent2['entropy'])
pplt.autoscale(enable=True, axis='x', tight=None)
pplt.show()


# In[26]:

decomposition = seasonal_decompose(calc_ent2.entropy.values, freq=24)  
fig = plt.figure()  
fig = decomposition.plot()  
fig.set_size_inches(15, 8)


# In[44]:

model=ARIMA(calc_ent2['entropy'],(1,0,0))    ## The endogenous variable needs to be type Float or you get a cast error
model_fit = model.fit()       # fit is a Function
model_fitted = model_fit.fittedvalues    # fittedvalues is a Series
print(model_fit.summary())
print(model_fitted)


# In[29]:

from pprint import pprint               # get a variety of different attributes from the object (including functions)
#pprint (dir(model))
#pprint (dir(model_fit))


# In[30]:

print(model.endog_names)
print(model.exog_names)
print(model.information)
print(model.predict)


# In[34]:

#print(model_fit.resid)
#print(model_fit.fittedvalues)

#Plot rolling statistics:
fig = plt.figure(figsize=(12, 8))
orig = plt.plot(model_fit.fittedvalues, color='blue',label='Fitted')
resid = plt.plot(model_fit.resid, color='red', label='Residuals')
   
plt.legend(loc='best')
plt.title('Residual Error')
plt.show()


# In[57]:

## One of the developers of the method states "Note that ARMA will fairly quickly converge to the long-run mean, 
## provided that your series is well-behaved, so don't expect to get too much out of these very long-run prediction exercises.
## http://stats.stackexchange.com/questions/76160/im-not-sure-that-statsmodels-is-predicting-out-of-sample
predict = model_fit.predict(start='2016-09-10 00:00:00' , end='2016-09-10 23:00:00' ,dynamic=True)
#forecast = model_fit.forecast(steps=10, exog=None, alpha=0.05)  ## Never got this to work
print(predict)
#dir(model_fit)

fig = plt.figure(figsize=(10,3))
ax = fig.add_subplot(111)
ax.plot(predict)
#ax.plot(forecast)

#ax.plot(calc_ent2.index, calc_ent2['entropy'])
##ax.plot(calc_ent2.index, calc_ent2['entropy']-calc_ent2['entropy'].mean())
#ax.plot(predict.index, predict, 'r-')

