

# In[3]:
# use print only as a function
from __future__ import print_function
import sys
#sys.version
#pd.show_versions()

# In[4]:
import warnings
warnings.filterwarnings('ignore')

#%%
#!pip freeze

# In[5]:


__author__ = 'swe03'

import numpy as np
import pandas as pd

import matplotlib.pylab as plt
import matplotlib.pyplot as pplt
from decimal import *
 

import statsmodels.tsa as tsa

from statsmodels.tsa.stattools import ccf 
from statsmodels.graphics.api import qqplot
from statsmodels.tsa.base import datetools
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.arima_model import ARMA

from statsmodels.tsa.stattools import acf  
from statsmodels.tsa.stattools import pacf
from statsmodels.tsa.seasonal import seasonal_decompose

from dateutil.relativedelta import relativedelta
import seaborn as sns
import statsmodels.api as sm  


from sklearn.metrics import mean_squared_error

from scipy.stats.stats import pearsonr
from scipy import stats
from sklearn import linear_model

get_ipython().magic(u'matplotlib inline')
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15, 6

from datetime import datetime, timedelta

desired_width = 250
pd.set_option('display.width',desired_width)
pd.set_option('display.max_rows', 500)


# In[6]:
from pandasql import PandaSQL 
pdsql = PandaSQL()

# In[8]:


from StringIO import StringIO
import IPython.core.magic as magic


#%%
## Concatenate a set of csv files into one file

unq_rel_cnts = pd.DataFrame()

for i in range(12,14):  ## This range is 3.12 to 3.25.  The 26th is there as the non-inclusive end boundary
    s = str(i)
    urc_single_file  = pd.DataFrame()
    
    fileString = '/Users/swe03/urc_3_'+ s +'.csv' 
   
    urc_single_file = pd.read_csv(fileString,sep=',',nrows=1000)
    
    unq_rel_cnts = pd.concat([unq_rel_cnts, urc_single_file],ignore_index=True)

unq_rel_cnts.info()
unq_rel_cnts.groupby('date_hour').count()
unq_rel_cnts.date_hour.unique()

# In[11]:


import matplotlib.pyplot as pplt
pplt.plot(unq_rel_cnts['distinct_freq'])
pplt.xticks(range(len(unq_rel_cnts['date_hour'])),unq_rel_cnts['date_hour'])
pplt.xticks(rotation=80)
#pplt.autoscale(enable=True, axis='x', tight=None)
pplt.show()


# In[12]:


unq_rel_cnts.sort_values(by='date_hour',inplace=True)
unq_rel_cnts.info()


# In[13]:


#Need a DateTime Index for all of the time series modeling functions.
#Note in the output from ent_calc_df1.index.dtype: dtype('<M8[ns]') but...  np.dtype('datetime64[ns]') = np.dtype('<M8[ns]') so should
# be OK:
# First create a new datetime dtype variable called datetime
unq_rel_cnts['datetime'] = pd.to_datetime(unq_rel_cnts['date_hour'])   ## unq_rel_cnts['datetime'].dtype displays dtype('<M8[ns]')


# In[14]:


unq_rel_cnts.reset_index(inplace=True)
unq_rel_cnts = unq_rel_cnts.set_index('datetime')
unq_rel_cnts.sort_index(inplace=True)


# In[15]:


### For the current time frame this filter removes the outliers
### NOTE: THIS IS AN OUTSTANDING DESIGN QUESTION FOR ME AND DAVID.  SEE ONENOTE NOTES IN TODO FOLDER
unq_rel_cnts1 = unq_rel_cnts.loc[(unq_rel_cnts['date_hour'] > '2017-08-11 05:00:00')]  ## use loc when using labels
unq_rel_cnts1


# In[16]:


####### Graph to just plot a Range (or ALL) of frequency values directly from the df ########
####### figsize(x-axis length, y-axis length)

fig, ax = plt.subplots(figsize=(9,4))  ## Changing these settings didn't seem to do anything other then change the color 

fig.suptitle('Unique SrcAd, DstAd/DstPt by Hour', fontsize=30)
plt.xticks(rotation=80)  # Not working
plt.xlabel('Hour', fontsize=28)   # The fontsize here is applied to the variable name not the label in the function
plt.ylabel('Unique Counts', fontsize=26)

# fontsize is just for the axes size
unq_rel_cnts1['distinct_freq'].loc[:].plot(figsize=(40,8), fontsize=30)  


# #### Execute some Univariate Statistics

# In[17]:


unq_rel_cnts1['distinct_freq'].describe()


# In[18]:


decomposition = seasonal_decompose(unq_rel_cnts1['distinct_freq'].values,freq=24 )  
  
fig = decomposition.plot()  
fig.set_size_inches(15, 8)


# In[19]:


# Graph Autocorrelation and Partial Autocorrelation data
fig, axes = plt.subplots(1, 2, figsize=(15,4))

fig = sm.graphics.tsa.plot_acf(unq_rel_cnts1['distinct_freq'], lags=12, ax=axes[0])
fig = sm.graphics.tsa.plot_pacf(unq_rel_cnts1['distinct_freq'], lags=12, ax=axes[1])


# ## Section for Statistical Modeling

# In[20]:


## Specify the SARIMAX model
## Default for the CI is 95%.  Set in the Alpha parameter for conf_int function

 
mod = sm.tsa.statespace.SARIMAX(unq_rel_cnts1['distinct_freq'], order=(3,0,0) )     ## ,seasonal_order=(1,0,0,24))
#mod1 = sm.tsa.statespace.SARIMAX(ent_calc_df['In_degree_entropy'], order=(8,0,0), seasonal_order=(2,0,0,6))
#mod2 = sm.tsa.statespace.SARIMAX(ent_calc_df['In_degree_entropy'], trend='n', order=(1,0,0), seasonal_order=(2,0,0,24))
results = mod.fit()
print(results.summary()) 


# #### Post model Goodness Of Fit assessments

# In[21]:


fig = results.plot_diagnostics()


# In[22]:


## Check to see if Residuals are still correlated
## Values are between 0 and 4.  Values close to 2: no Serial Correlation. Close to 0: Pos Corr. Close to 4: Neg Corr.
sm.stats.durbin_watson(results.resid.values)


# In[23]:


## Check to see if the distributionb of Residuals are Normally Distributed (i.e., This is undesirable)
##If the p-val is very small, it means it is unlikely that the data came from a normal distribution
resid1 = results.resid
stats.normaltest(resid1)


# In[24]:


fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111)
fig = qqplot(resid1, line='q', ax=ax, fit=True)


# In[25]:


# Graph the acf and pacf for the Residuals
fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(resid1.values.squeeze(), lags=10, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(resid1, lags=10, ax=ax2)


# In[26]:


# Box-Pierce Q statistic tests the Null Ho that *ALL* correlations up to Lag K are equal to Zero.  
## Thus, the desired output is to accept the Null Ho (i.e., no correlations) indicated by high Prob(>Q) values
# This is not the same as the correlogram above.
def skip():
  r,q,p = sm.tsa.acf(resid1.values.squeeze(), qstat=True)
  data = np.c_[range(1,41), r[1:], q, p]
  table = pd.DataFrame(data, columns=['lag', "AC", "Q", "Prob(>Q)"])
  print(table.set_index('lag'))


# In[27]:


# This is just a SARIMAX Results Wrapper.  It collects the model parameters for later use.
res = mod.filter(results.params)


# ### Section for Prediction and Forecasting 

# In[28]:


### In-sample prediction and out-of-sample forecasting
####################
### predict() with no parameters will produce just In-sample predictions
####################
### With index based Start/End, must start at less than or equal to Length of training data 
###   (zero based index: 504 records, last rec is 503)
### Can only use dates if index is datetime.

#predict_res = res.predict(alpha=.05, start=503, end=510)    ## 2017-02-26 23:00:00 is 503
#print(predict_res)

#predict_res = res.predict(alpha=.05, start='2017-02-26 00:00:00', end='2017-02-27 01:00:00') # Note limited range
def skip():
  predict_res = res.predict()
  predict_res


# In[29]:


### In-sample prediction and out-of-sample forecasting
###########################
### The get_prediction function works the same as above res.predict() except that exogenous variables can be 
###   included as a parameter in this function as well as more statistical functions in the predict object model 
###   like predicted_mean and predicted_results
###########################
predict = res.get_prediction()
#predict.predicted_mean


# In[32]:


## Create the Confidence Intervals for the entire range
## Note that using the full range will create initial C.I.'s that are large (i.e., not enough history for tighter C.I.'s)
##   This will change the scale of the graph and is not useful for visualization below
predict_ci = predict.conf_int()
predict_ci


# In[39]:


## Create the Predicted Means for Graph #1 below
## The Predicted Means are both In-Sample Predictions as well as Out-Of-Sample Forecast
## Can just use the last datetime of the training sample and get the next forecasted hour but any range with the training data will give the 
##  same forecasted results
#predict_dy.predicted_mean.astype(int)
#type(predicted_mean_1)  ## Series
predicted_mean_1 = predict.predicted_mean.loc['2017-08-14 23:00:00':'2017-08-15 01:00:00']   # include 8-14 02:00 since there is a spike
predicted_mean_1


# In[40]:


## Create the Predicted Means for Graph #2 below
## The Predicted Means are both In-Sample Predictions as well as Out-Of-Sample Forecast
## Can just use the last datetime of the training sample and get the next forecasted hour but any range with the training data will give the 
##  same forecasted results
#predict_dy.predicted_mean.astype(int)
predicted_mean_2 = predict.predicted_mean.loc['2017-08-12 12:00:00':'2017-08-15 02:00:00'] 
predicted_mean_2


# In[41]:


### Create the Confidence Intervals for Graph #1 below
### This is a constrained view based on dates and is used below in the graphing
### Note:  The Start date can be any date in the Training Set data.  Any date will apparently not change the predicted values and
###        and, here, the CI
predict = res.get_prediction(start='2017-08-14 01:00:00', end='2017-08-15 02:00:00')
predict_ci_1 = predict.conf_int()
predict_ci_1


# In[42]:


### Create the Confidence Intervals for Graph #2 below
### This is a constrained view based on dates and is used below in the graphing
### Note:  The Start date can be any date in the Training Set data.  Any date will apparently not change the predicted values and
###        and, here, the CI
predict = res.get_prediction(start='2017-08-12 12:00:00', end='2017-08-15 02:00:00')
predict_ci_2 = predict.conf_int()
#predict_ci_2


# In[43]:


#print(res.forecasts[0,:])  # res.forecasts is a one-dim array.  Single row(0) All cols(:) for all values
#print(res.forecasts[0,0:6]) # res.forecasts is a one-dim array.  Singel row(0) Range of cols(1:5) for 4 values
#res.forecast(steps=5)  # Should be number of forecasted values from the last obs


# In[44]:


predicted_mean_1_df = predicted_mean_1.reset_index()  ## Creates a DataFrame
predicted_mean_1_df = predicted_mean_1_df.set_index('index')  ## Set the index in the df to the same as the ci df
predicted_mean_1_df = predicted_mean_1_df.rename(columns={0:'mean'})
predicted_mean_1_df['mean']


# ### Create and populate the Model Tolerance Alert table 

# In[45]:


mod_tol = predicted_mean_1_df.join(predict_ci_1, how='inner')                    
mod_tol


# In[46]:


unq_rel_cnts1 = unq_rel_cnts1[['distinct_freq']]  ## Only select this single column. Use double brackets to return a DataFrame
unq_rel_cnts1


# In[47]:


## Only run this once otherwise you get additonaly distinct_freq_x and _y columns
mod_tol_w_freqs = pd.merge(mod_tol,unq_rel_cnts1, how='inner', left_index=True, right_index=True)             
mod_tol_w_freqs


# In[48]:


### distinct_freq are the "actuals" that are compared to the Upper and Lower C.I. Bounds
mod_tol_w_freqs.loc[(mod_tol_w_freqs['distinct_freq'] > mod_tol_w_freqs['upper distinct_freq']) | 
                  (mod_tol_w_freqs['distinct_freq'] < mod_tol_w_freqs['lower distinct_freq']) 
              ,'alert_level']=4


# In[49]:


mod_tol_w_freqs


# In[50]:


mod_tol_alert = mod_tol_w_freqs.loc[mod_tol_w_freqs['alert_level'] ==  4]
mod_tol_alert.reset_index(inplace=True)   ## date_hour needs to be removed from index and set as a column
mod_tol_alert.rename(columns={'index':'date_hour'},inplace=True)


# In[51]:


## Necessary to convert date_hour datetime w/o microseconds to a String with microseconds to allow groupby in the Summary 
##  processing below
mod_tol_alert['date_hour'] = mod_tol_alert.date_hour.dt.strftime('%Y-%m-%d %H:%M:%S.%f')
mod_tol_alert


# ## Section for Rules Engine

# In[52]:


### Set the base equal to a range to compare each new hour to....
dfx_base = pdsql("""select distinct date_hour, srcAd, dstAd, dstPt  
                              from dfx_distinct  
                              where date_hour between('2017-08-11 00:00:00.000000') and ('2017-08-14 00:00:00.000000')  
                              group by date_hour, srcAd, dstAd, dstPt
                              order by  date_hour, srcAd, dstPt""", locals()) 
dfx_base


# In[ ]:


### DateHour range for base above is: date_hour between('2017-08-11 00:00:00.000000') and ('2017-08-14 00:00:00.000000') 
### Thus, start the new hour at: date_hour in ('2017-08-14 01:00:00.000000')

dfx_newhr = pdsql("""select distinct date_hour, srcAd, dstAd, dstPt  
                              from dfx_distinct  
                              where date_hour in ('2017-08-14 01:00:00.000000')
                              group by date_hour, srcAd, dstAd, dstPt
                              order by  dstPt""", locals()) 
dfx_newhr


# In[54]:


##### Write the dataframe to BQ
#dfx_base.to_gbq('ConnectionModeling.dfx_base', "network-sec-analytics", verbose=True, reauth=False, 
#  if_exists='replace', private_key=None)


# ### Section to automate cycling through the New Hour

# In[55]:


#########################################################
### Iterate through various dates to create the New Hour for the Summary Statistics.
### This function is called by the Initialize and Iterate function below.
### DateHour range for base above is: date_hour between('2017-08-11 00:00:00.000000') and ('2017-08-14 00:00:00.000000') 
### Thus, START THE NEW HOUR AT: date_hour in ('2017-08-14 01:00:00.000000'). WILL ALWAYS BE THE FIRST HR AFTER THE LAST BASE HR.
#########################################################

def Get_recs_w_date_range(date_s):
    global dfx_newhr
    global query
    
    query = """select distinct date_hour, srcAd, dstAd, dstPt  
                              from dfx_distinct  
                              where date_hour in ('{}')
                              group by date_hour, srcAd, dstAd, dstPt
                              order by  dstPt""".format(date_s)
    print('The value of local var date_s is: {}'.format(date_s))
    dfx_newhr = pdsql(query, globals()) 

    return


# In[73]:


def Initialize_and_Iterate():
  global date_start
  global df_final
  global data_var
  date_start_str = '2017-08-14 05:00:00.000000'                          ## Was 01:00:00.000000
  date_start = pd.to_datetime(date_start_str)            
  date_end_interval   = pd.to_datetime('2017-08-14 05:00:00.000000')     ## Was 23:00:00.000000
  
  while date_start <= pd.to_datetime(date_end_interval):
    
    print('For get_recs function date_start=',date_start)
    
    Get_recs_w_date_range(date_start_str)  
    rules1a()
    rules1b()
    rules2a()
    rules2b()
    rules_output()
    
    date_start = date_start + timedelta(hours=1)           
    date_start_str = date_start.strftime('%Y-%m-%d %H:%M:%S.%f')
  
  return    


# In[57]:


##### Write the dataframe to BQ
#dfx_newhr.to_gbq('ConnectionModeling.dfx_newhr_8_14_2', "network-sec-analytics", verbose=True, reauth=False, 
#  if_exists='replace', private_key=None)


# In[58]:


### Test for a component(i.e., unknown srcIP) of the first rule(i.e., unknown srcIP / known destPort)
### The Left Outer Join will identify those srcIP's, in the new hour's data, that are not found in the Base set of srcIP's by 
###   using the bsrcAd element. This element will be populated with "None" when no match exists.
### dfx_newhour0 will always be the current, single hour
def rules1a():
    global rule1a
    rule1a = pdsql("""select nh.date_hour, nh.srcAd as nhsrcAd, b.srcAd as bsrcAd, nh.dstAd as nhdstAd, nh.dstPt as nhdstPt 
                                   from dfx_newhr nh left outer join dfx_base b 
                                     on nh.srcAd = b.srcAd  

                                  order by  nh.srcAd, nh.dstAd, nh.dstPt""", globals()) 

    ### Convert to NumPy array to enable filtering of the NoneType value "None" from the outer join and then overwrite the df
    rule1a_bsrcAd = rule1a['bsrcAd'].values.copy()
    rule1a = rule1a.loc[rule1a_bsrcAd == None]  ## This enables the loc to read None and only select those records
    return    


# In[59]:


### Test for the second component(i.e., unknown dstPt) of the first rule 
### This searches for any dest port in base history
def rules1b():
    global rule1b
    rule1b = pdsql("""select a.date_hour, a.nhsrcAd, a.bsrcAd, a.nhdstAd, a.nhdstPt, b.dstPt as bdstPt
                      from rule1a a left outer join dfx_base b 
                        on a.nhdstPt = b.dstPt 
                        order by a.nhsrcAd""", globals()) 
    

    ### Create the new column to store the alert level
    rule1b['alert_level'] = np.zeros(rule1b.shape[0])  

    ## Set both cols to a NumPy arrary 
    ## dtype of either column will either be dtype('O') for None OR dtype('float64') for NaN
    np_bsrcAd = rule1b['bsrcAd'].values.copy()
    np_bdstPt = rule1b['bdstPt'].values.copy()

    ## Execute the conditional logic to set the Alert Level

    if np_bdstPt.dtype == 'O':
        rule1b['alert_level'][(np_bsrcAd == None) & (np_bdstPt == None)] = 3   ## Set alert_level to 3 if both cond are satisfied
        rule1b   ## Why doesn't this display
    else:
        rule1b['alert_level'][(np_bsrcAd == None) & (np.isnan(np_bdstPt) == False)] = 1
        rule1b['alert_level'][(np_bsrcAd == None) & (np.isnan(np_bdstPt))] = 3
        print("if not satisfied")
    
    return   


# In[60]:


### Test for a component(i.e., unknown destPort) for the second rule(i.e., any srcIP / unknown destPort)
### The Left Outer Join will identify those dstPt's, in the new hour's data, that are not found in the Base set of dstPt's by 
###   using the dstPt/nhdstPt element. This element will be populated with "None" or "NaN" when no match exists.
### dfx_newhour0 will always be the current, single hour

def rules2a():
  global rule2a
  rule2a = pdsql("""select distinct nh.date_hour, nh.srcAd as nhsrcAd, b.srcAd as bsrcAd, nh.dstAd as nhdstAd, nh.dstPt as nhdstPt, 
                                    b.dstPt as bdstPt
                                 from dfx_newhr nh left outer join dfx_base b 
                                   on nh.srcAd = b.srcAd  

                           order by   nh.srcAd, nh.dstPt""", globals()) 
  
  return


# In[61]:


### Test for the second component(i.e., known dstPt)  
### This searches for any dest port in base history that is not in the rule2a set.
### The rule2a set will include recs where there is a known srcAd in both newhour and base

def rules2b():
  global rule2b
  rule2b = pdsql("""select distinct a.date_hour, a.nhsrcAd, a.bsrcAd, a.nhdstAd, a.nhdstPt, b.dstPt as bdstPt
                    from rule2a a left outer join dfx_base b 
                      on a.nhdstPt = b.dstPt 
                      order by a.nhsrcAd""", globals()) 
  
  ### Using the or condition above (and better test cases where the srcAd exists but the dstPt does not in the base)
  ###  try using a filter process where bsrcAd is not None(i.e., existing srcAd) and the bdstPt is NaN
  np_bsrcAd = rule2b['bsrcAd'].values.copy()
  rule2b = rule2b.loc[np_bsrcAd != None]  ## This enables the loc to read "not None" and only select those records   
  rule2b['alert_level'] = np.zeros(rule2b.shape[0])  
  rule2b['alert_level'][(np.isnan(rule2b['bdstPt']))] = 2
  rule2b = rule2b.loc[(rule2b['alert_level'] > 0)]
  
  return


# In[72]:


rule2b


# ### Process the Summary Rules dataframes

# In[62]:


#######################################
##  This is a automated process to create a set of dataframes with the Rules Engine results and Summary Statistics.
##  The value passed from the concat call corresponds to the hour set above in the dfx_newhour pdsql query with 
##    the: where date_hour in ('2017-08-14 hh:00:00.000000')  and hh is the hour value
#######################################
## Rather than use describe() this has more flexibility.  It takes the raw alert data(i.e., rule1b, rule2b) and performs
## the GroupBy and the statistics aggregation. 
## rulesn is a DataFrame
## The 'alert_level' is necessary but because it creates a MultiIndex on the columns this will cause the to_gbq write to fail.
## Can, however, remove the MultiIndex 
## The value parameter is used to create separate rulesn dataframes that are concatenated below
## The number string passed in the concat function call is arbitrary but in this case corresponds to the Hour
def rules_output():
  global raw_rules
  global summary_rules
  #global raw_rules_i
  #global summary_rules_i
  
  #raw_rules_i = pd.DataFrame()
  #raw_rules = pd.DataFrame()
  #summary_rules_i = pd.DataFrame()
  #summary_rules = pd.DataFrame()


  raw_rules_i = pd.concat([rule1b,rule2b] )
  summary_rules_i = raw_rules_i.groupby('date_hour')
  summary_rules_i = summary_rules_i.agg({'alert_level':{'Alert_Mean':'mean','Alert_Count':'count','Alert_Sum':'sum','Alert_Min':'min',
                                      'Alert_Max':'max'}})
  
  summary_rules_i.columns = summary_rules_i.columns.droplevel(0)  ## Drop the MultiIndex that prevents appending and writing df to BQ
  


  raw_rules = raw_rules.append(raw_rules_i)
  summary_rules = summary_rules.append(summary_rules_i)

  summary_rules = summary_rules.rename(columns={'Alert Count': 'Alert_Count','Alert Min':'Alert_Min','Alert Max':'Alert_Max',
                                       'Alert Sum':'Alert_Sum','Alert Mean':'Alert_Mean'})
  
    
  


# In[ ]:


##### Write the dataframes to BQ
### summary_rules is read into summary_stats below
### raw_rules is used for analysis triage
#############
def writeto_BQ():
  summary_rules.to_gbq ('ConnectionModeling.summary_rules', "network-sec-analytics", verbose=True, reauth=False, 
     if_exists='replace', private_key=None)
  raw_rules.to_gbq ('ConnectionModeling.raw_rules', "network-sec-analytics", verbose=True, reauth=False, 
     if_exists='replace', private_key=None)
  return


# ## Initiate the Process

# In[74]:


raw_rules = pd.DataFrame()         # Create the rules df's that will be appended to for each timeframe
summary_rules = pd.DataFrame()
Initialize_and_Iterate()           # For each Hour, iterate through the timeframes 
#writeto_BQ()


# In[75]:


### Join raw_rules with Modeling Alerts (mod_tol_alert)
### To enable the append, first filter raw_rules for just alert_level
raw_rules_w_alerts = raw_rules[['date_hour','alert_level']]
mod_tol_alert2 = mod_tol_alert[['date_hour','alert_level']]
raw_rules_w_all_alerts = raw_rules_w_alerts.append(mod_tol_alert2)
summary_rules_w_all_alerts = raw_rules_w_all_alerts.groupby('date_hour')
summary_rules_w_all_alerts = summary_rules_w_all_alerts.agg({'alert_level':{'Alert_Mean':'mean','Alert_Count':'count','Alert_Sum':'sum','Alert_Min':'min',
                                      'Alert_Max':'max'}})


# In[77]:


rule1b


# In[78]:


rule2b


# In[79]:


mod_tol_alert2


# In[76]:


raw_rules_w_all_alerts


# In[70]:


raw_rules_w_all_alerts.to_gbq ('ConnectionModeling.raw_rules_w_all_alerts', "network-sec-analytics", verbose=True, reauth=False, 
     if_exists='replace', private_key=None)


# In[69]:


summary_rules_w_all_alerts


# In[ ]:


## The date_hour label above was a String Index. Needs to be a DateTime Index to work with the Graphing below
summary_rules_w_all_alerts2 = summary_rules_w_all_alerts.alert_level[['Alert_Sum']]   ## Creates a new dataframe
summary_rules_w_all_alerts2.index = summary_rules_w_all_alerts2.index.to_datetime()   ## Converts the index to a datetime
#summary_rules_w_all_alerts2.info()


# In[ ]:


summary_rules_w_all_alerts2


# In[ ]:


summary_stats_sum['date_hour'] = pd.to_datetime(summary_stats_sum['date_hour'],format='%Y-%m-%d %H:%M:%S.%f' )
#summary_stats_sum.sort_values('date_hour',inplace=True)
summary_stats_sum.set_index('date_hour',inplace=True)
summary_stats_sum.sort_index(inplace=True)
summary_stats_sum


# ### Graphs

# In[ ]:


###################################################################################################
## Graphs both the Observed, In-Sample Predictions as well as Out-Of-Sample Forecasts and 
##  includes appropriate Confidence Intervals
## This is all controlled with the code block above using the constrained view of the C.I.'s
## These are Dynamic predictions
###################################################################################################

fig, ax = plt.subplots(figsize=(40,8))
npre = 4

fig.suptitle('Unique Relations by Hour', fontsize=30)
plt.xlabel('', fontsize=28)   # The fontsize here is applied to the df variable name not the label in the function
plt.ylabel('Distinct Freqs', fontsize=26)
plt.xticks(rotation='vertical')  

## Plot using different data ranges and formats of the data points
unq_rel_cnts1.loc['2017-08-14 00:00:00':'2017-08-15 02:00:00','distinct_freq'].plot(ax=ax, style='blue', label='Observed')  
#ent_calc_df1.ix[402:527,'In_degree_entropy'].plot(ax=ax, style='blue', label='Observed')  
#ent_calc_df1.ix[:,'In_degree_entropy'].plot(ax=ax, style='blue', label='Observed') 

## Plot predictions
# THIS RANGE IS SET ABOVE WHEN SETTING THE "predict.predicted_mean.ix" OBJECT ######
predicted_mean_1.iloc[:].plot(ax=ax, style='red', label='Dynamic forecast', fontsize=30)

## Optional inclusion of the Summary Stats
summary_rules_w_all_alerts2.plot(ax=ax, style='orange', label='Alert Sum', fontsize=30)

## Plot the Confidence Intervals 
#ci = predict_ci.ix['2017-02-26 23:00:00':'2017-02-27 10:00:00']
#ci = predict_ci.ix[402:527]
ci = predict_ci_1.iloc[:]

ax.fill_between(ci.index, ci.iloc[:,0], ci.iloc[:,1], color='black', alpha=0.2)

legend = ax.legend(loc='lower right', fontsize=30)


# In[ ]:


# Plot another graph to view a longer time interval for both Prediction and Forecasting

fig, ax = plt.subplots(figsize=(40, 8))
fig.suptitle('Unique Relations by Hour', fontsize=30)
plt.xlabel('date/hour',fontsize=30)
plt.ylabel('Unique Freqs',fontsize=30)
ax = unq_rel_cnts1['distinct_freq'].loc['2017-08-11 06:00:00':'2017-08-15 02:00:00'].plot(ax=ax)   

predicted_mean_2.iloc[:].plot(ax=ax, style='black', label='Prediction and Forecast', fontsize=40)
ci = predict_ci_2.iloc[:]  # ci is the display labels and values for the Upper and Lower bounds 
## :,0 - Lower values by hour   
## :,1 - Upper values by hour
## alpha is transparency value.  Higher is less transparent
ax.fill_between(ci.index, ci.iloc[:,0], ci.ix[:,1], color='r', alpha=0.40)  

legend = ax.legend(loc='lower right',fontsize=35)


# ### Needs to be developed

# In[ ]:


### Picked this up from https://machinelearningmastery.com/tune-arima-parameters-python/
def skip():
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import mean_absolute_error
    from math import sqrt
    #rmse = sqrt(mean_squared_error(test, prediction))
    mae = mean_absolute_error(test, prediction)
    #print('Test RMSE: %.3f' % rmse)
    print('Test MAE: %.3f' % mae)

