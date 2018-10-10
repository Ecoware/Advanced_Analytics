
# coding: utf-8

# ## Derive a Profile for Destination Hosts and associate Ports

# In[1]:

# use print only as a function
from __future__ import print_function
import sys
sys.version_info


# ## Connect to data and read into dataframe

# In[1]:

__author__ = 'swe03'

import argparse

import numpy as np
import pandas as pd
#import matplotlib.pylab as plt
#import matplotlib.pyplot as pplt
from decimal import *

#get_ipython().magic(u'matplotlib inline')
#from matplotlib.pylab import rcParams
#rcParams['figure.figsize'] = 15, 6

from scipy import stats

from datetime import datetime, timedelta

#desired_width = 250
#pd.set_option('display.width',desired_width)
#pd.set_option('display.max_rows', 500)


# ###### Install the SQL package if not already installed 

# In[3]:

#!pip install pandasql


# In[4]:

from pandasql import PandaSQL 
pdsql = PandaSQL()


# ##### Use an existing table but change the timestamps to create a contiguous distribution.

# In[5]:

def get_recs_w_date_range(date_s,date_e):
    """Iterate through various date ranges to create the a timeframe sample for later aggregation"""
    global dfx2   # Otherwise, dfx2 is considered Local and will not be global scope of the dataframe created above
    query = """SELECT Timestamp, dst_addr,
                 cast(dst_port as integer) as dst_port,
                 cast(duration_ms as integer) as duration_ms,
                 cast(bytes as integer) as bytes,
                 protocol, flow_direction 
               FROM ipfix.ipfix                
               WHERE Timestamp BETWEEN timestamp('{}') AND timestamp('{}')     
               LIMIT 10 """.format(date_s,date_e)
    #print('The value of local var date_s is: {}'.format(date_s))
    #print('The value of local var date_e is: {}'.format(date_e))
    dfx1 = pd.read_gbq(query, project_id="network-sec-analytics",reauth=True)
    dfx2 = dfx2.append(dfx1)                                       # Append onto the dfx2 dataframe
    return           


# In[6]:

def Initialize_and_Iterate():
  #date_start = pd.to_datetime('2016-11-20 00:00:00')       
  #date_end_interval   = pd.to_datetime('2016-11-26 23:59:59')    
  date_start = pd.to_datetime(args.datetime_start)       
  date_end_interval   = pd.to_datetime(args.datetime_end)  
  while date_start <= pd.to_datetime(date_end_interval):
    date_end = date_start + timedelta(minutes=59,seconds=59)   # Set the datetime end of the hour interval
    print('For get_recs function date_start=',date_start)
    print('For get_recs function date_end=',date_end)
    get_recs_w_date_range(date_start,date_end)             # Extract the query Limit above within the specified hour     
    date_start = date_end + timedelta(seconds=1)           # Add a second to the nn:59:59 end date to start the next 
                                                           # hour on nn:00:00 start time
  return      


# In[7]:

def Create_date_hour():
  global dfx2
  dfx2['duration_ms'].fillna(0, inplace=True)
  dfx2['date_hour'] = dfx2.Timestamp.dt.strftime('%Y-%m-%d-%H')  # This works and creates a Series with Date and Hour
  dfx2['date_hour'] = pd.to_datetime(dfx2['date_hour'] )
  dfx2.reset_index(drop=True, inplace=True)
  return 


# ##### Initiate the Process

# In[8]:

dfx2 = pd.DataFrame()
if not dfx2.empty:
  del dfx2    # Otherwise, the data will be concatenated on the existing df

parser = argparse.ArgumentParser()
parser.add_argument('--datetime_start',dest='datetime_start')
parser.add_argument('--datetime_end',dest='datetime_end')
args = parser.parse_args()

Initialize_and_Iterate()
Create_date_hour()


# In[83]:

def create_unistats(group):
    return {'min': group.min(), 'max': group.max(), 'count': group.count(), 'mean': group.mean()}
  
# These are DataFrames
bytes_dist = dfx_s['bytes'].groupby([dfx_s['date_hour'],   
                                  dfx_s['dst_port']]).apply(create_unistats).unstack().reset_index()
bytes_dist.rename(columns={'mean':'bytes_mean'}, inplace=True)
bytes_dist = bytes_dist.round(2)
bytes_dist = bytes_dist[(bytes_dist['dst_port']==53) ]   # all stats are displayed for just port 53

duration_dist = dfx_s['duration_ms'].groupby([dfx_s['date_hour'],   
                                  dfx_s['dst_port']]).apply(create_unistats).unstack().reset_index()


# In[84]:

bytes_dist.head(5)


# In[69]:

bytes_dist_s=bytes_dist.sort_values(['date_hour'],ascending=True)
bytes_dist_s


# ##### Write out the valid Time Series df

# In[85]:

def write_to_gbq():
  bytes_dist.to_gbq('prod.ts_port_53_test', "network-sec-analytics", verbose=True, reauth=False, 
   if_exists='replace', private_key=None)
  return
write_to_gbq()


