
# coding: utf-8

# In[1]:


#!pip install pandasql


# In[2]:


#!pip freeze


# In[15]:


import numpy as np
import pandas as pd

import pandas.io.gbq as pdg

from pandasql import PandaSQL 
pdsql = PandaSQL()

desired_width = 500
pd.set_option('display.width',desired_width)
pd.set_option('display.max_rows', 500)


# #  READ GCS FILES via Labrador;

# In[16]:


#!pip list | grep networkx


# In[17]:


import logging
logger = logging.getLogger()
logger.setLevel(logging.CRITICAL)


# In[6]:


### This block is for Servers that typically Request Services
### 207.11.1.213 is a Proxy Server that will typically request services via a Src Ad ephemeral port

import time
from labrador import labrador
lab = labrador.labrador()
tn = time.time()

### Egress -  Server scope as Source IP 
### 207.11.1.213  -  Proxy Server IP requesting 443 Service (i.e., web pages)
#sql_query_eg_5tuple = "Select date_hour, srcAd, dstAd, dstPt, proto, count(*) as distinct_freq \
#  from (select distinct substring(flStDttmUtc,1,13) as date_hour, srcAd, srcPt, dstAd, dstPt, proto \
#  from ipfix_5_min where srcAd in ('207.11.1.213') ) \
#  group by date_hour, srcAd, dstAd, dstPt, proto order by date_hour"

## 207.11.1.213 and .161 are Proxy Servers
#sql_query_eg_5tuple = "Select * from ipfix_5_min where srcAd in ('207.11.1.213')"


#sql_query_eg_5tuple = "Select * from ipfix_5_min where srcAd in ('207.11.1.145')"
sql_query_eg_5tuple = "Select * from ipfix_5_min where srcAd in ('207.11.1.161')"
#sql_query_eg_5tuple = "Select * from ipfix_5_min where dstAd in ('172.21.108.22')"
#sql_query_eg_5tuple = "Select * from ipfix_5_min where srcAd in ('207.11.1.161') and dstAd in ('172.21.108.22')"
#sql_query_eg_5tuple = "Select * from ipfix_5_min where srcAd in ('172.21.108.22') and dstAd in ('52.84.32.157')"   
#sql_query_eg_5tuple = "Select * from ipfix_5_min where srcAd in ('52.84.32.157') and dstAd in ('172.21.108.22')"    
#sql_query_ig_5tuple = "Select * from ipfix_5_min where srcAd in ('192.161.147.7')"

### Ingress - Server scope as Destination IP
#sql_query_ig_5tuple = "Select date_hour, srcAd, srcPt, dstAd, proto, count(*) as distinct_freq \
#  from (select distinct substring(flStDttmUtc,1,13) as date_hour, srcAd, srcPt, dstAd, dstPt, proto \
#  from ipfix_5_min where dstAd in ('165.130.144.80','165.130.144.81','165.130.144.82') ) \
#  group by date_hour, srcAd, srcPt, dstAd, proto order by date_hour"



raw_161_eg_5t = lab.fetch('ipfix_5_min',['2018-03-19T09:00:00','2018-03-19T10:55:00'],'',0,sql_query_eg_5tuple,'L')
#raw_145_eg_5t  = lab.fetch('ipfix_5_min',['2018-03-19T10:00:00','2018-03-19T15:55:00'],'',0,sql_query_eg_5tuple,'L')
#raw_161_eg_5t  = lab.fetch('ipfix_5_min',['2018-03-19T10:00:00','2018-03-19T15:55:00'],'',0,sql_query_eg_5tuple,'L')
#raw_52172_eg_5t  = lab.fetch('ipfix_5_min',['2018-03-19T10:00:00','2018-03-19T15:55:00'],'',0,sql_query_eg_5tuple,'L')
#raw_17252_eg_5t  = lab.fetch('ipfix_5_min',['2018-03-19T10:00:00','2018-03-19T15:55:00'],'',0,sql_query_eg_5tuple,'L')

#raw_213_eg_5t = lab.fetch('ipfix_5_min',['2018-01-22T05:00:00','2018-01-22T05:55:00'],'',0,sql_query_eg_5tuple,'L')
#cnt_213_eg_5t = lab.fetch('ipfix_5_min',['2018-01-21T00:00:00','2018-01-27T23:55:00'],'',0,sql_query_eg_5tuple,'L')
#cnt_213_eg_5t = lab.fetch('ipfix_5_min',['2018-01-22T00:00:00','2018-01-22T23:55:00'],'',0,sql_query_eg_5tuple,'L')
#cnt_8012_ig_5t = lab.fetch('ipfix_5_min',['2017-11-25T00:00:00','2017-11-25T23:55:00'],'',0,sql_query_ig_5tuple,'L')


#dfx_distinct_8012 = lab.fetch('ipfix_5_min',['2017-11-25T08:00:00','2017-11-25T12:00:00'],'',0,sql_query,'L')
#unq_con_cnt_8012 = lab.fetch('ipfix_5_min',['2017-11-25T08:00:00','2017-11-25T12:00:00'],'',0,sql_query,'L')
print "That took ",time.time() - tn,"seconds OR ",(time.time() - tn)/3600,"hours"


# In[8]:

### This block is for Servers that typically Provides a Service

import time
from labrador import labrador
lab = labrador.labrador()
tn = time.time()

### Egress -  Server scope as Source IP 
sql_query_eg_5tuple = "Select date_hour, srcAd, dstAd, dstPt, proto, count(*) as distinct_freq   from (select distinct substring(flStDttmUtc,1,13) as date_hour, srcAd, srcPt, dstAd, dstPt, proto   from ipfix_5_min where dstAd in ('207.11.1.145') )   group by date_hour, srcAd, dstPt, dstAd, proto order by date_hour"

### Ingress - Server scope as Destination IP
#sql_query_ig_5tuple = "Select date_hour, srcAd, dstAd, dstPt, proto, count(*) as distinct_freq \
#  from (select distinct substring(flStDttmUtc,1,13) as date_hour, srcAd, srcPt, dstAd, dstPt, proto \
#  from ipfix_5_min where dstAd in ('165.130.128.25') ) \
#  group by date_hour, srcAd, dstAd, dstPt, proto order by date_hour"


cnt_145_eg_5t = lab.fetch('ipfix_5_min',['2018-03-19T13:00:00','2018-03-19T14:55:00'],'',0,sql_query_eg_5tuple,'L')
#cnt_25_eg_5t = lab.fetch('ipfix_5_min',['2017-11-25T00:00:00','2017-11-25T23:55:00'],'',0,sql_query_eg_5tuple,'L')
#cnt_25_ig_5t = lab.fetch('ipfix_5_min',['2017-11-25T00:00:00','2017-11-25T23:55:00'],'',0,sql_query_ig_5tuple,'L')


#dfx_distinct_8012 = lab.fetch('ipfix_5_min',['2017-11-25T08:00:00','2017-11-25T12:00:00'],'',0,sql_query,'L')
#unq_con_cnt_8012 = lab.fetch('ipfix_5_min',['2017-11-25T08:00:00','2017-11-25T12:00:00'],'',0,sql_query,'L')
print "That took ",time.time() - tn,"seconds OR ",(time.time() - tn)/3600,"hours"


# In[6]:


## 

raw_161_eg_5t.info()
#raw_161_ig_5t.info()
#cnt_145_eg_5t.info()
#raw_145_eg_5t.info()


# In[ ]:


https1 = pdsql("""Select date_hour, srcAd, dstAd, dstPt, proto, sum(distinct_freq) as totals from cnt_145_eg_5t 
                  where date_hour > '2018-03-19 16'
                  group by date_hour, srcAd, dstAd, dstPt, proto
                  order by date_hour, dstAd, dstPt """, locals()) 
https1[0:10]


# In[ ]:


sorted = cnt_213_eg_5t.loc[(cnt_213_eg_5t.dstAd == '52.170.84.244')]
sorted.sort_values('date_hour')


# In[23]:


lab.write_pandas_to_csv_on_gcs(bucket='fin1_ipfix' ,data=raw_213_eg_5t ,fileName='raw_213_eg_5t.csv')


# In[ ]:


lab.write_pandas_to_csv_on_gcs(bucket='fin1_ipfix' ,data=cnt_25_ig_5t ,fileName='cnt_25_ig_5t.csv')


# In[9]:


## Wrote out 4.3.2018
lab.write_pandas_to_csv_on_gcs(bucket='swe-files' ,data=raw_161_eg_5t ,fileName='raw_161_eg_5t')


# In[18]:


from datalab.context import Context
import datalab.storage as storage
from StringIO import StringIO

import gcp 
import gcp.storage as storage
# In[ ]:


cnt_25_eg_5t = pd.DataFrame() 
data_var = pd.DataFrame()
bucket = "gs://fin1_ipfix/cnt_25_eg_5t.csv"  
get_ipython().magic(u'storage read --object $bucket --variable data_var')
cnt_25_eg_5t = pd.read_csv(StringIO(data_var),sep=',')  


# In[ ]:


cnt_25_ig_5t = pd.DataFrame() 
data_var = pd.DataFrame()
bucket = "gs://fin1_ipfix/cnt_25_ig_5t.csv"  
get_ipython().magic(u'storage read --object $bucket --variable data_var')
cnt_25_ig_5t = pd.read_csv(StringIO(data_var),sep=',')  


# In[19]:


raw_161_eg_5t = pd.DataFrame() 
data_var = pd.DataFrame()
bucket = "gs://swe-files/raw_161_eg_5t"  
get_ipython().magic(u'storage read --object $bucket --variable data_var')
raw_161_eg_5t = pd.read_csv(StringIO(data_var),sep=',')  


# #### Start of processing

# In[ ]:
####


## and dstPt in (443,53)
view1 = pdsql("""Select * from cnt_25_eg_5t 
                  where date_hour between ('2017-11-25 00') and ('2017-11-25 24')
                    
                    and srcAd = '165.130.128.25'
                  order by srcPt, date_hour, distinct_freq """, locals()) 
view1[0:10]


# In[ ]:


## 
view_raw1 = pdsql("""Select srcAD, srcPt, dstAd, dstPt, proto, flCnt, smOcts, smPkts, smECNEcho, smURG, smACK, smPSH, smRST, 
                     smSYN, smFIN, flStDttmUtc   
                     from raw_161_eg_5t
                     
                  order by srcAd """, locals()) 
view_raw1


# In[ ]:


## 
view_raw1 = pdsql("""Select srcAD, srcPt, dstAd, dstPt, proto, flCnt, smOcts, smPkts, smECNEcho, smURG, smACK, smPSH, smRST, 
                     smSYN, smFIN, flStDttmUtc   
                     from sample1 order by srcAd """, locals()) 
view_raw1


# In[20]:


##sample = raw_161_eg_5t.sample(frac=0.25, replace=True)
sample1 = raw_161_319_0001.sample(frac=0.001, replace=True)
sample1.info()



####

view_raw1
     .to_csv('/User/swe03/sample.csv')
# In[21]:


##### where srcAd like '207.%'
view_raw1 = pdsql("""Select srcAD,  dstAd,  count(dstAd) as dstAd_cnt
                     from sample    
                  group by  dstAD
                  order by dstAd_cnt desc""", locals()) 
#%%

view_raw1[0:10]  


# In[22]:


#view_raw1.groupby(pd.qcut(view_raw1.dstAd_cnt,2))['dstAd_cnt'].sum()
view_raw1['rank'] = view_raw1['dstAd_cnt'].rank(method='first') ## method='first' uses the sorted order from the dstAd_cnt
view_raw1[0:10]


# In[23]:


view_raw1['rank_int'] = pd.qcut(view_raw1['rank'].values,5)
view_raw1[0:10]


# In[24]:


view_raw1['rank_int'] = view_raw1.rank_int.astype(str)
#view_raw1.info()


# In[25]:


## Note:  rank_int is not in sorted order
view_dec = pdsql("""Select srcAd,  rank_int,  count(rank_int) as rank_int_cnt
              from view_raw1
              group by rank_int
              order by rank_int desc""", locals()) 
view_dec


# In[29]:


#str[1] will only retain the first qualifying string. Thus, str[1] will return the second string
# Bin was a string so needed to convert to numeric (int did not work)
view_dec['Bin'] = view_dec.rank_int.str.split(r'\s*\(\s*|\s*\,\s*').str[1]
view_dec['Bin'] = pd.to_numeric(view_dec['Bin'], errors='coerce')
view_dec.sort_values('Bin',ascending=False)


# In[30]:


view_dec_dst = pdsql("""Select d.srcAd, r.dstAd, r.dstAd_cnt, r.rank, d.rank_int,  d.Bin
              from view_raw1 r
              inner join view_dec d
                 on r.rank_int = d.rank_int
              order by r.dstAd_cnt desc""", locals()) 


# In[31]:


view_dec_dst[100:110]


# In[32]:


ext1 = raw_161_eg_5t.loc[(raw_161_eg_5t['dstAd'] == '74.119.119.84'),['flStDttmUtc','dstAd']]
#ext1 = ext1.sort_values(['dstAd','dstPt'],ascending=True)
ext1.info()


# In[ ]:


#where flStDttmUtc between('2018-03-19 17:15:00') and ('2018-03-19 17:25:00') 
#where dstAd in ('172.21.108.22')
#from sample
#from raw_161_eg_5t
#from ext1

view_52_5min = pdsql("""Select flStDttmUtc, dstAd,  count(dstAd) as dstAd_cnt
                     from raw_161_eg_5t  
                     where dstAd = '136.147.40.153'
                  group by  flStDttmUtc
                  order by flStDttmUtc""", locals()) 
view_52_5min[0:25]


# In[33]:


#where flStDttmUtc between('2018-03-19 17:15:00') and ('2018-03-19 17:25:00') 
#where dstAd in ('172.21.108.22')
#from sample
#from raw_161_eg_5t
#from ext1

view_52_5min = pdsql("""Select flStDttmUtc, dstAd,  count(dstAd) as dstAd_cnt
                     from ext1
                  group by  flStDttmUtc
                  order by flStDttmUtc""", locals()) 
view_52_5min


# In[34]:


import matplotlib.pyplot as pplt
get_ipython().magic(u'matplotlib inline')
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15, 6

from datetime import datetime, timedelta

desired_width = 250
pd.set_option('display.width',desired_width)
pd.set_option('display.max_rows', 500)

pplt.plot(view_52_5min['dstAd_cnt'] , c = 'b', label = 'count')

pplt.xticks(range(len(view_52_5min['flStDttmUtc'])),view_52_5min['flStDttmUtc'])

pplt.xticks(rotation=80)
pplt.title('Dest Addr: 52.2.253.3(dstAd) - frequencies for 2 hours')

pplt.show()


# In[14]:


## This has questionable value relative to the ranking above
view_raw1.loc[(view_raw1.dstAd_cnt == 1), 'DstGroup'] = '1'
view_raw1.loc[(view_raw1.dstAd_cnt > 1) & (view_raw1.dstAd_cnt <= 100) , 'DstGroup'] = '2-100'
view_raw1.loc[(view_raw1.dstAd_cnt > 101) & (view_raw1.dstAd_cnt <= 1000) , 'DstGroup'] = '101-1000'
view_raw1.loc[(view_raw1.dstAd_cnt > 1000) & (view_raw1.dstAd_cnt <= 10000) , 'DstGroup'] = '1001-10000'
view_raw1.loc[(view_raw1.dstAd_cnt > 10000) & (view_raw1.dstAd_cnt <= 20000) , 'DstGroup'] = '10000-20000'
view_raw1.loc[(view_raw1.dstAd_cnt > 20000) & (view_raw1.dstAd_cnt <= 30000) , 'DstGroup'] = '20000-30000'
view_raw1.loc[(view_raw1.dstAd_cnt > 30000), 'DstGroup'] = '30K+'


# In[19]:


view_raw1 = pdsql("""Select srcAD,  DstGroup,  count(DstGroup) as DstGroup_cnt
              from view_raw1
              group by DstGroup
              order by DstGroup_cnt desc""", locals()) 
view_raw1


# In[60]:


#where flStDttmUtc between('2018-03-19 17:15:00') and ('2018-03-19 17:25:00') 
#where dstAd in ('172.21.108.22')
view_raw2 = pdsql("""Select * from raw_145_eg_5t 
                   where dstAd like '172.21.%'            
                                
         order by dstAd                    """, locals()) 
view_raw2


# In[10]:


view_raw3 = pdsql("""Select * from raw_161_ig_5t 
                              where flStDttmUtc between('2018-03-19 17:00:00') and ('2018-03-19 17:45:00')  
                                and dstAd in ('207.11.1.161')
                              """, locals()) 


# In[12]:


view_raw3.info()


# In[ ]:


view_raw1[0:10]


# In[44]:


view_raw2[0:10]


# In[66]:


## count(dstPtCats) was just a frequency of the cat's ... lost the unique counts
## count(distinct_freq) was a freq of distinct port counts
raw_grps = pdsql("""Select dstPt, avg(flCnt) as avg_flCnts, round(avg(smOcts),0) as avg_smOcts, 
                      round(avg(smPkts),0) as avg_smPkts,
                      avg(smRST) as avg_RST
                  from view_raw1
                  group by dstPt
                  order by avg_smOcts DESC, dstPt """, locals()) 
raw_grps


# In[40]:


## count(dstPtCats) was just a frequency of the cat's ... lost the unique counts
## count(distinct_freq) was a freq of distinct port counts
raw_dst = pdsql("""Select flStDttmUtc, srcAD, dstAD, dstPt, count(dstAD) as dstAD_cnt
                  from view_raw2
                  where dstAd in ('192.161.147.7')
                  group by flStDttmUtc, srcAD, dstAD, dstPt 
                  order by dstAD_cnt desc  """, locals()) 
raw_dst


# In[48]:


## count(dstPtCats) was just a frequency of the cat's ... lost the unique counts
## count(distinct_freq) was a freq of distinct port counts
raw_dst1 = pdsql("""Select srcAD, srcPt, dstAd, dstPt, proto, flCnt, smOcts, smPkts, smECNEcho, smURG, smACK, smPSH, smRST, 
                     smSYN, smFIN, flStDttmUtc
                  from view_raw2
                  where dstAd in ('192.161.147.7')
                  and flStDttmUtc = '2018-03-19 17:30:00'
                  order by smOcts desc
                  """, locals()) 
raw_dst1


# In[13]:


## count(dstPtCats) was just a frequency of the cat's ... lost the unique counts
## count(distinct_freq) was a freq of distinct port counts
raw_dst1 = pdsql("""Select srcAD, srcPt, dstAd, dstPt, proto, flCnt, smOcts, smPkts, smECNEcho, smURG, smACK, smPSH, smRST, 
                     smSYN, smFIN, flStDttmUtc
                  from view_raw3
                  where dstAd in ('207.11.1.161')
                  and flStDttmUtc = '2018-03-19 17:30:00'
                  order by smOcts desc
                  """, locals()) 
raw_dst1


# In[89]:


pdsql("""Select flStDttmUtc,  count(flStDttmUtc) as fivemin_count 
  from view_raw1 
  group by flStDttmUtc 
  order by flStDttmUtc """, locals()) 


# In[ ]:


### Cross Tabulation of the Frequencies of raw_213_eg_5t by date_hour, srcAd, dstAd, dstPt
###
### This is for only the 165.130.144.82 dstAd and a single day/24 hour interval
#dfx82xtab = pd.crosstab([dfx82.date_hour,dfx82.srcAd],dfx82.dstAd,margins=True)  ## This creates a dataframe

### This is for only the 165.130.144.83 dstAd and a single day/24 hour interval
#dfx83xtab = pd.crosstab([dfx83.date_hour,dfx83.srcAd],dfx83.dstAd,margins=True)  ## This creates a dataframe

#dfx8012 = pd.crosstab([dfx_distinct_8012.flStDttmUtc,dfx_distinct_8012.srcAd,dfx_distinct_8012.srcPt],dfx_distinct_8012.dstAd,margins=True)  ## This creates a dataframe


# In[ ]:


### Cross Tabulation of the Frequencies of raw_213_eg_5t by date_hour, srcAd, dstAd, dstPt
###
### This is for only the 165.130.144.82 dstAd and a single day/24 hour interval
#dfx82xtab = pd.crosstab([dfx82.date_hour,dfx82.srcAd],dfx82.dstAd,margins=True)  ## This creates a dataframe

### This is for only the 165.130.144.83 dstAd and a single day/24 hour interval
#dfx83xtab = pd.crosstab([dfx83.date_hour,dfx83.srcAd],dfx83.dstAd,margins=True)  ## This creates a dataframe

#dfx8012 = pd.crosstab([dfx_distinct_8012.flStDttmUtc,dfx_distinct_8012.srcAd,dfx_distinct_8012.srcPt],dfx_distinct_8012.dstAd,margins=True)  ## This creates a dataframe
#ct8012_eg = pd.crosstab([view1.date_hour,view1.srcAd,view1.dstPtCats],view1.dstPt,margins=True)  ## This creates a dataframe


# In[85]:


dstAd_226 = view_raw1.loc[(view_raw1['dstAd'] == '40.121.180.226')]
ct_raw226 = pd.crosstab([dstAd_226.flStDttmUtc,dstAd_226.dstAd],dstAd_226.dstPt,margins=True)  ## This creates a dataframe
#ct_raw213 = pd.crosstab([view_raw1.flStDttmUtc,view_raw1.dstAd],view_raw1.dstPt,margins=True)  ## This creates a dataframe
ct_raw226


# In[80]:


#ct_raw213.info()
ct_raw213.sort_values(by="All",ascending=False)


# In[ ]:


spikes_eg = pd.crosstab([view1.date_hour,view1.srcAd],[view1.dstPt,view1.distinct_freq],margins=True)
spikes_eg


# In[ ]:


## Create the Src Port Categories
view1.loc[(view1.srcPt >= 0)     & (view1.srcPt <= 1023) , 'srcPtCats'] = 'SysPts 0-1023'
view1.loc[(view1.srcPt >= 1024)  & (view1.srcPt <= 49151), 'srcPtCats'] = 'RegPts 1024-49151'
view1.loc[(view1.srcPt >= 49152) & (view1.srcPt <= 65535), 'srcPtCats'] = 'EphPts 49152-65535'
#view1.loc[(view1.srcAd == '165.130.144.82') & (view1.srcPtCats =='RegPts 1024-49151' )]
view1[0:10]


# In[ ]:


## count(dstPtCats) was just a frequency of the cat's ... lost the unique counts
## count(distinct_freq) was a freq of distinct port counts
view1s = pdsql("""Select date_hour, srcAd, srcPtCats, sum(distinct_freq) as srcPtCat_cnts 
                  from view1
                  where srcAd = '165.130.128.25' 
                  group by date_hour, srcAd, srcPtCats
                  order by date_hour, srcAd """, locals()) 

view1s 


# In[ ]:


view1s_p = view1s.pivot(index='date_hour', columns='srcPtCats', values='srcPtCat_cnts')
view1s_p.reset_index(inplace=True)
view1s_p


# In[ ]:


view2 = pdsql("""Select * from cnt_25_ig_5t 
                  order by date_hour, srcAd, dstAd""", locals()) 
view2[0:10]


# In[ ]:


view2.loc[(view2.dstPt >= 0)     & (view2.dstPt <= 1023) , 'dstPtCats'] = 'SysPts 0-1023'
view2.loc[(view2.dstPt >= 1024)  & (view2.dstPt <= 49151), 'dstPtCats'] = 'RegPts 1024-49151'
view2.loc[(view2.dstPt >= 49152) & (view2.dstPt <= 65535), 'dstPtCats'] = 'EphPts 49152-65535'
#view2.loc[(view2.dstAd == '165.130.128.25') & (view2.dstPtCats =='SysPts 0-1023' )]
#view


# In[ ]:


## and dstPtCats in ('Ephemeral Ports','Registered Ports')
view2d = pdsql("""Select date_hour, dstAd, dstPtCats, sum(distinct_freq) as dstPtCat_cnts 
                  from view2
                  where dstAd = '165.130.128.25' 
                  group by date_hour, dstAd, dstPtCats
                  order by date_hour, dstAd """, locals()) 

view2d 


# In[ ]:


view2d_p = view2d.pivot(index='date_hour', columns='dstPtCats', values='dstPtCat_cnts')
view2d_p.reset_index(inplace=True)
view2d_p


# In[ ]:


import matplotlib.pyplot as pplt
get_ipython().magic(u'matplotlib inline')
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15, 6

from datetime import datetime, timedelta

desired_width = 250
pd.set_option('display.width',desired_width)
pd.set_option('display.max_rows', 500)

 
#E = view2.loc[(view2.srcPtCats == 'Ephemeral Ports')]
pplt.plot(view1s_p['SysPts 0-1023'] , c = 'b', label = 'SysPts 0-1023')

 
#R = view2.loc[(view2.srcPtCats == 'Registered Ports')]
#pplt.plot(view1s_p['RegPts 1024-49151'], c = 'r', label = 'RegPts 1024-49151')

#pplt.plot(view1s_p['EphPts 49152-65535'], c = 'g', label = 'EphPts 49152-65535')

pplt.xticks(range(len(view1s_p['date_hour'])),view1s_p['date_hour'])

pplt.xticks(rotation=80)
pplt.title('DNS: 165.130.128.25(srcAd) - Src Port frequencies for 11-25-2017 24 hours')

#pplt.autoscale(enable=True, axis='x', tight=None)
#pplt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
pplt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)
pplt.show()


# In[ ]:


import matplotlib.pyplot as pplt
get_ipython().magic(u'matplotlib inline')
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15, 6

from datetime import datetime, timedelta

desired_width = 250
pd.set_option('display.width',desired_width)
pd.set_option('display.max_rows', 500)

 
#E = view2.loc[(view2.srcPtCats == 'Ephemeral Ports')]
pplt.plot(view2d_p['SysPts 0-1023'] , c = 'b', label = 'SysPts 0-1023')

 
#R = view2.loc[(view2.srcPtCats == 'Registered Ports')]
#pplt.plot(view2d_p['RegPts 1024-49151'], c = 'r', label = 'RegPts 1024-49151')

#pplt.plot(view2d_p['EphPts 49152-65535'], c = 'g', label = 'EphPts 49152-65535')

pplt.xticks(range(len(view2d_p['date_hour'])),view2d_p['date_hour'])

pplt.xticks(rotation=80)
pplt.title('DNS: 165.130.128.25(dstAd) - Dst Port frequencies for 11-25-2017 24 hours')

#pplt.autoscale(enable=True, axis='x', tight=None)
#pplt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
pplt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)
pplt.show()


# In[ ]:


### Cross Tabulation of the Frequencies by date_hour, srcAd, dstAd, dstPt
###
### This is for only the 165.130.144.82 dstAd and a single day/24 hour interval
#dfx82xtab = pd.crosstab([dfx82.date_hour,dfx82.srcAd],dfx82.dstAd,margins=True)  ## This creates a dataframe

### This is for only the 165.130.144.83 dstAd and a single day/24 hour interval
#dfx83xtab = pd.crosstab([dfx83.date_hour,dfx83.srcAd],dfx83.dstAd,margins=True)  ## This creates a dataframe

#dfx8012 = pd.crosstab([dfx_distinct_8012.flStDttmUtc,dfx_distinct_8012.srcAd,dfx_distinct_8012.srcPt],dfx_distinct_8012.dstAd,margins=True)  ## This creates a dataframe
ct8012_eg = pd.crosstab([view1.date_hour,view1.srcAd,view1.dstPtCats],view1.dstPt,margins=True)  ## This creates a dataframe


# In[ ]:


ct8012_eg


# In[ ]:


#dfx82xtab2 = dfx82xtab.reset_index()   ## Flatten the multi-index to use date_hour in the sort
#dfx82xtab2.info()

#dfx83xtab2 = dfx83xtab.reset_index()   ## Flatten the multi-index to use date_hour in the sort

ct8012_eg = ct8012_eg.reset_index()   ## Flatten the multi-index to use date_hour in the sort
ct8012_eg.info()


# In[ ]:


## Apparently this is unecessary when the value is an int 
ct8012.rename(columns={"0":"_0", "22":"_22", "25":"_22", "53":"_53", "80":"_80", "123":"_123", "162":"_162", 
                         "443":"_443", "636":"_636", "9997":"_9997", "17472":"17472", "42091":"42091",
                         "51034":"_51034", "51051":"51051", "61613":"61613"},inplace=True) 


# In[ ]:


ct8012_eg


# In[ ]:


lab.write_pandas_to_csv_on_gcs(bucket='swe-files' ,data=ct8012 ,fileName='swe-files/ct8012.csv')


# In[ ]:


##### Write the dataframe to BQ.
dfxtest2.to_gbq('ConnectionModeling.dfxtest2', "network-sec-analytics", verbose=True, reauth=False, 
  if_exists='replace', private_key=None)

