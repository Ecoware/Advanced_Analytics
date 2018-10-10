
# coding: utf-8

# In[2]:
#get_ipython().system(u'pip install pandasql')
# In[117]:
#!pip freeze
# In[3]:


import numpy as np
import pandas as pd

import pandas.io.gbq as pdg

from pandasql import PandaSQL 
pdsql = PandaSQL()

desired_width = 500
pd.set_option('display.width',desired_width)
pd.set_option('display.max_rows', 500)


from google.cloud import storage

#%%

import time
from labrador import labrador
lab = labrador.labrador()
tn = time.time()


## 207.11.1.213 and .161 are Proxy Servers

sql_query_eg_5tuple = "Select s.dstAd, count(s.dstAd) as dstAd_count from ipfix_5_min s inner join ext_list e on s.dstAd = e.dstAd group by s.dstAd order by dstAd_count desc"


proxy_juniper_5_10_23 = lab.fetch('ipfix_5_min',['2018-03-19T09:00:00','2018-03-19T09:10:00'],'',0,sql_query_eg_5tuple,'L')

print "That took ",time.time() - tn,"seconds OR ",(time.time() - tn)/3600,"hours"

proxy_juniper_5_10_23
#%%
match1 = pdsql("""Select s.dstAd, count(s.dstAd) as dstAd_count 
              from proxy_juniper_5_10_23 s
              inner join ext_list e
                 on s.dstAd = e.dstAd
              group by s.dstAd
              order by dstAd_count desc""", locals()) 

match1
#%%

## Read from GCS
def ReadFromGCS():
    global proxy_juniper_5_10_23
    storage_client = storage.Client(project='network-sec-analytics')
    bucket = storage_client.get_bucket('juniper_logs')
    blob = bucket.get_blob('5_10_23_2018')
    
    ## This creates a csv locally
    blob.download_to_filename('/home/steve/Data/RawData/5_10_23_2018')
    
    
    ## Read a local file into a Dataframe
    proxy_juniper_5_10_23= pd.DataFrame() 
    proxy_juniper_5_10_23 = pd.read_csv('/home/steve/Data/RawData/5_10_23_2018',sep=',')  
    proxy_juniper_5_10_23.info()
    return()

ReadFromGCS()

#%%

## frac is the final sample count as a percentage of the full extracted file
#sample = proxy_juniper_5_10_23.sample(frac=0.75, replace=True)
sample = proxy_juniper_5_10_23.sample(frac=1.00, replace=True)
#sample2 = proxy_juniper_5_10_23.sample(frac=0.5, replace=True)

sample.info()
proxy_juniper_5_10_23.info()

#%%
### This was attempted after the deciles were created but never finished. DO NOT USE
#raw_ext = (proxy_juniper_5_10_23.loc[(proxy_juniper_5_10_23['dstAd'].isin([ext_list]))])
## This works however 
raw_ext = (proxy_juniper_5_10_23.loc[(proxy_juniper_5_10_23['dstAd'].isin(['52.48.97.225']))])
raw_ext.to_csv('/home/steve/Data/RawData/raw_ext.csv', index=False)
#raw_ext.info()
#%%

# In[123]:


### The pdsql was killing the kernel above a 50% sample. Even for a 50% sample, this is much faster than the sql
### Count the number of times each unique dstAd was contacted by the Proxy Server.  THIS IS NOT A COUNT DISTINCT.
### Sort by Count in Descending order.  This enables the Rank to create the arbitary percentile groups (e.g., Deciles)
view_raw1 = sample.groupby(['srcAd','dstAd']).size().reset_index(name='dstAd_cnt')
view_raw1.sort_values(by='dstAd_cnt', ascending=False, inplace=True)
view_raw1.loc[(view_raw1['dstAd'].isin(['52.48.97.225']))]

# In[124]:

#view_raw1[0:25]
  
## Some random search specs
#view_raw1.loc[(view_raw1['rank'].isin([21090,21091,21092]))]
#view_raw1.loc[(view_raw1['dstAd_cnt'].isin([70]))]
sample_raw_ext = view_raw1.loc[(view_raw1['dstAd'].isin(['52.48.97.225']))]
sample_raw_ext
sample_raw_ext.to_csv('/Users/swe03/sample_raw_ext.csv', index=False)

# In[125]:


## method='first' uses the sorted order from the dstAd_cnt above to assign a sequential nbr, starting from 24989 and ending
## at 1.
## That is, 27738 unique dstAd's and, for each unique dstAd, the Count of the number of times it was contacted from the src_ip
view_raw1['rank'] = view_raw1['dstAd_cnt'].rank(method='first') 
view_raw1[0:10]


# In[126]:


## Create the arbitrary quantiles(i.e., qcut).  Use .values,10 for Deciles
## This will create quantiles for the dstAd Counts(i.e., dstAd_cnt) associated with each unique dstAd
view_raw1['rank_int'] = pd.qcut(view_raw1['rank'].values,10)
view_raw1[0:12]


# In[128]:


## Need to convert the rank_int from data type Category to String
view_raw1['rank_int'] = view_raw1.rank_int.astype(str) 
view_raw1['rank'] = view_raw1['rank'].astype(int)
view_raw1.info()

# In[129]:

## This is only informative about confirming the approx equal interval sizes from the qcut
## Note:  rank_int is the rank_interval NOT int (integer)
## Note:  rank_int is not in sorted order
view_dec = pdsql("""Select srcAd,  rank_int,  count(rank_int) as rank_int_cnt
              from view_raw1
              group by rank_int
              order by rank_int desc""", locals()) 
view_dec

# In[130]:


#str[1] will only retain the second qualifying string. Thus, str[0] is the "(" and str[1] will return the interval start value
# BgnBinR was a string so needed to convert to numeric (int did not work)
view_dec['BgnBinR'] = view_dec.rank_int.str.split(r'\s*\(\s*|\s*\,\s*').str[1]
view_dec['BgnBinR'] = pd.to_numeric(view_dec['BgnBinR'], errors='coerce')
view_dec.sort_values('BgnBinR',ascending=False)


# In[131]:


## Join the raw view(get the counts and ranks) with the decile view to get the "start" Bin value.
## BgnBinR is Beginning Bin Rank
view_dec_dst = pdsql("""Select d.srcAd, r.dstAd, r.dstAd_cnt, r.rank, d.rank_int,  d.BgnBinR
              from view_raw1 r
              inner join view_dec d
                 on r.rank_int = d.rank_int
              order by r.dstAd_cnt desc""", locals()) 


# In[132]:


view_dec_dst.info()
view_dec_dst[0:10]
view_dec_dst.loc[(view_dec_dst['dstAd'].isin(['52.48.97.225']))]


# In[133]:


## Count is the number of Unique dstAd's IN EACH INTERVAL and, thus, Count column will SUM UP to 27738 which is the largest
## ENDING RANK BOUNDARY OF THE FIRST INTERVAL 
## The Univariate Statistics(e.g., mean, std, etc.) describes the set of the dstAd_cnt's within each quantile.
## 
view_dec_dst.groupby('BgnBinR')['dstAd_cnt'].describe()


# In[134]:


### The "head" 5 for each Interval will have the top 5 largest values for dstAd_cnt. The first Interval, because of large numbers,
###   will have more uniqueness in the TOP 5 as opposed to the lower decile which will have smaller numbers and more
###   duplicates as we see below. 
## What is the N first dstAds in each group

#view_dec_dst.groupby('BgnBinR').first()
#view_dec_dst.groupby('BgnBinR').head(5)


# In[135]:


### This is just a check of the dstAd_cnt above(i.e., the first dstAd in the 19416.9 beginning quantile boundard )
### As expected, this extracts 46 records
audit = (sample.loc[(sample['dstAd'].isin(['23.199.13.178']))])
#audit.info()
audit.sort_values(by='flStDttmUtc')  ## Just to confirm multiple recs per same timestamp
#audit.reset_index(drop=True)  ## reset index to confirm count of recs


# In[136]:


## Create the sample extract. This doesn't filter recs but only retains 3 features for the join later on
samp_ext = sample.loc[:,['flStDttmUtc','srcAd','dstAd']]
samp_ext.sort_values(by=['flStDttmUtc','srcAd','dstAd'],inplace=True)
samp_ext.info()
samp_ext[0:5]


# In[137]:


## This will extract only those records (i.e., unq dstAd's counts) associated with a single bin and only keep 3 columns  
## This extract is for the subsequent join to the sample extract
view_dd_ext = view_dec_dst.loc[(view_dec_dst['BgnBinR'] == (23243.5)),['dstAd','dstAd_cnt','BgnBinR']]
view_dd_ext.sort_values(by=['dstAd','dstAd_cnt'],ascending=False, inplace=True)
view_dd_ext.info()
view_dd_ext.to_csv('/Users/swe03/view_dd_ext.csv', index=False)

#%%
ext_list = view_dec_dst.loc[(view_dec_dst['BgnBinR'] == (10324.6)),['dstAd']]

#ext2 = ext_list.to_string(index=False)
ext3 = ext_list.set_index('dstAd')
ext4 = ext3.to_string(index=False)
print(ext4)
ext4[0:5]
print(ext2)
type(ext2)

#%%


# In[138]:


## This is all 10 Deciles (i.e., 2774 x 10)
## Create the quantile distribution extract. This doesn't filter recs but only retains 3 features for the join later on
view_dec_dst_ext = view_dec_dst.loc[:,['dstAd','dstAd_cnt','BgnBinR']]
view_dec_dst_ext.info()
view_dec_dst_ext

#%%
view_dec_dst_ext.to_csv('/Users/swe03/view_dec_dst_ext.csv', index=False)
#%%
# In[139]:


## Merge the sample(i.e., raw data with datetime) with the view_dec_dst to get the quantile BgnBinR
## BgnBinR is Beginning Bin Rank
# Merge performs an INNER JOIN by default
# The INNER JOIN with samp_ext will create the duplicates and is why bin_profiles have 3+ million records
bin_profiles = pd.merge(samp_ext, view_dec_dst_ext, on='dstAd')
bin_profiles.sort_values(by=['BgnBinR','flStDttmUtc'])
bin_profiles.info()
bin_profiles[0:10]


# In[140]:


## Without the subset=[] option in drop_duplicates(), all columns are used in the dedup
## Now, for each 5 min agg and quantile bin, there is a count(dstAd_cnt) of the number of times 
## the srcAd connected to the dstAd 
bin_profiles = bin_profiles.drop_duplicates()
bin_profiles.sort_values(by=['BgnBinR','flStDttmUtc'])
#bin_profiles.sort_values(by=['flStDttmUtc'])
bin_profiles.info()
bin_profiles[0:100]


# In[141]:


## Can now subset each quantile bin by the 5 min agg and the SUM of all dstAd's dstAd_cnt's 

#view_5min = bin_profiles.loc[(bin_profiles['BgnBinR'] == 2494.900),['flStDttmUtc','dstAd','dstAd_cnt']]
#view_5min = bin_profiles.loc[(bin_profiles['BgnBinR'] == 22440.700),['flStDttmUtc','dstAd_cnt']]

#view_5min = bin_profiles.loc[(bin_profiles['BgnBinR'] <= 1.0)]   ## Hack to get the 0.999 bin. Unclear why == .999 didn't work
#view_5min = bin_profiles.loc[(bin_profiles['BgnBinR'] == 2774.7)]
#view_5min = bin_profiles.loc[(bin_profiles['BgnBinR'] == 5548.4)]
#view_5min = bin_profiles.loc[(bin_profiles['BgnBinR'] == 8322.1)]
view_5min = bin_profiles.loc[(bin_profiles['BgnBinR'] == 10331.0)]
#view_5min = bin_profiles.loc[(bin_profiles['BgnBinR'] == 13869.5)]
#view_5min = bin_profiles.loc[(bin_profiles['BgnBinR'] == 16643.2)]
#view_5min = bin_profiles.loc[(bin_profiles['BgnBinR'] == 19416.9)]
#view_5min = bin_profiles.loc[(bin_profiles['BgnBinR'] == 19909.8)]
#view_5min = bin_profiles.loc[(bin_profiles['BgnBinR'] == 24964.3)]

#view_5min.groupby(['flStDttmUtc']).size()
view_5min.groupby('flStDttmUtc').sum()
#view_5min.sort_values(by='flStDttmUtc', ascending=True, inplace=True)
view_5min.sort_values(by='dstAd_cnt', ascending=True, inplace=True)
view_5min.reset_index(drop=True)


#%%
view_5min = pdsql("""Select flStDttmUtc, sum(dstAd_cnt) as dstAd_cnt_sum
              from bin_profiles
              where BgnBinR == 10331.0
                and flStDttmUtc >= '2018-03-19 12:55:00'
              group by flStDttmUtc
              order by flStDttmUtc""", locals()) 

#%%
view_5min.head(10)

#%%
# In[142]:

## NOTE:  ONLY RUN THIS FOR THE SMALLEST QUANTILE 
## Note that for the quantile that is less than 1 (i.e., 0.999) the where clause needs to be <= 1.0
def view_last_quantile():
    global view_5min
    view_5min = pdsql("""Select flStDttmUtc, sum(dstAd_cnt) as dstAd_cnt_sum
                  from bin_profiles
                  where BgnBinR <= 1.0
                    and flStDttmUtc >= '2018-03-19 12:55:00'
                  group by flStDttmUtc
                  order by flStDttmUtc""", locals()) 
    return()
#view_last_quantile()

#%%
#####  I only need the next two sections since I needed to restart the kernel
#####  and did not want to lose the view_5min df
view_5min.to_csv('/Users/swe03/view_5min')

#%%

## Read this local file into a Dataframe
view_5min = pd.DataFrame() 
view_5min = pd.read_csv('/Users/swe03/view_5min',sep=',')  
#view_5min.info()
# In[143]:


import matplotlib.pyplot as pplt

from matplotlib.pylab import rcParams

rcParams['figure.figsize'] = 15, 6

from datetime import datetime, timedelta

desired_width = 250
pd.set_option('display.width',desired_width)
pd.set_option('display.max_rows', 500)

pplt.plot(view_5min['dstAd_cnt_sum'] , c = 'b', label = 'count')

pplt.xticks(range(len(view_5min['flStDttmUtc'])),view_5min['flStDttmUtc'])

pplt.xticks(rotation=80)
pplt.title('BgnBinR = 10331.0, dstAd_cnt_sums for ~2 hours (9:00 am - 11:00 am)')

pplt.show()


# In[ ]:


### 
#where flStDttmUtc between('2018-03-19 17:15:00') and ('2018-03-19 17:25:00') 
#where dstAd in ('172.21.108.22')

view_raw2 = pdsql("""Select * from proxy_juniper_5_10_23 
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


### raws1 above was created like:
### view_raw1 = sample.groupby(['srcAd','dstAd']).size().reset_index(name='dstAd_cnt')
view_raw1[0:10]


# In[ ]:


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

