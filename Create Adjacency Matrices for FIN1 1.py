
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

import pandas.io.gbq as pdg

from pandasql import PandaSQL 
pdsql = PandaSQL()

desired_width = 500
pd.set_option('display.width',desired_width)
pd.set_option('display.max_rows', 500)

from StringIO import StringIO


# #  READ GCS FILES via Labrador;

# ### Read from the 5 minute aggregates csv file buckets (without Labrador)

# In[2]:


#########################################################
## For each of the 5 minute aggregations, the data is randomly distributed across the "parts".
#########################################################
def skip():
  df_final = pd.DataFrame() 
  #data_var = pd.DataFrame()
  bucket = "gs://ipfix-pre-processed/unq-rel-cnts-voltage/2017_11_01.csv"  
  get_ipython().run_line_magic('storage', 'read --object $bucket --variable data_var')
  df_final = pd.read_csv(StringIO(data_var),sep=',')  
    


# ### Read from an existing GCS bucket

# In[3]:


### 1.5 million records for one day
def skip():
  df_gcs_ext = pd.DataFrame() 
  bucket = "gs://fin1_ipfix/dfx_distinct/2017_10_25.csv"  
  get_ipython().run_line_magic('storage', 'read --object $bucket --variable data_var')
  df_gcs_ext = pd.read_csv(StringIO(data_var),sep=',')   


# In[4]:


#df_gcs_ext.info()
#del df_gcs_ext


# In[5]:


### Never ran this so I'm not sure it works
def sqlq():
  dfx = pdsql("""Select date_hour, srcAd, dstAd, dstPt 
                              from df_gcs_ext 
                              where date_hour between ('2017-10-25 07:00:00.000000' and '2017-10-25 07:05:00.000000')
                              and dstAd in ('165.130.217.229','165.130.217.230')    
                              order by date_hour, srcAd, dstAd, dstPt """, locals()) 


# ### Read from the 5 minute aggregates using Labrador

# In[2]:


### This 5 minute extract returned 15.7 K records
import time
from labrador import labrador
lab = labrador.labrador()
#sql_query = sql_query = "select distinct substring(flStDttmUtc,1,13) as date_hour, srcAd, dstAd, dstPt from ipfix_5_min where dstAd in ('165.130.144.80','165.130.144.81','165.130.144.82','165.130.144.83','165.130.144.84','165.130.144.85') order by date_hour"
sql_query = sql_query = "select flStDttmUtc, srcAd, dstAd, dstPt from ipfix_5_min where dstAd in ('165.130.144.80','165.130.144.82','165.130.144.83') order by flStDttmUtc"
tn = time.time()
##dfx_distinct = lab.fetch('ipfix_5_min',['2017-10-25T00:00:00','2017-10-25T00:05:00'],sql_query,'L')  ## This is just 5 minutes during Daylight Savings Time
dfx_distinct_3ds = lab.fetch('ipfix_5_min',['2017-11-25T07:00:00','2017-11-25T07:05:00'],sql_query,'L')
print "That took ",time.time() - tn,"seconds OR ",(time.time() - tn)/3600,"hours"


# In[32]:


lab.log


# In[8]:


#dfx_distinct_3ds.sort_values(by=['srcAd','flStDttmUtc'])


# In[3]:


## Change from Int to Str for the concatenation below
dfx_distinct_3ds['dstPt'] = dfx_distinct_3ds.dstPt.astype(str)


# In[4]:


dfx_distinct_3ds.info()


# In[41]:


### Just a view into the data with a Date Hour Count
#view1 = pdsql("""Select distinct flStDttmUtc, count(flStDttmUtc) as date_hour_cnt from dfx_distinct_3ds group by flStDttmUtc""", locals()) 


# In[42]:


#lab.write_pandas_to_csv_on_gcs(bucket='fin1_ipfix' ,data=unq_rel_cnts ,fileName='unq_rel_cnts/2017_10_30-11_26.csv')


# In[5]:


### Concatenate the Dest IP and the Dest Port into a single variable (dstPt is now a string)
dfx_distinct_3ds['dstAd_Pt'] = dfx_distinct_3ds['dstAd'].astype(str) + ' :' + dfx_distinct_3ds['dstPt'] 


# In[5]:


dfx_distinct_3ds[0:10]


# ### Create various Adjacency Matrixes

# In[16]:


## Select only one hour.  Otherwise the values will be greater than 1 since the relations are only distinct by the hour.
## This is only relevant for a matrix with 1's and 0's
#dfx_distinct_3ds = dfx_distinct_3ds.loc[dfx_distinct_3ds['flStDttmUtc'] == '2017-11-25 11:55:00']
#dfx_distinct_3ds = dfx_distinct_3ds[['srcAd','dstAd_Pt']]


# In[15]:


### Cross Tabulation of the Frequencies by flStDttmUtc, srcAd, dstAd and the cells are the unique dstPt counts
### This is just informational, not to create the Adjacency Matrix

#dfx_ct1 = pd.crosstab([dfx_distinct_3ds.flStDttmUtc,dfx_distinct_3ds.srcAd],dfx_distinct_3ds.dstAd,margins=True)  ## This creates a dataframe
#dfx_distinct_3ds = dfx_distinct_3ds[['srcAd','dstAd_Pt']]
#dfx_ct1


# In[7]:


### Cross Tabulation of the Frequencies by flStDttmUtc, srcAd, dstAd and the cells are the unique dstPt counts
### This creates the Adjacency Matrix

dfx_ct1 = pd.crosstab([dfx_distinct_3ds.srcAd],dfx_distinct_3ds.dstAd,margins=False)  ## This creates a dataframe
#dfx_distinct_3ds = dfx_distinct_3ds[['srcAd','dstAd_Pt']]
#dfx_ct1


# In[8]:


### Create a small view of the dfx_ct1
dfx_ct1 = dfx_ct1.loc['165.130.128.25':'165.130.221.20','165.130.144.80':'165.130.144.83']
dfx_ct1


# In[9]:


### Cross Tabulation of the Frequencies by flStDttmUtc, srcAd, dstAd_Pt and the cells are the unique srcAd by (dstAd and dstPt) counts
### This creates an Adjacency Matrix
#dfx_ct2 = pd.crosstab(dfx_distinct_3ds.srcAd,dfx_distinct_3ds.dstAd_Pt)  ## This creates a dataframe. Use margins=True to the sum of 1's


# In[9]:


## Create the Adjacency Matrix
idx = dfx_ct1.columns.union(dfx_ct1.index)


# In[10]:


## Create the Adjacency Matrix
idx = dfx_ct1.columns.union(dfx_ct1.index)
dfx_ct1 = dfx_ct1.reindex(index = idx, columns=idx, fill_value=0)


# In[11]:


from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt


# In[ ]:


# This needs more work. The Adjacency Matrix or CrossTab Matrix does not conform to the n x m shape or even logically !
#print dfx_ct1.shape  # n samples with m dimensions
#plt.scatter(dfx_ct1.loc['165.130.144.83'],dfx_ct1.loc['10.66.34.33'])
#plt.show()


# In[12]:


import networkx as nx
G = nx.Graph()
G = nx.from_pandas_adjacency(dfx_ct1)
G.name = 'Graph from dfx_ct1 adjacency matrix dataframe'


# In[13]:


print(nx.info(G))


# In[15]:


H = nx.DiGraph(G)


# In[16]:


print(nx.info(H))


# In[ ]:


nx.draw_spectral(H)


# In[17]:


from networkx.algorithms import approximation


# In[18]:


from networkx.algorithms import community


# In[19]:


communities_generator = community.girvan_newman(G)


# In[20]:


top_level_communities = next(communities_generator)
top_level_communities


# In[72]:


next_level_communities = next(communities_generator)
next_level_communities


# In[54]:


approximation.k_components(G)


# In[55]:


## This crashes the session
approximation.max_clique(G)


# In[86]:


#lab.write_pandas_to_csv_on_gcs(bucket='swe-files' ,data=dfxtest2 ,fileName='swe-files/dfxtest2.csv')


# In[32]:


##### Write the dataframe to BQ.
#dfxtest2.to_gbq('ConnectionModeling.dfxtest2', "network-sec-analytics", verbose=True, reauth=False, 
  #if_exists='replace', private_key=None)

