
# coding: utf-8

# # Creation of an Analytical Data Set (ADS) for Probabilistic Classification Modeling of Click-Stream data

# ##### Import the BigQuery library for BigQuery SQL functions

# In[3]:

import gcp.bigquery as bq


# ##### Create a SQL module called bq_omn_requests to access the data

# In[4]:

get_ipython().run_cell_magic('sql', '--module bq_table', 'select post_visid, date_time, visit_num, visit_page_num, visit_start_time_gmt, hit_time_gmt, \n  channel, last_hit_time_gmt, page_event, page_event_var1, page_event_var2, page_type, pagename,  \n  prop1, prop30, evar4, evar10, evar26, evar30, event_list, product_list, cust_visid,\n  visit_dt \nfrom Modeling_Data.omniture_data_onedaysample_500K\norder by post_visid, visit_start_time_gmt, hit_time_gmt\nlimit 500')


# ##### Explore the result set - view some collection of records

# In[5]:

get_ipython().run_cell_magic('bigquery', 'sample --count 10 --query bq_table', '')


# ##### The code below constructs a BigQuery Query instance, executes the query, and converts the results into a DataFrame.  
# ##### The len statement counts the number of records

# In[6]:

df = bq.Query(bq_table).to_dataframe()
len(df)


# ##### Examine a few records from the DataFrame created above

# In[7]:

df.head(10)


# ##### -----------------------------------------------------------------------------------------------------------------
# #####      Start the processing to create the Analytical Data Set                                                
# ##### -----------------------------------------------------------------------------------------------------------------

# ##### Install the SQL package (does not come pre-installed)

# In[8]:

get_ipython().run_cell_magic('bash', '', 'pip install pandasql  ')


# ##### Import some additional libraries

# In[9]:

import numpy as np
import pandas as pd
from pandasql import PandaSQL 


# ##### Create a small development sample from the BigQuery table above

# In[10]:

get_ipython().run_cell_magic('sql', '--module bq_table_devsample', 'select post_visid, date_time, visit_num, visit_page_num, visit_start_time_gmt, hit_time_gmt, \n  channel, last_hit_time_gmt, page_event, page_event_var1, page_event_var2, page_type, pagename,  \n  prop1, prop30, evar4, evar10, evar26, evar30, event_list, product_list, cust_visid,\n  visit_dt \nfrom Modeling_Data.omniture_data_onedaysample_500K\norder by post_visid, visit_start_time_gmt, hit_time_gmt\nlimit 10000')


# ##### Create a Dataframe from the development sample and confirm the sample size

# In[11]:

ClickRecs = bq.Query(bq_table_devsample).to_dataframe()
len(ClickRecs)


# ##### Sort the records by Visid and Hit Time so the GroupBy is correct

# In[12]:

ClickRecs.sort_values(['post_visid','hit_time_gmt'], ascending=True, inplace=True )
#ClickRecs.head(15)


# ##### Create two new flag variable to control groupby processing below
# ##### This will create two new variables in ClickRecs and populate both vectors with 0's

# In[13]:

ClickRecs['first_flg'] = 0
ClickRecs['last_flg'] = 0


Flag_FRecs = ClickRecs.groupby(['post_visid']).first()
Flag_FRecs['first_flg'] = 1
Flag_FRecs.reset_index(inplace=True)
##Flag_FRecs.ix[:2,['post_visid','date_time','first_flg','last_flg']]

Flag_LRecs = ClickRecs.groupby(['post_visid']).last()
Flag_LRecs['last_flg'] = 1
Flag_LRecs.reset_index(inplace=True)
##Flag_LRecs.ix[:2,['post_visid','date_time','first_flg','last_flg']]


# ##### Examine a few records from both of the new dataframes

# In[14]:

Flag_FRecs.ix[:3,['post_visid','date_time','first_flg','last_flg']]


# In[15]:

Flag_LRecs.ix[:3,['post_visid','date_time','first_flg','last_flg']]
type(Flag_LRecs)


# ##### Concatenate the two groupby df's

# In[16]:

dataframes = [Flag_FRecs, Flag_LRecs]  # This is a List to just create the dataframes object (unecessary but illustrative)
Flag_All = pd.concat(dataframes)  # This is a dataframe

Flag_All.reset_index(inplace=True)  # If False then it will create a copy. True prevent KeyError sorting on post_visid
# and will set the index to a sequential integer
Flag_All.sort_values(['post_visid','hit_time_gmt'], ascending=True, inplace=True)

##Flag_All.ix[:9,['post_visid','date_time','first_flg','last_flg']]
##ClickRecs.ix[:9,['post_visid','date_time','first_flg','last_flg']]
# This will sort correctly by the post_visid column and ignores the index

Flag_All
#Flag_All.to_csv("H:\\Notebook_Project\\Flag_All_out.csv", encoding='utf-8', columns=Flag_All.columns.values.tolist())


# ##### Remove the duplicates where there was only a single click associated with the visit(i.e., which created two records in Flag_All)

# In[17]:

Flag_All2 = Flag_All.drop_duplicates(subset=('post_visid','hit_time_gmt'),keep='last') 
Flag_All2


# ##### Identify the visits with a single click event

# In[18]:

FList=[]
t=0
f=0

for index1, row1 in Flag_FRecs.iterrows():
     for index2, row2 in Flag_LRecs.iterrows():
         if ((row1['post_visid'] == row2['post_visid']) and (row1['hit_time_gmt'] == row2['hit_time_gmt'])):
             row1['last_flg'] = 1
             FList.append(row1)     #this works since you can append a Series to a List
             #FlagRecs1.append(row1) # this doesn't fail but doesn't write to the df
             t=t+1
             #print('in true cond', t)
         #else:
             #Flist.append(row1)
             #f=f+1 
             #print('in else cond', f)
type(FList)
#print(FList)   #Note that Name is the Index value in the record list
#len(FList)
#type(FlagRecs1)
#print(FlagRecs1)
#len(FlagRecs1)


# ##### Create a new Dataframe from the List object created above

# In[19]:

Fdf = pd.DataFrame(FList, columns=['post_visid','date_time','visit_num','visit_page_num', 'visit_start_time_gmt', 
                                  'hit_time_gmt', 'channel','last_hit_time_gmt','page_event', 'page_event_var1',
                                  'page_event_var2','page_type','pagename',
                                  'prop1', 'prop30','evar4', 'evar10','evar26','evar30','event_list','product_list', 
                                  'cust_visid visit_dt','first_flg','last_flg'] )
type(Fdf)
#Fdf


# ##### Execute a simple SQL statement on the dataframe to confirm the functionality of PandaSQL . Note: the Locals() scope variable enable visiability to the dataframe.  This is not necessary in Python 3.5 but only 2.7 apparently.

# In[20]:

pdsql = PandaSQL()
#type(pdsql)
pdsql("SELECT r.post_visid FROM Fdf r limit 5;",locals())
#local = locals()


# #### Merge the two "Flag" dataframes 

# In[21]:

Final_Recs = pdsql("""Select f.post_visid, f.hit_time_gmt, f.first_flg, f.last_flg, r.first_flg as first_flg_1rec, 
       r.last_flg as last_flg_1rec    
       From Flag_All2 f  
       Left Outer Join Fdf r 
       On  r.post_visid = f.post_visid""", locals())
       ##And f.last_flg = r.last_flg""")
Final_Recs


# ##### Merge all of the dataframes 

# In[22]:

CR_merge = pdsql("""Select cr.post_visid, cr.hit_time_gmt, fr.first_flg as first_flg_mrecs, fr.last_flg as last_flg_mrecs,
       fr.first_flg_1rec, fr.last_flg_1rec,
       date_time,visit_num,visit_page_num,visit_start_time_gmt,channel,last_hit_time_gmt,page_event,page_event_var1,
       page_event_var2,page_type,pagename,prop1,prop30,evar4,evar10,evar26,evar30,event_list,product_list,cust_visid,visit_dt
       From ClickRecs cr  
       Left Outer Join Final_Recs fr 
       On  cr.post_visid = fr.post_visid
       and cr.hit_time_gmt = fr.hit_time_gmt""", locals())
CR_merge  

# Write the file out to the local .csv file
#CR_merge2.to_csv("H:\\Notebook_Project\\ClickSample_out2.csv", encoding='utf-8', columns=CR_merge.columns.values.tolist())


# ##### Iterate through all of the records to create the aggregates for the ADS (for continued dev consider a set of groupby functions)

# In[23]:

temp_list = []

# Loop through the records to create the ADS
n = 0
for index, row in CR_merge.iterrows():  # if you don't include index then "TypeError: tuple indices must be integers or slices, not str"
    #print('top o the for statement')
    #print('evar4 =', (row['evar4 ']))
    #print ('first_flg=', CR_merge['first_flg_y'][n])
    
    if row['first_flg_mrecs'] == 1 or row['first_flg_1rec'] == 1:  # This is for variable initialization for the first visit in series
        order = 0
        browse_plp_cnt = 0
        browse_scat_cnt = 0
        int_srch_pip_cnt = 0
        pd_cmpgn_content_cnt = 0
        pip_cnt = 0
        plp_cnt = 0
        content = 0
        spc_buy = 0
        refind_srch_cnt = 0
        thumbnail_vw_cnt = 0
        acct_sgn_in = 0
        search_cnt = 0
        appliances = 0
        page_view_cnt = 0
        visit_duration = 0

    if not (pd.isnull(row['page_event_var2'])):
        if 'submit order' in row['page_event_var2']:
            order = 1
        if not (pd.isnull(row['evar4'])) and not (pd.isnull(row['prop30'])):
            if 'browse' in row['evar4'] and 'plp' in row['prop30']:
                browse_plp_cnt += 1
            if 'browse' in row['evar4'] and 'subcategory' in row['prop30']:
                browse_scat_cnt += 1
            if 'internal search' in row['evar4'] and 'pip' in row['prop30']:
                int_srch_pip_cnt += 1
            if 'internal search' in row['evar4'] and 'subcategory' in row['prop30']:
                browse_scat_cnt += 1
            if 'paid campaign' in row['evar4'] and 'content' in row['prop30']:
                pd_cmpgn_content_cnt += 1
        if not (pd.isnull(row['prop30'])):
            if 'pip' in row['prop30']:
                pip_cnt += 1
            if 'plp' in row['prop30']:
                plp_cnt += 1
            if 'content' in row['prop30']:
                content += 1
            if 'special buy' in row['prop30']:
                spc_buy += 1
        if not (pd.isnull(row['page_event_var2'])):
            if 'product thumbnail click' in row['page_event_var2']:
                thumbnail_vw_cnt += 1
            if 'Refine Search' in row['page_event_var2']:
                refind_srch_cnt += 1
            if 'account sign in' in row['page_event_var2']:
                acct_sgn_in += 1
        if not (pd.isnull(row['channel'])):
            if 'search' in row['channel']:
                search_cnt += 1
            if 'appliances' in row['channel']:
                appliances += 1
            # this is to check for the last visitor ID in the series
    if row['last_flg_mrecs'] == 1 or row['last_flg_1rec'] == 1:
        ## use the sequential numbering of the page views to set the total number of page views
        if not (pd.isnull(row['visit_page_num'])) and not (pd.isnull(row['hit_time_gmt']))                 and not (pd.isnull(row['visit_start_time_gmt'])):
            page_view_cnt = row['visit_page_num']
            visit_duration = int(row['hit_time_gmt']) - int(row['visit_start_time_gmt']) / 60

        temp_dict = {'post_visid': row['post_visid'],                      'order': order,                      'browse_plp_cnt': browse_plp_cnt,                      'browse_scat_cnt': browse_scat_cnt,                      'int_srch_pip_cnt': int_srch_pip_cnt,                      'pd_cmpgn_content_cnt': pd_cmpgn_content_cnt,                      'pip_cnt': pip_cnt,                      'plp_cnt': plp_cnt,                      'content': content,                      'spc_buy': spc_buy,                      'thumbnail_vw_cnt': thumbnail_vw_cnt,                      'refind_srch_cnt': refind_srch_cnt,                      'page_view_cnt': page_view_cnt,                      'visit_duration': visit_duration,             }

        temp_list.append(temp_dict)

    n = n + 1
out = pd.DataFrame(temp_list)


# In[24]:

out 


# ### <u>Output Options (In Development)</u>
# #####  <hr>  </hr> 

# In[25]:

import gcp
import gcp.storage as storage
import gcp.bigquery as bq
import pandas as pd


# ##### Create a BigQuery table for the ADS (this works) 

# In[ ]:

ads = bq.DataSet('ClickADS2')  # First, create the dataset.... this is not the table !
ads.create(friendly_name = 'ClickStream ADS', description = 'ADS created from Sample Omniture data')
ads.exists()

bigquery_dataset_name = 'ClickADS2'
bigquery_table_name = 'ADS_Logit1'

# Define BigQuery dataset and table
dataset = bq.DataSet(bigquery_dataset_name)
table = bq.Table(bigquery_dataset_name + '.' + bigquery_table_name)

# Create or overwrite the existing table if it exists
table_schema = bq.Schema.from_dataframe(out)
table.create(schema = table_schema, overwrite = True)

# Write the DataFrame to a BigQuery table
table.insert_data(out)


# In[ ]:

print(table_schema)


# -------------------------------
# ### In Development
# -------------------------------

# ### Different write functions into the VM Files System or GCS for audit and/or persistent storage 

# ##### Create a bucket in GCS and either write from the Python DataFrame or write to this bucket from the VM (see below)
# #####  (This works)

# In[26]:

import gcp
import gcp.storage as storage
from StringIO import StringIO


# In[34]:

project = gcp.Context.default().project_id   # correct Project ID is found
bucket_name = 'steve-temp2'           ## .... or can create a new bucket using the bucket.create() below
bucket_path  = 'gs://' + bucket_name   
bucket_object = bucket_path + '/out.csv'
#bucket_object = bucket_path + '/out2.csv'
bucket = storage.Bucket(bucket_name)

# Create the bucket if it doesn't exist
if not bucket.exists():
  bucket.create()

bucket  ## This command will display the bucket name  
# The name of the bucket is:  gs://steve-temp2

bucket.exists()
# Confirmed created in GCS both with a visual inspection and a "True" result from the this function
#bucket_path
#bucket_object


# ##### Use the line command storage magic to write to GCS

# In[48]:

get_ipython().magic('storage write --variable out --object $bucket_object')


# ##### Write out the ADS dataframe above to the VM file system ( This worked )

# In[23]:

out.to_csv("ads_out2.csv", encoding='utf-8', columns=out.columns.values.tolist()) 


# In[45]:

# Write the file to the storage bucket
#file = bucket.item('ads_out2.csv')
file
##file.write_to(bucket)


# ##### This shells out to the VM and executes the gsutil ( This works )

# In[ ]:

get_ipython().run_cell_magic('bash', '', '##gsutil cp -r /content/steven_einbender@homedepot.com gs://steve-temp2\ngsutil cp /content/steven_einbender@homedepot.com/ads_out2.csv  gs://steve-temp')


# ##### Read from GC Storage and create a Python DataFrame (This works and retains the schema from the .to_csv above)

# In[55]:

gcs_ads_in = storage.Item('steve-temp','ads_out2.csv').read_from()
#The following will just display the file contents as a continuous string. Str is the object type
#gcs_ads_in
ads_df = pd.read_csv(StringIO(gcs_ads_in))
#type(ads_df)  #This is now a DataFrame
#ads_df


# ##### Read from GC Storage and create a Python DataFrame ( This works too but was schema-less from the %storage write above )

# In[29]:

gcs_ads2_in = storage.Item('steve-temp2','out.csv').read_from()
ads2_df = pd.read_csv(StringIO(gcs_ads2_in))
#type(ads2_df)  #This is now a DataFrame
#ads2_df


# In[37]:

#ads2_df.ix[:3,['browse_plp_cnt']]
#pdsql("SELECT * FROM ads2_df limit 5;",locals())
#%storage view --object $bucket_object


#  

# ##### The following two both work and produce the same result

# In[52]:

##list(bucket.items())  ## This can be a long list


# In[55]:

##%%storage list --bucket $bucket_path


# ##### This lists all files in the parent specified

# In[43]:

##%%storage list  --bucket gs://steve-temp   


# ##### Other development

# In[ ]:

#bucket_object = bucket_path + '/ClickSample_out1.csv'
#bucket_object2 = bucket_path + '/ClickSample_out1.csv'
#bucket = storage.Bucket(bucket_path)
#bucket.create()

#bucket.exists()
#project
bucket_path


# In[ ]:

##%%bash
##gsutil cp 'ads_out1.csv' 


#print(project)
#print(bucket_name)
#print('bucket path is:', bucket_path)
#print('bucket object/table is:', bucket_object)


# In[ ]:

#bucket_item = bucket.item('ClickSample_out1.csv')
#%storage write -h
#%storage write --variable CR_merge --object $bucket_object
#type(bucket_item)
#print(bucket_item)
#bucket_item.exists()
#list(bucket_item.items())


# ##### Execute CLI commands in the VM

# In[23]:

get_ipython().run_cell_magic('bash', '', 'ls -al\npwd\nhead ClickSample_out1.csv')

