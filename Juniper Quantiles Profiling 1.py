
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

import xlsxwriter

#%%
 
run_ReadFromBQTable = 0
run_ReadAndJoinFromBQTables = 1
run_ReadFromBQTableAndAgg = 0
run_crosstab1 = 1
run_writeoutput1 = 1
bgn_time = ' "2018-09-10 00:00:00" '
end_time = ' "2018-09-23 23:59:59" '       
#end_time = ' "2018-09-16 23:59:59" '

 ## The explicit listing of IP's is optional.  The "Join" query below can also utilize a list of IP's stored in a 
 ## Big Query table
#DstIPList = ' "165.130.144.33","165.130.144.31","165.130.144.30","165.130.144.32","165.130.144.84","165.130.144.83" '
DstIPList = ' "165.130.144.83" ' 

#%%

if run_ReadFromBQTable == 1:
   pci_df =  ReadFromBQTable()
    
if run_ReadAndJoinFromBQTables == 1:
    pci_df = ReadAndJoinFromBQTables()
    WriteDataFrameToBQ2(pci_df)
    
if run_ReadFromBQTableAndAgg == 1:
   juniper_df = ReadFromBQTableAndAgg(DstIPList)
   srcpt_cat()
   ct_jun_df = crosstab1()
   writeoutput1("ct_2wk.xlsx",ct_jun_df)
   WriteDataFrameToBQ()

#%%

## How do we use protocol_id ????
def ReadFromBQTableAndAgg(DstIPList):
    client = bigquery.Client()
    query = (
            'SELECT FORMAT_TIMESTAMP("%Y-%m-%d %H",timestamp) as hour, destination_address, destination_port, \
             source_address, count(*) as rel_counts, sum(bytes_from_server) as bytes_from_server \
             FROM `io1-datalake-views.security_pr.juniper_junos_firewall` '
            'where destination_address in('+ DstIPList + ') '
            'and TIMESTAMP >= '+ bgn_time +' and TIMESTAMP <= ' + end_time +    
            'and protocol_id = 6 '
            'and session_id is not null '
            'group by hour, destination_address, destination_port, source_address '
            'order by hour, destination_address, source_address, rel_counts desc' )


    juniper_df = client.query(query).to_dataframe()
    
    return(juniper_df)    

#%%
## 
##
def ReadAndJoinFromBQTables():
    
    client = bigquery.Client()
    query = ('Select FORMAT_TIMESTAMP("%Y-%m-%d %H",timestamp) as hour, j.destination_address, j.destination_port, \
             j.source_address, count(*) as rel_counts, sum(bytes_from_server) as bytes_from_server \
             from `io1-datalake-views.security_pr.juniper_junos_firewall` j '
             'where TIMESTAMP >= '+ bgn_time +' and TIMESTAMP <= ' + end_time +  
             'and j.destination_address in('+ DstIPList + ') '  
             'and j.source_address not between "10.77.101.128" and "10.77.101.191" '
             'and j.source_address not between "10.69.101.128" and "10.69.101.191"'
             'and protocol_id = 6 '
             'and session_id is not null '
             'group by hour, destination_address, destination_port, source_address '
             'order by hour, destination_address, source_address, rel_counts desc')
            

    pci_df = client.query(query).to_dataframe()
    
    return(pci_df)

#%%         
## 
##
def ReadAndJoinFromBQTables2():
    
    client = bigquery.Client()
    query = ('Select timestamp, j.destination_address, j.destination_port, j.source_address, j.protocol_id \
             from `io1-datalake-views.security_pr.juniper_junos_firewall` j,  \
                  `network-sec-analytics.reference.Fortress_aus_ern_sfpci_pri_100` i '
             'where timestamp >= "2018-09-10 00:00:00" and timestamp <= "2018-09-10 01:59:59" '
             'and j.destination_address = i.destination_address '
        
             )
            

    pci_df = client.query(query).to_dataframe()
    
    return(pci_df)
#%%
## 
## Raw data extract from Juniper io1 Big Query dataset
## Execution time was ~ 20 seconds
##
def ReadFromBQTable():
    
    client = bigquery.Client()
    query = ('Select timestamp, destination_address, destination_port, source_address, protocol_id \
             from `io1-datalake-views.security_pr.juniper_junos_firewall` '
             'where timestamp >= "2018-09-10 00:00:00" and timestamp <= "2018-09-23 23:59:59" '
             'and destination_address = "165.130.217.224" '
        
             )
            

    pci_df = client.query(query).to_dataframe()
    
    return(pci_df)
   


   
#%%
def srcpt_cat():
    juniper_df.loc[(juniper_df.destination_port >= 0) & (juniper_df.destination_port <= 1023) , 'ports'] = juniper_df.destination_port
    juniper_df.loc[(juniper_df.destination_port >= 1024) & (juniper_df.destination_port <= 49151) , 'ports'] = '1024-49151'
    juniper_df.loc[(juniper_df.destination_port >= 49152) & (juniper_df.destination_port <= 65535) , 'ports'] = '49152-65535'
    juniper_df.loc[(juniper_df.destination_port == 3389) , 'ports'] = juniper_df.destination_port
srcpt_cat()

    
    
#%%    
def crosstab1():
    ct_jun_df = pd.crosstab([juniper_df.destination_address,juniper_df.ports]
    ,juniper_df.rel_counts,margins=True)
    ct_jun_df.reset_index(inplace=True)
    ct_jun_df.sort_values(by=['destination_address','All'],ascending=[True,False],inplace=True)
    print(ct_jun_df.head(10))
    return(ct_jun_df)

#%%
def writeoutput1(filename,df):
    import xlsxwriter
    writer = pd.ExcelWriter(filename,engine='xlsxwriter')
    df.to_excel(writer,'Sheet1')
    writer.save()
    return()

#%%
def WriteDataFrameToBQ():
   juniper_df.to_gbq('ConnectionModeling.juniper_2wk', "network-sec-analytics", verbose=True, reauth=False, 
         if_exists='replace', private_key=None)
   return()
 
#%%
def WriteDataFrameToBQ2(df):
   df.to_gbq('ConnectionModeling.pci_df_2wk', "network-sec-analytics", verbose=True, reauth=False, 
         if_exists='replace', private_key=None)
   return()

#%%
def plot1():
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
    return()



##----------------------------------------------------------------------------
## Main processing section
##----------------------------------------------------------------------------



    
      
    


