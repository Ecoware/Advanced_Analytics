import time
import os
from googleapiclient import discovery
from oauth2client import client as oauth2client
import pandas as pd

"""
Function to run certain queries
"""


def RunQuery(bigquery):
    ProjectID = 'network-sec-analytics'


    QueryRequest = bigquery.jobs()
    QueryData = {
        'query': (

            'SELECT dst_addr '
            'FROM [network-sec-analytics:ipfix.ipfix]' 
            'LIMIT 5;'
        )

    }

    QueryResponse = QueryRequest.query(
        projectId=ProjectID,
        body=QueryData).execute()

    print ('Query Results:')

    # for row in QueryResponse['rows']:
    #     print ('\t'.join(field['v'] for field in row['f']))

    column_list = QueryResponse['schema']['fields']

    # response = request.execute()
    # query = response['rows'][0]['f'][0]['v']
    row_list = []
    for row in QueryResponse['rows']:
        row_dict = {}
        for column in range(len(row['f'])):
            value = row['f'][column]['v']
            key = column_list[column]['name']
            row_dict[key] = value

        row_list.append(row_dict)

    df = pd.DataFrame(row_list)

    print df

"""
Function to create bigquery client
"""


def CreateBigQueryClient():
    credentials = oauth2client.GoogleCredentials.get_application_default()


    return discovery.build('bigquery', 'v2', credentials=credentials)

if __name__ == '__main__':

    bigquery = CreateBigQueryClient()
    RunQuery(bigquery)