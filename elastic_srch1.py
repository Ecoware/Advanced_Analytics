##Pgm: elastic_srch1.py
from elasticsearch import Elasticsearch
es = Elasticsearch(hosts=[{"host":'10.43.40.28', "port":9200}])
query_result = {"query": {"match_all":{}}}
result = es.search(index='ipfix-2017-04-28',doc_type="agg5mins",body=query_result)

#print result['hits']['hits']
for doc in result['hits']['hits']:
    print("%s) %s" % (doc['_id'], doc['_source']))