import random
import pinecone
import os
import configparser

['config.ini']
def open_file(filepath):       
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()

config = configparser.ConfigParser()
config.read('config.ini')

pinecone_api_key = open_file(config['pinecone']['api_key'])
pinecone_environment_name = environment=config['pinecone']['environment']
pinecone.init(api_key=pinecone_api_key, environment=pinecone_environment_name)
index = pinecone.Index(config['pinecone']['index'])


# Fetch all IDs in the given namespace
def fetch_ids(depth = 1, top_k = 24):
    # pinecone_namespace = 'episodic_depth_%s' % str(depth)
    pinecone_namespace = 'themes'
    vector = [random.random() for _ in range (1536)]
    response = index.query(vector=vector, top_k=top_k, include_values=False, namespace=pinecone_namespace)
    ids = ','.join(t['id'] for t in response['matches'])
    print(ids)
    return ids

if __name__ == '__main__':
    fetch_ids()