import pinecone
import openai
import configparser
import os
import glob
import json
from UtilityFunctions import *
config = get_config()
# config = configparser.ConfigParser()
# config.read('config.ini')

# def open_file(filepath):
#     with open(filepath, 'r', encoding='utf-8') as infile:
#         return infile.read()

# openai.api_key = open_file(config['open_ai']['api_key'])
# pinecone.init(api_key=open_file(config['pinecone']['api_key']), environment=config['pinecone']['environment'])
# vector_db = pinecone.Index(config['pinecone']['index'])

# def gpt3_embedding(content):
#     engine = config['open_ai']['input_engine']
#     content = content.encode(encoding='ASCII',errors='ignore').decode()  # fix any UNICODE errors
#     response = openai.Embedding.create(input=content,engine=engine)
#     vector = response['data'][0]['embedding']  # this is a normal list
#     return vector

# def load_json(filepath):
#     with open(filepath, 'r', encoding='utf-8') as infile:
#         return json.load(infile)


# def query_pinecone(vector, return_n, namespace = "", search_all = False):
#     if search_all:
#         results = vector_db.query(vector=vector, top_k=return_n)
#     else:
#         results = vector_db.query(vector=vector, top_k=return_n, namespace=namespace)
#     return results

if __name__ == '__main__':
    # depth = 0
    # namespace = config['memory_management']['memory_namespace_template'] % depth
    # files = glob.glob('memory_management\memory_stash\depth_%s_stash\*.json' % str(depth))
    # metadata = {'memory_type': 'episodic', 'depth': str(depth), 'speaker':''}
    # payload = list()
    # for f in files:
    #     memory = load_json(f)
    #     if depth == 0:
    #         vector = gpt3_embedding(memory['content'])
    #     else:
    #         vector = gpt3_embedding(memory['summary'])
    #     metadata['speaker'] = memory['speaker']
    #     payload.append({'id': memory['id'], 'values': vector, 'metadata': metadata})
    # print('uploading %s vectors to pinecone under namespace %s' % (str(len(payload)), namespace))
    # vector_db.upsert(payload, namespace=namespace)
    # print('done')
    theme_namespace = config['memory_management']['theme_namespace_template']
    theme_folderpath = config['memory_management']['theme_stash_dir']
    existing_theme_id = '9622547f-30b8-4007-9299-3f0a5272ea3f'
    theme_filepath = '%s/%s.json' % (theme_folderpath, existing_theme_id)
    existing_theme = load_json(theme_filepath)
    new_theme_string = ','.join(existing_theme['themes'])
    new_theme_vector = gpt3_embedding(new_theme_string)
    print(new_theme_string)
    ## Update existing pinecone record's vector
    update_pinecone_vector(existing_theme_id, new_theme_vector, theme_namespace)
