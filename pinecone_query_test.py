import pinecone
import openai
import configparser
import os
import glob
import json
config = configparser.ConfigParser()
config.read('config.ini')

def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()

openai.api_key = open_file(config['open_ai']['api_key'])
pinecone.init(api_key=open_file(config['pinecone']['api_key']), environment=config['pinecone']['environment'])
vector_db = pinecone.Index(config['pinecone']['index'])

def gpt3_embedding(content):
    engine = config['open_ai']['input_engine']
    content = content.encode(encoding='ASCII',errors='ignore').decode()  # fix any UNICODE errors
    response = openai.Embedding.create(input=content,engine=engine)
    vector = response['data'][0]['embedding']  # this is a normal list
    return vector

def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return json.load(infile)

def query_pinecone(vector, return_n, namespace = "", search_all = False):
    if search_all:
        results = vector_db.query(vector=vector, top_k=return_n)
    else:
        results = vector_db.query(vector=vector, top_k=return_n, namespace=namespace)
    return results

if __name__ == '__main__':
    depth = 1
    namespace = config['memory_management']['memory_namespace_template'] % depth
    search_string = 'Yesterday we were talking about an air fryer recipe for dounuts, can you repeat the instructions?'
    vector = gpt3_embedding(search_string)
    results = query_pinecone(vector, 3, namespace)
    if 'matches' in results:
        matches = results['matches']
        folder_path = 'memory_management\memory_stash\depth_%s_stash' % str(depth)
        for match in matches:
            file_path = '%s\%s.json' % (folder_path, match['id']) 
            memory = load_json(file_path)
            if depth == 0:
                memory_content = memory['content']
            else:
                memory_content = memory['summary']
            print('Memory with id {%s} and a match score of %s:\n%s' % (match['id'], str(match['score']), memory_content))

    print('done')


    # files = glob.glob('memory_management\memory_stash\depth_%s_stash\*.json' % str(depth))
    # metadata = {'memory_type': 'episodic', 'depth': str(depth)}
    # payload = list()
    # for f in files:
    #     memory = load_json(f)
    #     if depth == 0:
    #         vector = gpt3_embedding(memory['content'])
    #     else:
    #         vector = gpt3_embedding(memory['summary'])
    #     payload.append({'id': memory['id'], 'values': vector, 'metadata': metadata})

