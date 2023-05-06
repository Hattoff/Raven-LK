import configparser
import os
import json
import glob
from time import time,sleep
import datetime
from uuid import uuid4
import pinecone
import tiktoken
import re
import openai
_raven_update_debug = None

#####################################################
                ## File Management ##

def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()

def save_file(filepath, content):
    with open(filepath, 'w', encoding='utf-8') as outfile:
        outfile.write(content)

def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return json.load(infile)

def save_json(filepath, payload):
    with open(filepath, 'w', encoding='utf-8') as outfile:
        json.dump(payload, outfile, ensure_ascii=False, sort_keys=True, indent=2)

def string_to_json(content):
    return json.loads(content)

#####################################################

config = configparser.ConfigParser()
config.read('config.ini')
enable_all_debug_message = False

openai.api_key = open_file(config['open_ai']['api_key'])
pinecone_indexing_enabled = config.getboolean('pinecone', 'pinecone_indexing_enabled')
pinecone.init(api_key=open_file(config['pinecone']['api_key']), environment=config['pinecone']['environment'])
vector_db = pinecone.Index(config['pinecone']['index'])

def get_config():
    return config

def reload_config():
    global config
    config = configparser.ConfigParser()
    config.read('config.ini')

def timestamp_to_datetime(unix_time):
    return (datetime.datetime.fromtimestamp(unix_time).strftime("%A, %B %d, %Y at %I:%M:%S%p %Z")).strip()

#####################################################
                ## TikToken ##
def get_token_estimate(content):
    content = content.encode(encoding='ASCII',errors='ignore').decode()
    # encoding = tiktoken.encoding_for_model(str(config['open_ai']['model']))
    ## TODO: TikToken doesn't know about the new model names, will need to hard code for now.
    encoding = tiktoken.encoding_for_model('gpt-3.5-turbo-0301')
    tokens = encoding.encode(content)
    token_count = len(tokens)
    return token_count

#####################################################
                ## OpenAI ##
def gpt3_embedding(content):
    engine = config['open_ai']['input_engine']
    content = content.encode(encoding='ASCII',errors='ignore').decode()
    response = openai.Embedding.create(input=content,engine=engine)
    vector = response['data'][0]['embedding']
    return vector

def gpt_completion(messages, temp=0.0, tokens=400, stop=['USER:', 'RAVEN:'], print_response = False):
    engine = config['open_ai']['model']
    top_p=1.0
    freq_pen=0.0
    pres_pen=0.0

    max_retry = 5
    retry = 0
    while True:
        try:
            response = openai.ChatCompletion.create(
                model=engine,
                messages=messages,
                temperature=temp,
                max_tokens=tokens,
                top_p=top_p,
                frequency_penalty=freq_pen,
                presence_penalty=pres_pen,
                stop=stop)
            total_tokens = int(response['usage']['total_tokens'])
            response_str = response['choices'][0]['message']['content'].strip()
            if print_response:
                print_response_stats(response)
            return response_str, total_tokens
        except Exception as oops:
            retry += 1
            if retry >= max_retry:
                return "GPT3.5 error: %s" % oops, -1
            print('Error communicating with OpenAI:', oops)
            sleep(2)

def print_response_stats(response):
    response_id = ('\nResponse %s' % str(response['id']))
    prompt_tokens = ('\nPrompt Tokens: %s' % (str(response['usage']['prompt_tokens'])))
    completion_tokens = ('\nCompletion Tokens: %s' % str(response['usage']['completion_tokens']))
    total_tokens = ('\nTotal Tokens: %s\n' % (str(response['usage']['total_tokens'])))
    print(response_id + prompt_tokens + completion_tokens + total_tokens)

#####################################################
                ## Debug ##
def breakpoint( message = '\n\nEnter to continue...'):
    input(message+'\n')

def debug_message(message, display_debug = False):
    if display_debug or enable_all_debug_message:
        print(message)

#####################################################
                ## Pinecone ##

def query_pinecone(vector, return_n, namespace = "", search_all = False):
    if search_all:
        results = vector_db.query(vector=vector, top_k=return_n)
    else:
        results = vector_db.query(vector=vector, top_k=return_n, namespace=namespace)
    return results

def enable_pinecone_indexing():
    global pinecone_indexing_enabled
    pinecone_indexing_enabled = True

def disable_pinecone_indexing():
    global pinecone_indexing_enabled 
    pinecone_indexing_enabled = False

def save_payload_to_pinecone(payload, namespace):
    if not pinecone_indexing_enabled:
        return
    vector_db.upsert(payload, namespace=namespace)

def save_vector_to_pinecone(vector, unique_id, metadata, namespace=""):
    if not pinecone_indexing_enabled:
        return
    debug_message('Saving vector to pinecone.')
    payload_content = {'id': unique_id, 'values': vector, 'metadata': metadata}
    payload = list()
    payload.append(payload_content)
    vector_db.upsert(payload, namespace=namespace)

def update_pinecone_vector(id, vector, namespace):
    if not pinecone_indexing_enabled:
        return
    update_response = {}
    update_response = vector_db.update(id=id,values=vector,namespace=namespace)
    return update_response
