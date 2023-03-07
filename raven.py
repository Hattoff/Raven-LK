import configparser
import os
import openai
import json
import numpy as np
from numpy.linalg import norm
import re
from time import time,sleep
from uuid import uuid4
import datetime
import pinecone
import math

config = configparser.ConfigParser()
config.read('config.ini')

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


def timestamp_to_datetime(unix_time):
    return datetime.datetime.fromtimestamp(unix_time).strftime("%A, %B %d, %Y at %I:%M%p %Z")


def gpt3_embedding(content):
    engine = config['open_ai']['input_engine']
    content = content.encode(encoding='ASCII',errors='ignore').decode()  # fix any UNICODE errors
    response = openai.Embedding.create(input=content,engine=engine)
    vector = response['data'][0]['embedding']  # this is a normal list
    return vector



def gpt_completion(prompt, engine=config['open_ai']['model'], temp=0.0, top_p=1.0, tokens=400, freq_pen=0.0, pres_pen=0.0, stop=['USER:', 'RAVEN:']):
    if 'gpt-3.5-turbo' in engine:
        gpt3_5_completion(prompt, engine, temp, top_p, tokens, freq_pen, pres_pen, stop)
    else:
        gpt3_completion(prompt, engine, temp, top_p, tokens, freq_pen, pres_pen, stop)

def gpt3_5_completion(prompt, engine, temp, top_p, tokens, freq_pen, pres_pen, stop):
    breakpoint('Using gpt3.5')
    max_retry = 5
    retry = 0
    prompt = prompt.encode(encoding='ASCII',errors='ignore').decode()
    while True:
        try:
            response = openai.ChatCompletion.create(
                model=engine,
                messages=[
                    {'role': 'user', 'content': prompt}
                ],
                temperature=temp,
                max_tokens=tokens,
                top_p=top_p,
                frequency_penalty=freq_pen,
                presence_penalty=pres_pen,
                stop=stop)
            print(response)
            print(response['choices'])
            text = response['choices'][0]['message']['content'].strip()
            text = re.sub('[\r\n]+', '\n', text)
            text = re.sub('[\t ]+', ' ', text)
            filename = '%s_gpt3.txt' % time()
            foldername = config['raven']['gpt_log_dir']
            if not os.path.exists(foldername):
                os.makedirs(foldername)
            save_file('%s/%s' % (foldername,filename), prompt + '\n\n==========\n\n' + text)
            return text
        except Exception as oops:
            retry += 1
            if retry >= max_retry:
                return "GPT3.5 error: %s" % oops
            print('Error communicating with OpenAI:', oops)
            sleep(2)

def gpt3_completion(prompt, engine, temp, top_p, tokens, freq_pen, pres_pen, stop):
    breakpoint('Using gpt3')
    max_retry = 5
    retry = 0
    prompt = prompt.encode(encoding='ASCII',errors='ignore').decode()
    while True:
        try:
            response = openai.Completion.create(
                engine=engine,
                prompt=prompt,
                temperature=temp,
                max_tokens=tokens,
                top_p=top_p,
                frequency_penalty=freq_pen,
                presence_penalty=pres_pen,
                stop=stop)
            text = response['choices'][0]['text'].strip()
            text = re.sub('[\r\n]+', '\n', text)
            text = re.sub('[\t ]+', ' ', text)
            filename = '%s_gpt3.txt' % time()
            foldername = config['raven']['gpt_log_dir']
            if not os.path.exists(foldername):
                os.makedirs(foldername)
            save_file('%s/%s' % (foldername,filename), prompt + '\n\n==========\n\n' + text)
            return text
        except Exception as oops:
            retry += 1
            if retry >= max_retry:
                return "GPT3 error: %s" % oops
            print('Error communicating with OpenAI:', oops)
            sleep(2)

def load_conversation(results):
    recalled = list()
    for m in results['matches']:
        info = load_json('nexus/%s.json' % m['id'])
        recalled.append(info)
    ordered = sorted(recalled, key=lambda d: d['time'], reverse=False)  # sort them all chronologically
    messages = [i['message'] for i in ordered]
    return '\n'.join(messages).strip(), recalled

def summarize_memories(memories):  # summarize a block of memories into one payload
    memories = sorted(memories, key=lambda d: d['time'], reverse=False)  # sort them chronologically

    blocked_summary = ''
    chunked_summary = ''
    blocks = chunk_memories(memories)
    block_count = len(blocks)
    for chunks in blocks:
        for chunk in chunks:
            chunked_message = ''
            for mem in chunk:
                message = format_summary_memory(mem)
                chunked_message += message + '\n\n'
            chunked_message = chunked_message.strip()
            chunked_prompt = open_file('prompt_notes.txt').replace('<<INPUT>>', chunked_summary + chunked_message)
            chunk_prompt_filename = 'summary_chunk_prompt_%s.json' % time()
            save_json('summary_prompts/%s' % chunk_prompt_filename, chunked_prompt)

            chunked_notes = gpt_completion(chunked_prompt)

            chunk_notes_filename = 'summary_chunk_notes_%s.json' % time()
            save_json('summary_prompts/%s' % chunk_notes_filename, chunked_notes)
            breakpoint('Finished with first prompt')
            # chunked_summary += prompt + '\n\n'
        # chunks_filename = 'summary_summarized_chunks_prompt_%s.json' % time()
        # chunks_prompt = open_file('prompt_notes.txt').replace('<<INPUT>>', chunked_summary)
        # save_json('summary_prompts/%s' % chunks_filename, chunks_prompt)
        # breakpoint()
        # if block_count > 1:
            # blocked_summary += prompt + '\n\n'

    # block = ''
    # identifiers = list()
    # timestamps = list()
    # for mem in memories:
    #     message = format_summary_memory(mem)
    #     block += message + '\n\n'
    #     identifiers.append(mem['uuid'])
    #     timestamps.append(mem['time'])
    # block = block.strip()
    # prompt = open_file('prompt_notes.txt').replace('<<INPUT>>', block)
    # filename = 'summary_prompt_%s.json' % time()
    # save_json('summary_prompts/%s' % filename, prompt)
    breakpoint('finished...')
    # TODO - do this in the background over time to handle huge amounts of memories
    # notes = gpt_completion(prompt)
    ####   SAVE NOTES
    # vector = gpt3_embedding(block)
    # info = {'notes': notes, 'uuids': identifiers, 'times': timestamps, 'uuid': str(uuid4()), 'vector': vector, 'time': time()}
    # filename = 'notes_%s.json' % time()
    # save_json('internal_notes/%s' % filename, info)
    notes = ''
    return notes

def format_summary_memory(memory):
    message = '%s: %s - %s' % (memory['speaker'], memory['timestring'], memory['message'])
    return message

#### Breakup memories into chunks so they meet token restrictions
def chunk_memories(memories, token_response_limit=int(config['raven']['summary_token_response'])):
    #### Each subsequent chunk will subtract the max_token_input from the token_response_limit
    max_token_input = int(config['open_ai']['max_token_input'])
    token_per_character = int(config['open_ai']['token_per_character'])

    blocks = list() # When the token limit overflows after chunking, a new block will be needed
    chunks = list() # All chunks which will fall within the token limit including responses go here
    chunk = list() # All memories which will fall within the token limit will go here

    memory_count = len(memories)-1
    block_count = 0
    block_token_response_limit = token_response_limit
    current_memory_index = 0

    max_iter = 100 # If this is hit there is something wrong
    iter = 0

    blocking_done = False
    while not blocking_done:
        chunking_done = False
        ## TODO: Come back to this decision of doubling the token response limit for each block
        ## Initial reasoning behind this, a summary of summaries for the chunks would be created
        ## then a summary of the next block, so we would need double the response space for each block.
        block_token_response_limit = (2 * block_count * token_response_limit)
        remaining_chunk_tokens = max_token_input - block_token_response_limit
        while not chunking_done:
            chunk_length = 0
            chunk.clear()
            memories_this_chunk = 0
            for i in range(current_memory_index, memory_count):
                iter += 1
                mem = memories[i]
                message = format_summary_memory(mem)
                ## TODO: Get actual token length from open ai
                message_length = math.ceil(len(message)/token_per_character) # Estimate token length
                chunk_length += message_length
                if chunk_length > remaining_chunk_tokens and memories_this_chunk == 0:
                    current_memory_index = i
                    #### Chunking cannot continue, new block will be created
                    block_count += 1
                    blocks.append(chunks.copy())
                    chunking_done = True
                    breakpoint('\n\nChunking cannot continue until a new block is created...\n\n')
                    break
                elif chunk_length > remaining_chunk_tokens:
                    current_memory_index = i
                    #### Chunking can continue, new chunk will be created
                    remaining_chunk_tokens -= token_response_limit
                    chunks.append(chunk.copy())
                    chunking_done = True
                    break
                elif i == memory_count-1:
                    #### End of process, append remaining chunk and add chunks to block
                    chunk.append(mem)
                    chunks.append(chunk.copy())
                    blocks.append(chunks.copy())
                    chunking_done = True
                    blocking_done = True
                else:
                    #### Chunking continues, decrement remaining tokens
                    chunk.append(mem)
                    memories_this_chunk += 1
                    continue
                
                if iter >= max_iter:
                    chunking_done = True
                    blocking_done = True
                    breakpoint('\n\nSomething went wrong with the memory chunker. Max iterations reached.\n\n')
    return blocks


#### get user input, vectorize it, index it, save to pinecone
def prompt_user(payload = list()):
    user_input = input('\n\nUSER: ')
    vector = gpt3_embedding(user_input)
    message, unique_id = generate_message_metadata('USER',user_input)
    index_message(message,unique_id)
    payload.append((unique_id, vector))
    return payload, vector

#### Index message with unique id for memeory recall
def index_message(msg, unique_id=str(uuid4())):
    save_json('%s/%s.json' % (config['raven']['nexus_dir'],unique_id), msg)
    return unique_id

#### Generate message metadata for logging and memory indexing
def generate_message_metadata(speaker, msg, timestamp = time(), unique_id=str(uuid4())):
    timestring = timestamp_to_datetime(timestamp)
    message = {'speaker': speaker, 'time': timestamp, 'message': msg, 'timestring': timestring, 'uuid': unique_id}
    return message, unique_id

#### Debug functions
def breakpoint(message = '\n\nEnter to continue...'):
    input(message)

def print_pinecone_results(results):
    result = list()
    for m in results['matches']:
        print(m)
        info = load_json('nexus/%s.json' % m['id'])
        print(info)

if __name__ == '__main__':
    convo_length = 30
    openai.api_key = open_file(config['open_ai']['api_key'])
    pinecone.init(api_key=open_file(config['pinecone']['api_key']), environment=config['pinecone']['environment'])
    vdb = pinecone.Index(config['pinecone']['index'])
    while True:
        #### Prepare payload for pinecone upload
        payload = list()
        payload, vector = prompt_user(payload)
        #### Search for relevant messages, and generate a response
        results = vdb.query(vector=vector, top_k=convo_length)
        
        #### Load past conversations which match the user prompt and summarize
        conversation, recalled = load_conversation(results)
        print(conversation)
        breakpoint()
        notes = summarize_memories(recalled)
        print(notes)
        breakpoint()

        # prompt = open_file('prompt_response.txt').replace('<<CONVERSATION>>', conversation).replace('<<MESSAGE>>', a)
        # #### generate response, vectorize, save, etc
        # output = gpt_completion(prompt)
        # timestamp = time()
        # timestring = timestamp_to_datetime(timestamp)
        # #message = '%s: %s - %s' % ('RAVEN', timestring, output)
        # message = output
        # vector = gpt3_embedding(message)
        # unique_id = str(uuid4())
        # metadata = {'speaker': 'RAVEN', 'time': timestamp, 'message': message, 'timestring': timestring, 'uuid': unique_id}
        # save_json('nexus/%s.json' % unique_id, metadata)
        # payload.append((unique_id, vector))
        # vdb.upsert(payload)
        # print('\n\nRAVEN: %s' % output) 