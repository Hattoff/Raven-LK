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

def print_response_stats(response):
    response_id = ('\nResponse %s' % str(response['id']))
    prompt_tokens = ('\nPrompt Tokens: %s' % (str(response['usage']['prompt_tokens'])))
    completion_tokens = ('\nCompletion Tokens: %s' % str(response['usage']['completion_tokens']))
    total_tokens = ('\nTotal Tokens: %s\n' % (str(response['usage']['total_tokens'])))
    breakpoint(response_id + prompt_tokens + completion_tokens + total_tokens)

def gpt_completion(prompt, engine=config['open_ai']['model'], temp=0.0, top_p=1.0, tokens=400, freq_pen=0.0, pres_pen=0.0, stop=['USER:', 'RAVEN:']):
    if 'gpt-3.5-turbo' in engine:
        return gpt3_5_completion(prompt, engine, temp, top_p, tokens, freq_pen, pres_pen, stop)
    else:
        return gpt3_completion(prompt, engine, temp, top_p, tokens, freq_pen, pres_pen, stop)

def gpt3_5_completion(prompt, engine, temp, top_p, tokens, freq_pen, pres_pen, stop):
    # breakpoint('Using gpt3.5')
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

            print_response_stats(response)
            # breakpoint(response['choices'])
            # breakpoint(response['choices'][0]['message'])
            # breakpoint(response['choices'][0]['message']['content'])
            text = response['choices'][0]['message']['content'].strip()
            text = re.sub('[\r\n]+', '\n', text)
            text = re.sub('[\t ]+', ' ', text)
            filename = '%s_gpt3.txt' % time()
            foldername = config['raven']['gpt_log_dir']
            if not os.path.exists(foldername):
                os.makedirs(foldername)
            save_file('%s/%s' % (foldername,filename), prompt + '\n\n==========\n\n' + text)
            breakpoint(text)
            breakpoint('\nAbove are the text results from the prompt...')
            return text
        except Exception as oops:
            retry += 1
            if retry >= max_retry:
                return "GPT3.5 error: %s" % oops
            print('Error communicating with OpenAI:', oops)
            sleep(2)

def gpt3_completion(prompt, engine, temp, top_p, tokens, freq_pen, pres_pen, stop):
    # breakpoint('Using gpt3')
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

def load_conversation(results, current_prompt_id = ''):
    recalled = list()
    if current_prompt_id != '':
        breakpoint('Including current prompt id...')
        info = load_json('nexus/%s.json' % current_prompt_id)
        recalled.append(info)

    for m in results['matches']:
        # breakpoint('Loading memory: ' + m['id'])
        info = load_json('nexus/%s.json' % m['id'])
        recalled.append(info)
    ordered = sorted(recalled, key=lambda d: d['time'], reverse=False)  # sort them all chronologically
    messages = [i['message'] for i in ordered]
    return '\n'.join(messages).strip(), recalled

def summarize_memories(memories):  # summarize a block of memories into one payload
    # breakpoint('fetching memories...')
    memories = sorted(memories, key=lambda d: d['time'], reverse=False)  # sort them chronologically
    blocked_summary = '' ## TODO: Implement block summary
    chunked_summary = ''
    blocks = chunk_memories(memories)
    block_count = len(blocks)
    for chunks in blocks:
        chunked_summary = '' # Combine all chunk summaries into one long prompt and use it for context
        for chunk in chunks:
            chunked_message = '' # Combine all memories into a long prompt and have it summarized
            for mem in chunk:
                message = format_summary_memory(mem)
                chunked_message += message.strip() + '\n\n'
            chunked_message = chunked_message.strip()
            chunked_prompt = open_file('prompt_notes.txt').replace('<<INPUT>>', chunked_summary + chunked_message)
            chunk_prompt_filename = 'summary_chunk_prompt_%s.json' % time()
            save_json('summary_prompts/%s' % chunk_prompt_filename, chunked_prompt)

            chunked_notes = ''
            chunked_notes = gpt_completion(chunked_prompt)
            # print(chunked_notes)
            # breakpoint('above are the chunked notes...')
            chunk_notes_filename = 'summary_chunk_notes_%s.json' % time()
            save_json('summary_prompts/%s' % chunk_notes_filename, chunked_notes)
            # breakpoint('Finished with first prompt')
            chunked_summary = chunked_notes.strip() + '\n'
    return chunked_summary.strip()

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
def prompt_user(payload = list(), cache_message = False):
    user_input = input('\n\nUSER: ')
    vector = gpt3_embedding(user_input)
    message, unique_id = generate_message_metadata('USER',user_input)
    if cache_message:
        index_message(message,unique_id)
        mem_wipe = open('mem_wipe.txt','a')
        mem_wipe.write('%s\n' % str(unique_id))
        mem_wipe.close()
    payload.append((unique_id, vector))
    return payload, vector, unique_id

#### Index message with unique id and save it locally for memeory recall
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


#### Get the last n conversation memories and summarize them
# def initialize_context(message_history_count = 30):


#### Based on a conversation notes and recent messages determine if the most recent question
#### could be answered with the information provided of if another query is needed.
def determine_query(memories, conversation_notes):
    recent_messages = ''
    for mem in memories:
        recent_messages += ('%s\n' % format_summary_memory(mem).strip())
    prompt = open_file('determine_query.txt').replace('<<CONVERSATION_NOTES>>', conversation_notes).replace('<<RECENT_MESSAGES>>', recent_messages)

    prompt_filename = 'query_prompt_%s.json' % time()
    save_json('summary_prompts/%s' % prompt_filename, prompt)

    results = gpt_completion(prompt)

    prompt_results_filename = 'query_prompt_results_%s.json' % time()
    save_json('summary_prompts/%s' % prompt_results_filename, results)
    
    action_result = re.search(r"(ACTION:)(.*\n)", results)
    keywords_result = re.search(r"(KEYWORDS:)(.*\n)", results)
    reason_result = re.search(r"(REASON FOR MY DECISION:)(.*\n)", results)
    response_result = re.search(r"(MY RESPONSE:)(.*\n*.*)", results)

    action, keywords, reason, response = '','','',''
    if action_result is not None:
        if len(action_result.groups()) >= 2:
            action = action_result.groups()[1].strip()

    if keywords_result is not None:
        if len(keywords_result.groups()) >= 2:
            keywords = keywords_result.groups()[1].strip()
        
    if reason_result is not None:
        if len(reason_result.groups()) >= 2:
            reason = reason_result.groups()[1].strip()

    if response_result is not None:
        if len(response_result.groups()) >= 2:
            response = response_result.groups()[1].strip()

    msg = ('\nAction: %s\nKeywords: %s\nReason: %s\nResponse: %s\n' % (action, keywords, reason, response))
    breakpoint(msg)
    return action, keywords, reason, response

#### Based on a conversation notes and recent messages use prompt to determine
#### if the most recent message was a question, statement, request to modify, or request to create
def determine_intent(memories, conversation_notes):
    recent_messages = ''
    for mem in memories:
        recent_messages += ('%s\n' % format_summary_memory(mem).strip())
    prompt = open_file('determine_intention.txt').replace('<<CONVERSATION_NOTES>>', conversation_notes).replace('<<RECENT_MESSAGES>>', recent_messages)

    prompt_filename = 'intent_prompt_%s.json' % time()
    save_json('summary_prompts/%s' % prompt_filename, prompt)

    results = gpt_completion(prompt)

    prompt_results_filename = 'intent_prompt_results_%s.json' % time()
    save_json('summary_prompts/%s' % prompt_results_filename, results)
    
    # breakpoint(results)

    speaker_result = re.search(r"(SPEAKER:)(.*\n)", results)
    intent_result = re.search(r"(INTENT:)(.*\n)", results)
    keywords_result = re.search(r"(KEYWORDS:)(.*\n)", results)
    reason_result = re.search(r"(REASON FOR MY DECISION:)(.*\n)", results)
    response_result = re.search(r"(MY RESPONSE:)(.*\n*.*)", results)

    speaker, intent, keywords, reason, response = '','','','',''
    if speaker_result is not None:
        if len(speaker_result.groups()) >= 2:
            speaker = speaker_result.groups()[1].strip()
    
    if intent_result is not None:
        if len(intent_result.groups()) >= 2:
            intent = intent_result.groups()[1].strip()

    if keywords_result is not None:
        if len(keywords_result.groups()) >= 2:
            keywords = keywords_result.groups()[1].strip()
        
    if reason_result is not None:
        if len(reason_result.groups()) >= 2:
            reason = reason_result.groups()[1].strip()

    if response_result is not None:
        if len(response_result.groups()) >= 2:
            response = response_result.groups()[1].strip()

    msg = ('\nSpeaker: %s\nIntent: %s\nKeywords: %s\nReason: %s\nResponse: %s\n' % (speaker, intent, keywords, reason, response))
    breakpoint(msg)
    return speaker, intent, keywords, reason, response

if __name__ == '__main__':
    convo_length = 30
    openai.api_key = open_file(config['open_ai']['api_key'])
    pinecone.init(api_key=open_file(config['pinecone']['api_key']), environment=config['pinecone']['environment'])
    vdb = pinecone.Index(config['pinecone']['index'])
    while True:
        #### Prepare payload for pinecone upload
        payload = list()
        payload, vector, prompt_id = prompt_user(payload, True)
        #### Search for relevant messages, and generate a response
        breakpoint('Searching for context...')
        results = vdb.query(vector=vector, top_k=convo_length)
        #### Search for story elements related to the topic needing queried.
        # results = vdb.query(vector=vector, top_k=convo_length, namespace="story-elements")
        #### Load past conversations which match the user prompt and summarize
        breakpoint('Loading conversation...')
        conversation, recalled = load_conversation(results, prompt_id)
        notes = summarize_memories(recalled)
        tmp_msg = recalled.copy()
        sorted(tmp_msg, key=lambda d: d['time'], reverse=True)
        recent_messages = tmp_msg[0:4]
        breakpoint('Determining Intent...')
        intent_speaker, intent_intent, intent_keywords, intent_reason, intent_response = determine_intent(recent_messages, notes)
        breakpoint('Determining Query...')
        query_action, query_keywords, query_reason, query_response = determine_query(recent_messages, notes)
        breakpoint('end of main loop')