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
import glob
from ConversationManagement import ConversationManager

config = configparser.ConfigParser()
config.read('config.ini')
conversation_manager = ConversationManager()

def timestamp_to_datetime(unix_time):
    return datetime.datetime.fromtimestamp(unix_time).strftime("%A, %B %d, %Y at %I:%M%p %Z")

def load_conversation(pinecone_results, current_prompt_id = ''):
    recalled = list()
    if current_prompt_id != '':
        info = load_json('nexus/%s.json' % current_prompt_id)
        recalled.append(info)

    for m in pinecone_results['matches']:
        info = load_json('nexus/%s.json' % m['id'])
        recalled.append(info)
    ordered = sorted(recalled, key=lambda d: d['time'], reverse=False)  # sort them all chronologically
    messages = [i['message'] for i in ordered]
    return '\n'.join(messages).strip(), recalled


## get user input, vectorize it, index it, save to pinecone
def prompt_user(payload = list(), cache_message = False):
    user_input = input('\n\nUSER: ')
    vector = gpt3_embedding(user_input)
    message, unique_id = generate_message_metadata('USER',user_input)
    if cache_message:
        nexus_message(message,unique_id)
        mem_wipe = open('mem_wipe.txt','a')
        mem_wipe.write('%s\n' % str(unique_id))
        mem_wipe.close()
    payload.append((unique_id, vector))
    return payload, vector, unique_id

## Generate message metadata for logging and memory indexing
def generate_message_metadata(speaker, msg, timestamp = time(), unique_id=str(uuid4())):
    timestring = timestamp_to_datetime(timestamp)
    message = {'speaker': speaker, 'time': timestamp, 'message': msg, 'timestring': timestring, 'uuid': unique_id}
    return message, unique_id

## Debug functions
def breakpoint(message = '\n\nEnter to continue...'):
    input(message)

def print_pinecone_results(results):
    result = list()
    for m in results['matches']:
        print(m)
        info = load_json('nexus/%s.json' % m['id'])
        print(info)


## Based on a conversation notes and recent messages determine if the most recent question
## could be answered with the information provided of if another query is needed.
def determine_query(memories, conversation_notes):
    recent_messages = ''
    for mem in memories:
        recent_messages += ('%s\n' % format_summary_memory(mem).strip())
    prompt = open_file('determine_query.txt').replace('<<CONVERSATION_NOTES>>', conversation_notes).replace('<<RECENT_MESSAGES>>', recent_messages)
    save_prompt('query_prompt_',prompt)

    results = gpt_completion(prompt)
    save_prompt('query_prompt_results_',results)
    
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
    print(msg)
    return action, keywords, reason, response

## Based on a conversation notes and recent messages use prompt to determine
## if the most recent message was a question, statement, request to modify, or request to create
def determine_intent(memories, conversation_notes):
    recent_messages = ''
    for mem in memories:
        recent_messages += ('%s\n' % format_summary_memory(mem).strip())
    prompt = open_file('determine_intention.txt').replace('<<CONVERSATION_NOTES>>', conversation_notes).replace('<<RECENT_MESSAGES>>', recent_messages)
    save_prompt('intent_prompt_',prompt)

    results = gpt_completion(prompt)
    save_prompt('intent_prompt_results_',results)

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
    print(msg)
    return speaker, intent, keywords, reason, response

def subprocess_input(memories, notes, most_recent_message):
    recent_messages_combined=''
    for mem in memories:
        recent_messages_combined += ('%s\n' % format_summary_memory(mem).strip())
        
    print('Determining Intent...')
    intent_speaker, intent_intent, intent_keywords, intent_reason, intent_response = determine_intent(memories, notes)
    print('Determining Query...')
    query_action, query_keywords, query_reason, query_response = determine_query(memories, notes)

    if intent_intent.lower() == 'question' and intent_speaker.lower() == 'user':
        ## TODO: Give raven ability to query memories and files
        # if query_action.lower() == 'query':
        ## Generate a vector based on the missing information
        print('Fetching wiki info...')
        vector = gpt3_embedding('%s\n%s\n%s' % (intent_keywords, query_keywords, query_reason))
        wiki_pinecone_results = query_pinecone(vector, 1, 'wiki_pages')
        if len(wiki_pinecone_results['matches']) > 0:
            pinecone_result = wiki_pinecone_results['matches'][0]
            info = load_json('%s/%s.json' % (config['wiki']['wiki_metadata'],pinecone_result['id']))
            wiki_page_filename = info['filename']
            wiki_summary = summarize_wiki_page(wiki_page_filename)
            
            prompt = open_file('prompt_notes_with_wiki_summary.txt').replace('<<CONVERSATION_NOTES>>',notes).replace('<<DOCUMENT_SUMMARY>>', wiki_summary).replace('<<RECENT_MESSAGES>>', recent_messages_combined)
            save_prompt('response_with_wiki_page_prompt_',prompt)

            results = gpt_completion(prompt, tokens=300)
            save_prompt('response_with_wiki_page_prompt_results_',results)
            print('done recalling')
            return results
        else:
            breakpoint('no recall')
        
        memory_search = intent_keywords + '\n' + intent_reason
    # elif (intent_intent.lower() == 'query' or intent_intent.lower() == 'create' or intent_intent.lower() == 'modify') and intent_speaker.lower() == 'user':
        ## TODO: Have raven search wiki and determine if the file exist

def prompt_gpt_completion(self, messages, temp=0.0, tokens=400, stop=['USER:', 'RAVEN:']):
        engine = self.__config['open_ai']['model']
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

                # self.print_response_stats(response)
                response_id = str(response['id'])
                prompt_tokens = int(response['usage']['prompt_tokens'])
                completion_tokens = int(response['usage']['completion_tokens'])
                total_tokens = int(response['usage']['total_tokens'])
                response_obj = json.loads(response['choices'][0]['message']['content'].strip())
                return response_obj, total_tokens
            except Exception as oops:
                retry += 1
                if retry >= max_retry:
                    return "GPT3.5 error: %s" % oops
                print('Error communicating with OpenAI:', oops)
                sleep(2)

def post_conversation(messages):
    output = prompt_gpt3_embedding(prompt)
    return output

if __name__ == '__main__':
    active_tokens = 0
    messages = list()
    while True:
        ## Do things
        user_input = input('USER: ')
        conversation_manager.log_message('USER', user_input)
        raven_response = conversation_manager.generate_response()
        print('\nRAVEN: %s\n' % raven_response)
        conversation_manager.log_message('RAVEN', raven_response)
        breakpoint('\n\n........')