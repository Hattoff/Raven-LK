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
from MemoryManagement import MemoryManager

## This class will initialize the Memory Manager and control conversation flow.
## Raven will need to be able to do the following:
    ## Keep memories in their conversation message log
    ## Keep summaries of their conversation in the message log
    ## Load higher-level summaries into their conversation log based on the current conversation
## The conversation loop will go something like this:
    ## N-number of eidetic memories are loaded into the conversation cache
        ## This number of eidetic memories will remain the same as the conversation continues.
    ## If idetic memories are compressed the next depth summary will rollup the conversation
        ## These notes will be presented to Raven before the conversation messages
    ## subsequent higher-level summaries will all be compressed in a separate loop untill the
    ## next level rollup occurs.
class ConversationManager:
    def __init__(self):
        self.__config = configparser.ConfigParser()
        self.__config.read('config.ini')
        self.__max_tokens = self.__config['open_ai']['max_token_input']

        self.__topic_notes = list()
        self.__conversation_notes = list()
        self.__conversation_log = list()
        ## TODO: memory manager will need a way to initialize the topic notes, conversation notes, and conversation log. Likely passing them to a summarization function in the conversation manager.
        self.__memory_manager = MemoryManager()

        openai.api_key = self.open_file(self.__config['open_ai']['api_key'])
        pinecone.init(api_key=self.open_file(self.__config['pinecone']['api_key']), environment=self.__config['pinecone']['environment'])

        self.__vector_db = pinecone.Index(self.__config['pinecone']['index'])

    def add_message(self, speaker, content):
        eidetic_memory, tokens, episodic_memory, episodic_tokens = self.__memory_manager.generate_eidetic_memory(speaker, content)
        if episodic_memory is not None:
            conversation_log_count = len(self.__conversation_log)
            preserve_log_count = 3
            ## Only trim conversation log if there are enough messages
            if conversation_log_count > preserve_log_count:
                ## Keep the last 3 messages for refrence
                self.__conversation_log = self.__conversation_log[__conversation_log-preserve_log_count:conversation_log_count]
                       
            ## When a memory compression occurs, get the new summary and reduce the number of visible messages in the conversation log.
            ## TODO: Compress conversation notes:
            self.summarize_conversation_notes(episodic_memory['original_summary'])
        
        ## Update the conversation log
        self.__conversation_log.append(eidetic_memory['original_summary'])
        ## Get the topics of the recent conversation and get update the topic summary
        self.extract_conversation_topic()

        prompt = self.generate_prompt
        ## TODO: Prompt Raven with newly updated prompt, extract the response, then pass it back to the parent of this object for display and continue communication loop.

        
    def compose_gpt_message(self, content, role, name=''):
        if name == '':
            return {"role":role, "content": content}
        else:
            role = 'system'
            return {"role":role,"name":name,"content": content}

    # ## Based on a conversation notes and recent messages determine if the most recent question
    # ## could be answered with the information provided of if another query is needed.
    # def determine_query(memories, conversation_notes):
    #     recent_messages = ''
    #     for mem in memories:
    #         recent_messages += ('%s\n' % format_summary_memory(mem).strip())
    #     prompt = open_file('determine_query.txt').replace('<<CONVERSATION_NOTES>>', conversation_notes).replace('<<RECENT_MESSAGES>>', recent_messages)
    #     save_prompt('query_prompt_',prompt)

    #     results = gpt_completion(prompt)
    #     save_prompt('query_prompt_results_',results)
        
    #     action_result = re.search(r"(ACTION:)(.*\n)", results)
    #     keywords_result = re.search(r"(KEYWORDS:)(.*\n)", results)
    #     reason_result = re.search(r"(REASON FOR MY DECISION:)(.*\n)", results)
    #     response_result = re.search(r"(MY RESPONSE:)(.*\n*.*)", results)

    #     action, keywords, reason, response = '','','',''
    #     if action_result is not None:
    #         if len(action_result.groups()) >= 2:
    #             action = action_result.groups()[1].strip()

    #     if keywords_result is not None:
    #         if len(keywords_result.groups()) >= 2:
    #             keywords = keywords_result.groups()[1].strip()
            
    #     if reason_result is not None:
    #         if len(reason_result.groups()) >= 2:
    #             reason = reason_result.groups()[1].strip()

    #     if response_result is not None:
    #         if len(response_result.groups()) >= 2:
    #             response = response_result.groups()[1].strip()

    #     msg = ('\nAction: %s\nKeywords: %s\nReason: %s\nResponse: %s\n' % (action, keywords, reason, response))
    #     print(msg)
    #     return action, keywords, reason, response

    # ## Based on a conversation notes and recent messages use prompt to determine
    # ## if the most recent message was a question, statement, request to modify, or request to create
    # def determine_intent(self, memories, conversation_notes):
    #     recent_messages = ''
    #     for mem in memories:
    #         recent_messages += ('%s\n' % format_summary_memory(mem).strip())
    #     prompt = open_file('determine_intention.txt').replace('<<CONVERSATION_NOTES>>', conversation_notes).replace('<<RECENT_MESSAGES>>', recent_messages)
    #     save_prompt('intent_prompt_',prompt)

    #     results = gpt_completion(prompt)
    #     save_prompt('intent_prompt_results_',results)

    #     speaker_result = re.search(r"(SPEAKER:)(.*\n)", results)
    #     intent_result = re.search(r"(INTENT:)(.*\n)", results)
    #     keywords_result = re.search(r"(KEYWORDS:)(.*\n)", results)
    #     reason_result = re.search(r"(REASON FOR MY DECISION:)(.*\n)", results)
    #     response_result = re.search(r"(MY RESPONSE:)(.*\n*.*)", results)

    #     speaker, intent, keywords, reason, response = '','','','',''
    #     if speaker_result is not None:
    #         if len(speaker_result.groups()) >= 2:
    #             speaker = speaker_result.groups()[1].strip()
        
    #     if intent_result is not None:
    #         if len(intent_result.groups()) >= 2:
    #             intent = intent_result.groups()[1].strip()

    #     if keywords_result is not None:
    #         if len(keywords_result.groups()) >= 2:
    #             keywords = keywords_result.groups()[1].strip()
            
    #     if reason_result is not None:
    #         if len(reason_result.groups()) >= 2:
    #             reason = reason_result.groups()[1].strip()

    #     if response_result is not None:
    #         if len(response_result.groups()) >= 2:
    #             response = response_result.groups()[1].strip()

    #     msg = ('\nSpeaker: %s\nIntent: %s\nKeywords: %s\nReason: %s\nResponse: %s\n' % (speaker, intent, keywords, reason, response))
    #     print(msg)
    #     return speaker, intent, keywords, reason, response

    # def subprocess_input(self, memories, notes, most_recent_message):
    #     recent_messages_combined=''
    #     for mem in memories:
    #         recent_messages_combined += ('%s\n' % format_summary_memory(mem).strip())
            
    #     print('Determining Intent...')
    #     intent_speaker, intent_intent, intent_keywords, intent_reason, intent_response = determine_intent(memories, notes)
    #     print('Determining Query...')
    #     query_action, query_keywords, query_reason, query_response = determine_query(memories, notes)

    #     if intent_intent.lower() == 'question' and intent_speaker.lower() == 'user':
    #         ## TODO: Give raven ability to query memories and files
    #         # if query_action.lower() == 'query':
    #         ## Generate a vector based on the missing information
    #         print('Fetching wiki info...')
    #         vector = gpt3_embedding('%s\n%s\n%s' % (intent_keywords, query_keywords, query_reason))
    #         wiki_pinecone_results = query_pinecone(vector, 1, 'wiki_pages')
    #         if len(wiki_pinecone_results['matches']) > 0:
    #             pinecone_result = wiki_pinecone_results['matches'][0]
    #             info = load_json('%s/%s.json' % (config['wiki']['wiki_metadata'],pinecone_result['id']))
    #             wiki_page_filename = info['filename']
    #             wiki_summary = summarize_wiki_page(wiki_page_filename)
                
    #             prompt = open_file('prompt_notes_with_wiki_summary.txt').replace('<<CONVERSATION_NOTES>>',notes).replace('<<DOCUMENT_SUMMARY>>', wiki_summary).replace('<<RECENT_MESSAGES>>', recent_messages_combined)
    #             save_prompt('response_with_wiki_page_prompt_',prompt)

    #             results = gpt_completion(prompt, tokens=300)
    #             save_prompt('response_with_wiki_page_prompt_results_',results)
    #             print('done recalling')
    #             return results
    #         else:
    #             breakpoint('no recall')
            
    #         memory_search = intent_keywords + '\n' + intent_reason
    #     # elif (intent_intent.lower() == 'query' or intent_intent.lower() == 'create' or intent_intent.lower() == 'modify') and intent_speaker.lower() == 'user':
    #         ## TODO: Have raven search wiki and determine if the file exist