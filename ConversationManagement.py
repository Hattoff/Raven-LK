import configparser
import os
import json
import glob
from time import time,sleep
import datetime
from uuid import uuid4
import pinecone
import openai
import tiktoken
import re
from MemoryManagement import MemoryManager

class ConversationManager:
    def __init__(self):
        self.__config = configparser.ConfigParser()
        self.__config.read('config.ini')
        self.__memory_manager = MemoryManager()
        self.__eidetic_memory_log = self.MemoryLog(1600,4)
        self.__episodic_memory_log = self.MemoryLog(1600,4)
        
        openai.api_key = self.open_file(self.__config['open_ai']['api_key'])
        pinecone.init(api_key=self.open_file(self.__config['pinecone']['api_key']), environment=self.__config['pinecone']['environment'])
        self.__vector_db = pinecone.Index(self.__config['pinecone']['index'])

        self.load_state()

    class MemoryLog:
        def __init__(self, max_log_tokens, min_log_count):
            self.__max_log_tokens = int(max_log_tokens)
            self.__min_log_count = int(min_log_count)
            self.__memories = list()
            self.__token_count = 0
            self.__memory_string = ''
            
        def add(self, memory, tokens):
            self.__memories.append((memory,tokens))
            self.__token_count += int(tokens)
            self.__check_memory_log()
            self.__generate_memory_string()

        ## Checks to see if the memory list needs rebuilt
        def __check_memory_log(self):
            memory_count = len(self.__memories)
            ## Note that if the memory count is equal to the min log count then I don't rebuild the list
            if self.__token_count > self.__max_log_tokens and memory_count > self.__min_log_count:
                ## Rebuild the memory list with a minumum number of memories, but continue to add memories until token count is reached
                memories = self.__memories.copy()
                memories.reverse()
                self.__memories.clear()
                token_count = 0
                for m in memories:
                    ## Don't exit until the minimum number of memories have been added
                    if len(self.__memories) < self.__min_log_count:
                        token_count += int(m[1])
                        self.__memories.append(m)
                    else: 
                        ## Memory minimum has been reached, add memories until token space runs out
                        if token_count + int(m[1]) <= self.__max_log_tokens:
                            token_count += int(m[1])
                            self.__memories.append(m)
                            continue
                        else:
                            ## Exit loop early if no more token space
                            break
                ## The memory list is backwards after this process so correct it
                self.__memories.reverse()
                ## Once complete, update token count
                self.__token_count = token_count
        
        def __generate_memory_string(self):
            if len(self.__memories) > 0:
                if 'content' in self.__memories[0][0]:
                    self.__memory_string = '\n'.join('%s: %s' % (m[0]['speaker'],m[0]['content']) for m in self.__memories)
                else:
                    self.__memory_string = '\n'.join(m[0]['original_summary'] for m in self.__memories)

        @property
        def memory_string(self):
            return self.__memory_string

        @property
        def memory_count(self):
            return len(self.__memories)

    ## Load recent conversation log and conversation notes
    def load_state(self):
        if self.__memory_manager.get_cache_memory_count(0) > 0:
            eidetic_memories = self.__memory_manager.get_memories_from_cache(0)
            for m in eidetic_memories:
                self.__eidetic_memory_log.add(m, m['original_tokens'])
                print('state loaded...')

        cache_count = self.__memory_manager.cache_count
        for d in list(reversed(range(cache_count)))[0:cache_count-1]:
            if self.__memory_manager.get_cache_memory_count(d) > 0:
                episodic_memories = self.__memory_manager.get_memories_from_cache(d)
                for m in episodic_memories:
                    self.__episodic_memory_log.add(m, m['original_tokens'])

        self.reestablish_conversation_context()


    ######## This is not working so don't run it...#######
    ## My thoughts are this... on rare occasions there will be a rollup on all levels of
    ## memory depth except the heighest depth. This means that, in order for you to get
    ## the recent conversation context you will need to walk your way down the chain of memories
    ## in the cache until you get to the most recent eidetic memories list so you can load it...
    ########
    ## Solutions include:
    ##  1.) Every time the memory manager flushes memories, it keeps a cache of the flushed
    ##      memories for reference, or at least the ids and reloads them at initialization. -- I like this one best
    ##  2.) You attempt to figure out the insanity that is currently below this comment where I attempt to iterate
    ##      down the chain of memories to find the eidetic memories most recently flushed... -- I hate this one the most
    ## Note, wine didn't help this problem... I will come back to this tomorrow, but will commit in case I
    ##      get another surprise update or something is lost...
    ## Reload the most recent messages
    def reestablish_conversation_context(self, reload_message_count = 2):
        if self.__memory_manager.get_cache_memory_count(0) > 0:
            memory_cache_count = self.__memory_manager.cache_count
            for c in range(memory_cache_count):
                eidetic_memories = self.__memory_manager.get_memories_from_cache(c)
                memory_count = len(eidetic_memories)
                if memory_count > 0:
                    reload_count = min(int(reload_message_count), memory_count)
                    eidetic_memories.reverse()
                    self.initalize_reloaded_context(eidetic_memories, memory_count)
                    break
                elif c < memory_cache_count-1:
                    episodic_memories = self.__memory_manager.get_memories_from_cache(c+1)
                    if len(episodic_memories) > 0:
                        eidetic_memory_ids = episodic_memories[0]['lower_memory_ids']
                        eidetic_memories = []
                        depth = c+1
                        memory_folder = self.__config['memory_management']['memory_namespace_template'] % str(depth)
                        memory_file_path = '%s/%s' % (self.__config['memory_management']['memory_stash_dir'], memory_folder)
                        for mem_id in idetic_memory_ids:
                            try:
                                memory_path = '%s/%s.json' % (memory_file_path, memory_folder, mem_id)
                                memory = self.load_json(memory_path)
                                eidetic_memories.append(memory)
                            except Exception as err:
                                print('ERROR: unable to load memory %s.json from %s' % (mem_id, memory_folder))
                        memory_count = len(eidetic_memories)
                        if memory_count > 0:
                            reload_count = min(int(reload_message_count), memory_count)
                            eidetic_memories = sorted(eidetic_memories, key=lambda x: x[''], reverse()
                            self.initalize_reloaded_context(eidetic_memories, memory_count)
                            break

    def initalize_reloaded_context(self, eidetic_memories, reload_count):
        ## Display the last two messages with dates and speaker stamps for context.
        for i in reversed(range(reload_count)):
            print('\n[%s] %s: %s\n' % (eidetic_memories[i]['original_timestring'],eidetic_memories[i]['speaker'],eidetic_memories[i]['content']))
        ## If the last speaker was the user, prompt Raven to respond to their last message
        if eidetic_memories[0]['speaker'] == 'USER':
                self.generate_response()
            
    def log_message(self, speaker, content):
        eidetic_memory, eidetic_tokens, episodic_memory, episodic_tokens = self.__memory_manager.create_new_memory(speaker, content)
        self.__eidetic_memory_log.add(eidetic_memory, eidetic_tokens)
        if episodic_memory is not None:
            self.__episodic_memory_log.add(episodic_memory, episodic_tokens)
    
    def generate_response(self):
        prompt_obj = self.load_json('%s/%s.json' % (self.__config['conversation_management']['conversation_management_dir'], 'conversation_prompt'))

        conversation_notes_obj = prompt_obj['conversation_prompt']['conversation_notes']
        conversation_log_obj = prompt_obj['conversation_prompt']['conversation_log']
        
        episodic_string = self.__episodic_memory_log.memory_string
        eidetic_string = self.__eidetic_memory_log.memory_string
        if self.__episodic_memory_log.memory_count > 0:
            prompt_instructions = '%s and %s' % (conversation_notes_obj['instruction'], conversation_log_obj['instruction'])
            prompt_body = '%s\n%s\n%s\n%s' % (conversation_notes_obj['prompt_body'], episodic_string, conversation_log_obj['prompt_body'], eidetic_string)
        else:
            prompt_instructions = conversation_log_obj['instruction']
            prompt_body = '%s\n%s' % (conversation_log_obj['prompt_body'], eidetic_string)

        prompt_template = prompt_obj['conversation_prompt']['prompt_template']
        core_heuristic = prompt_obj['conversation_prompt']['core_heuristic']

        prompt = prompt_template % (core_heuristic, prompt_instructions, prompt_body)
        ## Save prompt for debug
        prompt_stash_dir = self.__config['conversation_management']['conversation_prompt_stash_dir']
        prompt_path = '%s/%s.txt' % (prompt_stash_dir, str(time()))
        self.save_file(prompt_path, prompt)

        temperature = float(prompt_obj['conversation_prompt']['temperature'])
        response_tokens = int(prompt_obj['conversation_prompt']['response_tokens'])
        gpt_messages = list()
        gpt_messages.append(self.compose_gpt_message(prompt, 'user'))

        response, tokens = self.gpt_completion(gpt_messages, temperature, response_tokens)

        ## Save response for debug
        conversation_stash_dir = self.__config['conversation_management']['conversation_stash_dir']
        conversation_path = '%s/%s.txt' % (conversation_stash_dir, str(time()))
        self.save_file(conversation_path, response)

        return response
        
    def save_file(self, filepath, content):
        with open(filepath, 'w', encoding='utf-8') as outfile:
            outfile.write(content)

    def open_file(self, filepath):
        with open(filepath, 'r', encoding='utf-8') as infile:
            return infile.read()

    def load_json(self, filepath):
        with open(filepath, 'r', encoding='utf-8') as infile:
            return json.load(infile)

    def save_json(self, filepath, payload):
        with open(filepath, 'w', encoding='utf-8') as outfile:
            json.dump(payload, outfile, ensure_ascii=False, sort_keys=True, indent=2)

    def compose_gpt_message(self, content, role, name=''):
        if name == '':
            return {"role":role, "content": content}
        else:
            role = 'system'
            return {"role":role,"name":name,"content": content}

    

    def gpt3_embedding(self, content):
        engine = self.__config['open_ai']['input_engine']
        content = content.encode(encoding='ASCII',errors='ignore').decode()  # fix any UNICODE errors
        response = openai.Embedding.create(input=content,engine=engine)
        vector = response['data'][0]['embedding']  # this is a normal list
        return vector

    def gpt_completion(self, messages, temp=0.0, tokens=400, stop=['USER:', 'RAVEN:']):
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

                response_id = str(response['id'])
                prompt_tokens = int(response['usage']['prompt_tokens'])
                completion_tokens = int(response['usage']['completion_tokens'])
                total_tokens = int(response['usage']['total_tokens'])
                response_text = response['choices'][0]['message']['content'].strip()
                response_text = re.sub('[\r\n]+', '\n', response_text)
                response_text = re.sub('[\t ]+', ' ', response_text)
                return response_text, total_tokens
            except Exception as oops:
                retry += 1
                if retry >= max_retry:
                    return "GPT3.5 error: %s" % oops, -1
                print('Error communicating with OpenAI:', oops)
                sleep(10)

    def print_response_stats(self, response):
        response_id = ('\nResponse %s' % str(response['id']))
        prompt_tokens = ('\nPrompt Tokens: %s' % (str(response['usage']['prompt_tokens'])))
        completion_tokens = ('\nCompletion Tokens: %s' % str(response['usage']['completion_tokens']))
        total_tokens = ('\nTotal Tokens: %s\n' % (str(response['usage']['total_tokens'])))
        print(response_id + prompt_tokens + completion_tokens + total_tokens)

    def get_token_estimate(self, content):
        content = content.encode(encoding='ASCII',errors='ignore').decode()  # fix any UNICODE errors
        encoding = tiktoken.encoding_for_model(str(self.__config['open_ai']['model']))
        tokens = encoding.encode(content)
        token_count = len(tokens)
        return token_count
