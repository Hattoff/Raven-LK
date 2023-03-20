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

## NOTE:Likely will move the OpenAi stuff later, not sure.
import openai

##### NOTE: Token counts should leave enough room for a variety of prompt instructions, since their use may vary. I am thinking of leaving a buffer of 1000 to ensure there is enough room, but I will generalize it so adjustments can be made easily

##### NOTE: Memories may need to be chunked if the token estimation is significantly off. This can be done on a prompt-by-prompt basis

## When a new message is added the memory manager will assess when memory compression should occur.
class MemoryManager:
    def __init__(self):
        self.__config = configparser.ConfigParser()
        self.__config.read('config.ini')
        self.__token_buffer = int(self.__config['memory_management']['token_buffer'])
        self.__max_tokens = int(self.__config['open_ai']['max_token_input'])
        self.__episodic_memory_caches = [] # index will represent memory depth, useful for dynamic memory expansion
        self.__max_episodic_depth = 2 # will restrict memory expansion. 0 is unlimited depth.
        self.debug_messages_enabled = True
        
        openai.api_key = self.open_file(self.__config['open_ai']['api_key'])
        pinecone.init(api_key=self.open_file(self.__config['pinecone']['api_key']), environment=self.__config['pinecone']['environment'])

        self.__vector_db = pinecone.Index(self.__config['pinecone']['index'])

        ## When initialized, attempt to load cached state, otherwise make a new state
        if not (self.load_state()):
            self.create_state()

    ## Houses memories of a particular depth. Each change will trigger will be followed with a state save
    class MemoryCache:
        ## Depth should not be changed after initialization.
        ## If cache is not empty then load it, otherwise start fresh.Cache should be JSON.
        def __init__(self, depth, token_buffer, max_tokens, cache=None):
            self.__id = str(uuid4())
            self.__depth = int(depth)
            self.__token_buffer = int(token_buffer)
            self.__max_tokens = int(max_tokens)
            self.__memories = list()
            self.__previous_memory_ids = list()
            ## token count is the token estimate for each memory without modification
            self.__token_count = 0
            if cache is not None:
                self.import_cache(cache)

        @property
        def depth(self):
            return self.__depth

        @property
        def token_buffer(self):
            return self.self.__token_buffer

        ## Returns a copy of memories; useful for compression and will not bork stuff when flushed
        @property
        def memories(self):
            return self.__memories.copy()

        @property
        def memory_count(self):
            return len(self.__memories)

        @property
        def token_count(self):
            return self.__token_count

        ## Return a list of all ids in memories tracked by the memory cache
        @property
        def memory_ids(self):
            ids = list(map(lambda m: m['id'], self.__memories))
            return ids
        
        @property
        def previous_memory_ids(self):
            return self.__previous_memory_ids

        ## Set all class attributes from the cache JSON
        def import_cache(self, cache):
            self.__id = str(cache['id'])
            self.__depth = int(cache['depth'])
            self.__token_buffer = int(cache['token_buffer'])
            self.__max_tokens = int(cache['max_tokens'])
            self.__token_count = int(cache['token_count'])
            self.__memories = list(cache['memories'])
            self.__previous_memory_ids = list(cache['previous_memory_ids'])
            ## TODO: If cache is invalid, throw error.

        ## Return a JSON version of this object to save
        @property
        def cache(self):
            timestamp = time()
            timestring = (datetime.datetime.fromtimestamp(timestamp).strftime("%A, %B %d, %Y at %I:%M:%S%p %Z")).strip()
            cache = {
                "id":self.__id,
                "depth":self.__depth,
                "token_buffer":self.__token_buffer,
                "max_tokens":self.__max_tokens,
                "token_count":self.__token_count,
                "memories":self.__memories,
                "previous_memory_ids": self.__previous_memory_ids,
                "timestamp": timestamp,
                "timestring": timestring
            }
            return cache

        ## Memories are in the raven eidetic or episodic JSON format
        ## Memory space check is NOT enforced so check it before adding a memory
        def add_memory(self, memory, number_of_tokens):
            self.__token_count += int(number_of_tokens)
            self.__memories.append(memory)

        ## Before adding a memory, check to see if there will be space with next memory.
        def has_memory_space(self, next_number_of_tokens):
            if self.__token_count + int(next_number_of_tokens) <= (self.__max_tokens - self.__token_buffer):
                return True
            else:
                return False

        def flush_memory_cache(self):
            ## Cache the current caches memory ids and their timestamps for loading purposes
            self.__previous_memory_ids.clear()
            self.__previous_memory_ids = list(map(lambda m : m['id']),self.__memories)
            ## Clear memory cache and reset token count
            self.__memories.clear()
            self.__token_count = 0
            self.__id = str(uuid4())

    ## Return the number of memories of a given cache
    def get_cache_memory_count(self,depth):
        if int(depth) > len(self.__episodic_memory_caches):
            return -1
        return self.__episodic_memory_caches[int(depth)].memory_count

    ## Return the list of memories currently in cache
    def get_memories_from_cache(self, depth):
        return self.__episodic_memory_caches[int(depth)].memories.copy()

    ## Return id list of memories in previous cache before it was flushed.
    def get_previous_memory_ids_from_cache(self, depth):
        return self.__episodic_memory_caches[int(depth)].previous_memory_ids.copy()

    @property
    def cache_count(self):
        return len(self.__episodic_memory_caches)

    ## Load JSON object representing state. State is all memory caches not yet summarized, active tasks, and active context.
    def load_state(self):
        ## Get all json files ordered by name. The name should have a timestamp prefix.
        files = glob.glob('%s/*.json' % (self.__config['memory_management']['memory_state_dir']))
        files.sort(key=os.path.basename, reverse=True)

        # there were no state backups so make and implement a new one
        if len(list(files)) <= 0:
            self.debug_message("no state backups found...")
            return False

        self.debug_message("state backups loaded...")
        state_path = list(files)[0].replace('\\','/')
        state = self.load_json(state_path)

        for cache in state['memory_caches']:
            depth = cache['depth']
            self.__episodic_memory_caches.append(self.MemoryCache(depth, self.__token_buffer, self.__max_tokens, cache))

        ## TODO:
        ## load state, if that fails, try backup files
        ## If memory caches exceed max episodic depth then throw error or set new max
        ## initialize missing memory caches
        return True

    def create_state(self):
        for i in range(self.__max_episodic_depth+1):
            ## Initialize all episodic memory caches
            self.__episodic_memory_caches.append(self.MemoryCache(i, self.__token_buffer, self.__max_tokens))
        self.save_state()

    def save_state(self):
        ## TODO: Restrict backups to max_backup_states
        timestamp = time()
        timestring = self.timestamp_to_datetime(timestamp)
        unique_id = str(uuid4())

        memory_caches = list()
        for i in range(len(self.__episodic_memory_caches)):
            memory_caches.append(self.__episodic_memory_caches[i].cache)

        state = {
            "id":unique_id,
            "memory_caches":memory_caches,
            "timestamp":timestamp,
            "timestring":timestring
        }

        filename = ('%s_memory_state_%s.json' %(str(timestamp),unique_id))
        filepath = ('%s/%s' % (self.__config['memory_management']['memory_state_dir'], filename))
        self.save_json(filepath, state)
        self.debug_message('State saved...')

    def timestamp_to_datetime(self, unix_time):
        return (datetime.datetime.fromtimestamp(unix_time).strftime("%A, %B %d, %Y at %I:%M:%S%p %Z")).strip()

    def open_file(self, filepath):
        with open(filepath, 'r', encoding='utf-8') as infile:
            return infile.read()

    def save_file(self, filepath, content):
        with open(filepath, 'w', encoding='utf-8') as outfile:
            outfile.write(content)

    def load_json(self, filepath):
        with open(filepath, 'r', encoding='utf-8') as infile:
            return json.load(infile)

    def save_json(self, filepath, payload):
        with open(filepath, 'w', encoding='utf-8') as outfile:
            json.dump(payload, outfile, ensure_ascii=False, sort_keys=True, indent=2)

    ## Assemble and index eidetic memories from a message by a speaker
    ## Eidetic memory is the base form of all episodic memory
    def generate_eidetic_memory(self, speaker, content):
        timestamp = time()
        self.debug_message('Generating eidetic memory...')
        unique_id = str(uuid4())
        timestring = self.timestamp_to_datetime(timestamp)
        depth = 0

        ## Execute sub-prompts to get metadata and summaries of the memories
        eidetic_keywords_response = self.generate_memory_element('keywords', '%s: %s' % (speaker, content), depth)
        eidetic_keywords = self.cleanup_keywords_memory_response(eidetic_keywords_response)

        eidetic_summary_result = self.generate_memory_element('summary', '%s: %s' % (speaker, content), depth)
        eidetic_summary = self.cleanup_summary_memory_response(eidetic_summary_result)
        eidetic_tokens = self.get_token_estimate(eidetic_summary)

        content_tokens = self.get_token_estimate(content)

        ## Build episodic memory object
        eidetic_memory = {
            "id": unique_id,
            "episodic_parent_id":"",
            "speaker": speaker,
            "content": content,
            "content_tokens":content_tokens,
            "original_summary": eidetic_summary,
            "original_timestamp": timestamp,
            "original_timestring": timestring,
            "original_tokens":int(eidetic_tokens),
            "is_retconned": False,
            "retcon_summary": "",
            "retcon_rank":0.0,
            "retcon_time": "",
            "retcon_tokens":0,
            "keywords": eidetic_keywords,
            "depth":int(depth)
        }
        return eidetic_memory, eidetic_tokens

    ## Assemble an eposodic memory from a collection of eidetic or lower-depth episodic memories
    def generate_episodic_memory(self, memories, depth):
        timestamp = time()
        self.debug_message('Generating episodic memory of cache depth (%s)...' % str(depth))
        unique_id = str(uuid4())
        timestring = self.timestamp_to_datetime(timestamp)

        ## Append all summaries together, get a token total, and get a list of memory ids
        content = ''
        content_tokens = 0
        memory_ids = []
        for memory in memories:
            content += '%s\n' % (memory['original_summary'])
            content_tokens += int(memory['original_tokens'])
            memory_ids.append(str(memory['id']))

        ## Execute sub-prompts to get metadata and summaries of the memories
        episodic_keywords_response = self.generate_memory_element('keywords', content, depth)
        episodic_keywords = self.cleanup_keywords_memory_response(episodic_keywords_response)

        episodic_summary_result = self.generate_memory_element('summary', content, depth)
        episodic_summary = self.cleanup_summary_memory_response(episodic_summary_result)
        episodic_tokens = self.get_token_estimate(episodic_summary)

        ## Build episodic memory object
        episodic_memory = {
            "id": unique_id,
            "episodic_parent_id":"",
            "lower_memory_ids": memory_ids,
            "original_summary": episodic_summary,
            "original_timestamp": timestamp,
            "original_timestring": timestring,
            "original_tokens":int(episodic_tokens),
            "is_retconned": False,
            "retcon_summary": "",
            "retcon_rank":0.0,
            "retcon_time": "",
            "retcon_tokens":0,
            "keywords": episodic_keywords,
            "depth": int(depth)
        }
        return episodic_memory, episodic_tokens

    def cleanup_keywords_memory_response(self, keywords):
        if type(keywords) == list:
                return keywords
        keywords_obj = {}
        try:
            keywords_obj = json.loads(keywords)
        except Exception as err:
            print('ERROR: unable to parse the json object when creating eidetic memory keyword element')
            print('Value from GPT:\n\n%s' % keywords)
            self.breakpoint('\n\npausing...')
            return []

        dict_keys = list(keywords_obj.keys())
        if len(dict_keys) > 1:
            print('ERROR: unknown response for keyword memory element')
            print('Value from GPT:\n\n%s' % keywords)
            print(keywords_obj)
            self.breakpoint('\n\npausing...')
            return []
        else:
            key = dict_keys[0]
            if type(keywords_obj[key]) == list:
                return keywords_obj[key]
            else:
                print('ERROR: unknown response for keyword memory element')
                print('Value from GPT:\n\n%s' % keywords)
                print(keywords_obj)
                self.breakpoint('\n\npausing...')
                return []

    ## Strip away unwanted characters from summary response
    def cleanup_summary_memory_response(self, response):
        response = response.strip()
        response = re.sub('[\r\n]+', '\n', response)
        response = re.sub('[\t ]+', ' ', response)
        return response

    ## Caching memories may cascade and compress higher depth caches
    ## Check to see if cache as room, if so then add memory, otherwise compress that cache before adding
    def cache_memory(self, memory, tokens, depth):
        self.debug_message('adding memory to cache (%s)' % str(depth))
        if self.__episodic_memory_caches[int(depth)].has_memory_space(tokens):
            self.debug_message('There is enough space in the cache...')
            self.__episodic_memory_caches[int(depth)].add_memory(memory, tokens)
        else:
            self.debug_message('There is not enough space in the cache (%s), compressing...' % str(depth))    
            self.compress_memory_cache(depth)
            self.debug_message('Flushing cache of depth (%s)...' % str(depth))
            self.__episodic_memory_caches[int(depth)].flush_memory_cache()
            # if not self.__episodic_memory_caches[int(depth)].has_memory_space(tokens):
            #     ## TODO: There is an error because the next message is exceptionally long.
            # Add the new message to the recently flushed memory cache
            self.__episodic_memory_caches[int(depth)].add_memory(memory, tokens)
        self.debug_message('Saving state...')
        self.index_memory(memory)
        self.save_state()

    def create_new_memory(self, speaker, content):
        episodic_memory = None
        episodic_tokens = 0
        depth = 0
        memory, tokens = self.generate_eidetic_memory(speaker, content)
        self.debug_message('adding memory to cache (%s)' % str(depth))
        if self.__episodic_memory_caches[depth].has_memory_space(tokens):
            self.debug_message('There is enough space in the cache...')
            self.__episodic_memory_caches[depth].add_memory(memory, tokens)
        else:
            self.debug_message('There is not enough space in the cache, compressing...')    
            episodic_memory, episodic_tokens = self.compress_memory_cache(depth)
            self.__episodic_memory_caches[depth].flush_memory_cache()
            self.__episodic_memory_caches[depth].add_memory(memory, tokens)
        self.debug_message('Saving state...')
        self.index_memory(memory)
        self.save_state()
        return memory, tokens, episodic_memory, episodic_tokens

    ## Summarize all active memories and return the new memory id
    ## TODO: This process could cascade several memory caches, but only depth 1 caches will be returned.
    ## Need to account for this by passing a list down the line, then let the conversation manager
    ## decide how to utilize the returned memories. will figure that out when I am less drunk.
    def compress_memory_cache(self, depth):
        self.debug_message('Compressing cache of depth (%s)...' % str(depth))
        ## Get a copy of the cached memories
        memories = self.__episodic_memory_caches[int(depth)].memories

        self.debug_message('Pushing compressed memory to cache of depth (%s)...' % str(int(depth)+1))
        ## Generate a higher depth memory
        episodic_memory, episodic_tokens = self.generate_episodic_memory(memories, int(depth)+1)

        ## Add the new memory to the cache
        self.cache_memory(episodic_memory, episodic_tokens, int(depth)+1)
        self.debug_message('Flushing cache of depth (%s)...' % str(depth))
        self.__episodic_memory_caches[int(depth)].flush_memory_cache()
        return episodic_memory, episodic_tokens

    ## Format eidetic memory with a speaker and timestamp for context
    def format_eidetic_memory(self, memory):
        message = '%s: %s - %s' % (memory['speaker'], memory['original_timestring'], memory['content'])
        return message

    ## Format episodic memory with a timestamp for context
    def format_episodic_memory(self, memory):
        message = '%s: %s' % (memory['original_timestring'], memory['original_summary'])
        return message

    ## Generate a summary, keyword, or other element for a eidetic messages or episodic summaries
    def generate_memory_element(self, element_type, content, depth, content_tokens=10):
        ## Choose which memory processing prompt to use
        if int(depth) == 0:
            prompt_name = 'eidetic_memory'
        elif int(depth) == 1:
            prompt_name = 'eidetic_to_episodic_memory'
        else:
            prompt_name = 'episodic_to_episodic_memory'

        ## Load the prompt from a .json file
        prompt_obj = self.load_json('%s/%s.json' % (self.__config['memory_management']['memory_prompts_dir'], prompt_name))
        
        ## Sum the token count for the content with the intended prompt token count
        response_tokens = int(prompt_obj[element_type]['response_tokens']) + int(content_tokens)
        if response_tokens > self.__max_tokens:
            response_tokens = self.__max_tokens
        temperature = float(prompt_obj[element_type]['temperature'])

        ## Generate memory element
        messages = list()
        prompt_content = prompt_obj[element_type]['system_message'] % content
        messages.append(self.compose_gpt_message(prompt_content,'user'))
        memory_element, total_tokens = self.gpt_completion(messages, temperature, response_tokens)
        return memory_element

    ## The role can be either system or user. If the role is system then you are either giving the model instructions/personas or example prompts.
    ## Then name field is used for example prompts which guide the model on how to respond.
    ## If the name field has data, the model will not consider them part of the conversation; the role will be system by default.
    def compose_gpt_message(self, content, role, name=''):
        content = content.encode(encoding='ASCII',errors='ignore').decode() ## Cheeky way to remove encoding errors
        if name == '':
            return {"role":role, "content": content}
        else:
            role = 'system'
            return {"role":role,"name":name,"content": content}

    ## Save memory locally, update local child memories, and save memory vector to pinecone
    def index_memory(self, memory):
        depth = int(memory['depth'])
        memory_id = memory['id']
        self.debug_message('indexing memory (%s)' % memory_id)

        self.save_memory_locally(memory)
        self.parent_local_child_memories(memory)
        vector = self.gpt3_embedding(str(memory['original_summary']))
        ## The metadata and namespace are redundant but I need the data split for later
        metadata = {'memory_type': 'episodic', 'depth': str(depth)}
        namespace = self.__config['memory_management']['memory_namespace_template'] % depth
        self.save_vector_to_pinecone(vector, memory_id, metadata, namespace)
        
        ## Cache the id to delete for testing.
        with open('mem_wipe.txt', 'a', encoding='utf-8') as mem_wipe:
            mem_wipe.write('%s\n' % str(memory_id))

    ## Stash the memory in the appropriate folder locally
    def save_memory_locally(self, memory):
        self.debug_message('saving memory locally...')
        depth = int(memory['depth'])
        memory_id = memory['id']
        ## Get the folder for this memory depth, make it if missing
        memory_stash_folder = (self.__config['memory_management']['stash_folder_template'] % int(depth))
        memory_dir = '%s/%s' % (self.__config['memory_management']['memory_stash_dir'], memory_stash_folder)

        if not os.path.exists(memory_dir):
            os.makedirs(memory_dir)
        memory_path = '%s/%s.json' % (memory_dir, memory_id)
        self.save_json(memory_path, memory)

    ## Update all lower-depth memories with higher-depth memory id
    def parent_local_child_memories(self, memory):
        self.debug_message('Parenting child memories...')
        depth = int(memory['depth'])
        memory_id = memory['id']
        if "lower_memory_ids" in memory:
            child_ids = list(memory['lower_memory_ids'])
            child_memory_dir = (self.__config['memory_management']['stash_folder_template'] % (int(depth)-1))
            if os.path.exists(child_memory_dir):
                for id in child_ids:
                    child_path = '%s/%s.json' % (child_memory_dir,id)
                    if os.path.isfile(child_path):
                        content = load_json(child_path)
                        content['episodic_parent_id'] = memory_id
                        self.save_json(child_path, content)

    ## Debug functions
    def breakpoint(self, message = '\n\nEnter to continue...'):
        input(message+'\n')
    
    def debug_message(self, message):
        if self.debug_messages_enabled:
            print(message)

######################################################
## Pinecone stuff... might move later...
#### Query pinecone with vector. If search_all = True then name_space will be ignored.
    def query_pinecone(self, vector, return_n, namespace = "", search_all = False):
        if search_all:
            results = self.__vector_db.query(vector=vector, top_k=return_n)
        else:
            results = self.__vector_db.query(vector=vector, top_k=return_n, namespace=namespace)

    #### Save vector to pinecone
    def save_vector_to_pinecone(self, vector, unique_id, metadata, namespace=""):
        self.debug_message('Saving vector to pinecone.')
        payload_content = {'id': unique_id, 'values': vector, 'metadata': metadata}
        payload = list()
        payload.append(payload_content)
        self.__vector_db.upsert(payload, namespace=namespace)

    def update_vector(self, vector, unique_id, metadata, namespace=""):
        ## TODO: figure out how to update existing vectors.
        ## This will be used when memories are retconned.
        return unique_id

######################################################
## Open AI stuff... might move later...

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
                response_str = response['choices'][0]['message']['content'].strip()
                return response_str, total_tokens
            except Exception as oops:
                retry += 1
                if retry >= max_retry:
                    return "GPT3.5 error: %s" % oops, -1
                print('Error communicating with OpenAI:', oops)
                sleep(2)

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
######################################################


    # ## Update memory with the episodic id.
    # ## Each memory will have a parent (episodic) id to allow for
    # ## recursive retcon updates from either depth direction.
    # def set_episodic_id(self, memory_id, episodic_id, memory_path, memory_type):
    #     ## TODO: Open memory file, if the eposodic id is empty, add it and overwrite the old content.
    #     ## Don't update if the episodic_id has data. Throw error or something.
    #     return ""



    # ## Save prompt as a json file in the prompt directory
    # def save_prompt(self, prefix, content, directory = config['raven']['prompt_dir'], filetime = time()):
    #     prefix.replace('.json','')
    #     filename = '%s%s.json' % (prompt_prefix, filetime)
    #     save_json('%s/%s' % (directory, filename, content))

    # ## Get the last n conversation memories
    # def get_recent_messages(self, message_history_count = 30, sort_decending = True):
    #     files = glob.glob('%s/*.json' % config['raven']['nexus_dir'])
    #     files.sort(key=os.path.getmtime, reverse=True)
    #     message_count = int(message_history_count)
    #     if len(files) < message_count:
    #         message_count = len(files)
    #     messages = list()
    #     for message in list(files):
    #         message = message.replace('\\','/')
    #         info = load_json(message)
    #         messages.append(info)
    #     return_messages = messages[0:message_count-1]
    #     if sort_decending:
    #         sorted(return_messages, key=lambda d: d['time'], reverse=False)
    #     return return_messages

    # ## Breakup memories into chunks so they meet token restrictions
    # def chunk_memories(self, memories, token_response_limit=int(config['raven']['summary_token_response'])):
    #     ## Each subsequent chunk will subtract the max_token_input from the token_response_limit
    #     max_token_input = int(config['open_ai']['max_token_input'])
    #     token_per_character = int(config['open_ai']['token_per_character'])

    #     blocks = list() # When the token limit overflows after chunking, a new block will be needed
    #     chunks = list() # All chunks which will fall within the token limit including responses go here
    #     chunk = list() # All memories which will fall within the token limit will go here

    #     memory_count = len(memories)-1
    #     block_count = 0
    #     block_token_response_limit = token_response_limit
    #     current_memory_index = 0

    #     max_iter = 100 # If this is hit there is something wrong
    #     iter = 0

    #     blocking_done = False
    #     while not blocking_done:
    #         chunking_done = False
    #         ## TODO: Come back to this decision of doubling the token response limit for each block
    #         ## Initial reasoning behind this, a summary of summaries for the chunks would be created
    #         ## then a summary of the next block, so we would need double the response space for each block.
    #         block_token_response_limit = (2 * block_count * token_response_limit)
    #         remaining_chunk_tokens = max_token_input - block_token_response_limit
    #         while not chunking_done:
    #             chunk_length = 0
    #             chunk.clear()
    #             memories_this_chunk = 0
    #             for i in range(current_memory_index, memory_count):
    #                 iter += 1
    #                 mem = memories[i]
    #                 message = format_summary_memory(mem)
    #                 ## TODO: Get actual token length from open ai
    #                 message_length = math.ceil(len(message)/token_per_character) # Estimate token length
    #                 chunk_length += message_length
    #                 if chunk_length > remaining_chunk_tokens and memories_this_chunk == 0:
    #                     current_memory_index = i
    #                     ## Chunking cannot continue, new block will be created
    #                     block_count += 1
    #                     blocks.append(chunks.copy())
    #                     chunking_done = True
    #                     breakpoint('\n\nChunking cannot continue until a new block is created...\n\n')
    #                     break
    #                 elif chunk_length > remaining_chunk_tokens:
    #                     current_memory_index = i
    #                     ## Chunking can continue, new chunk will be created
    #                     remaining_chunk_tokens -= token_response_limit
    #                     chunks.append(chunk.copy())
    #                     chunking_done = True
    #                     break
    #                 elif i == memory_count-1:
    #                     ## End of process, append remaining chunk and add chunks to block
    #                     chunk.append(mem)
    #                     chunks.append(chunk.copy())
    #                     blocks.append(chunks.copy())
    #                     chunking_done = True
    #                     blocking_done = True
    #                 else:
    #                     ## Chunking continues, decrement remaining tokens
    #                     chunk.append(mem)
    #                     memories_this_chunk += 1
    #                     continue

    #                 if iter >= max_iter:
    #                     chunking_done = True
    #                     blocking_done = True
    #                     breakpoint('\n\nSomething went wrong with the memory chunker. Max iterations reached.\n\n')
    #     return blocks

    # def summarize_memories(self, memories):  # summarize a block of memories into one payload
    #     memories = sorted(memories, key=lambda d: d['time'], reverse=False)  # sort them chronologically
    #     blocked_summary = '' ## TODO: Implement block summary strategy
    #     chunked_summary = ''
    #     blocks = chunk_memories(memories)
    #     block_count = len(blocks)
    #     for chunks in blocks:
    #         chunked_summary = '' # Combine all chunk summaries into one long prompt and use it for context
    #         for chunk in chunks:
    #             chunked_message = '' # Combine all memories into a long prompt and have it summarized
    #             for mem in chunk:
    #                 message = format_summary_memory(mem)
    #                 chunked_message += message.strip() + '\n\n'
    #             chunked_message = chunked_message.strip()

    #             chunked_prompt = open_file('prompt_notes.txt').replace('<<INPUT>>', chunked_summary + chunked_message)
    #             save_prompt('summary_chunk_prompt_',chunked_prompt)

    #             chunked_notes = gpt_completion(chunked_prompt)
    #             save_prompt('summary_chunk_notes_',chunked_notes)

    #             chunked_summary = chunked_notes.strip() + '\n'
    #     return chunked_summary.strip()

    #     def get_token_count(self, content):
    #         content = content.encode(encoding='ASCII',errors='ignore').decode()  # fix any UNICODE errors
    #         encoding = tiktoken.get_encoding(str(self.config['open_ai']['model']))
    #         tokens = encoding.encode(content)
    #         token_count = len(tokens)
    #         return token_count


# ## The following is from #6 - Counting tokens for chat API calls: https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
# def estimate_message_tokens (messages, model=str(['openai']['model'])):
#     try:
#         encoding = tiktoken.encoding_for_model(model)
#     except KeyError:
#         encoding = tiktoken.get_encoding("cl100k_base")
#     if model == "gpt-3.5-turbo-0301":  # note: future models may deviate from this
#         num_tokens = 0
#         for message in messages:
#             num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
#             for key, value in message.items():
#                 num_tokens += len(encoding.encode(value))
#                 if key == "name":  # if there's a name, the role is omitted
#                     num_tokens += -1  # role is always required and always 1 token
#         num_tokens += 2  # every reply is primed with <im_start>assistant
#         return num_tokens
#     else:
#         raise NotImplementedError(f"""num_tokens_from_messages() is not presently implemented for model {model}.
# See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens.""")