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
        self.__token_buffer = self.__config['memory_management']['token_buffer']
        self.__max_tokens = self.__config['open_ai']['max_token_input']
        self.__episodic_memory_caches = [] # index will represent memory depth, useful for dynamic memory expansion
        self.__max_episodic_depth = 2 # will restrict memory expansion. 0 is unlimited depth.

        openai.api_key = self.open_file(self.__config['open_ai']['api_key'])

        ## When initialized, attempt to load cached state, otherwise make a new state
        if not (self.load_state()):
            self.create_state()

    ## Houses memories of a particular depth. Each change will trigger will be followed with a state save
    class MemoryCache:
        ## Depth should not be changed after initialization.
        ## If cache is not empty then load it, otherwise start fresh.Cache should be JSON.
        def __init__(self, depth, token_buffer, max_tokens, cache=None):
            self.__depth = depth
            self.__token_buffer = token_buffer
            self.__max_tokens = max_tokens
            self.__memories = list()
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

        def import_cache(self, cache):
            ## Set all class attributes from the cache JSON
            ## If cache is invalid, throw error.
            return ""

        ## Return a JSON version of this object to save
        @property
        def cache(self):
            timestamp = time()
            unique_id = str(uuid4())
            timestring = datetime.datetime.fromtimestamp(timestamp).strftime("%A, %B %d, %Y at %I:%M:%S%p %Z")
            cache = {
                "depth":self.__depth,
                "token_buffer":self.__token_buffer,
                "max_tokens":self.__max_tokens,
                "memories":self.__memories,
                "timestamp": timestamp,
                "timestring": timestring,
                "id":unique_id
            }
            return cache

        ## Memories are in the raven eidetic or episodic JSON format
        ## Memory space check is NOT enforced so check it before adding a memory
        def add_memory(self, memory, number_of_tokens):
            self.__token_count += number_of_tokens
            self.__memories.append(memory)
            ## If no more memory space return True, this will trigger a memory compression
            return not (self.has_memory_space(0))

        ## Before adding a memory, check to see if there will be space with next memory.
        def has_memory_space(next_number_of_tokens):
            if self.__token_count + next_number_of_tokens <= (self.__max_tokens - self.__token_buffer):
                return True
            else:
                return False

        def flush_memory_cache():
            self.__memories.clear()
            self.__token_count = 0

    @property
    def episodic_memory_caches(self):
        print(self.__episodic_memory_caches)

    ## Load JSON object representing state. State is all memory caches not yet summarized, active tasks, and active context.
    def load_state(self):
        ## Get all json files ordered by name. The name should have a timestamp prefix.
        files = glob.glob('%s/*.json' % (self.__config['memory_management']['memory_state_dir']))
        files.sort(key=os.path.basename, reverse=True)

        # there were no state backups so make and implement a new one
        if len(list(files)) <= 0:
            print("no state backups found...")
            return False

        print("state backups loaded...")
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
        self.cache_state()

    def cache_state(self):
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

    def timestamp_to_datetime(self, unix_time):
        return datetime.datetime.fromtimestamp(unix_time).strftime("%A, %B %d, %Y at %I:%M:%S%p %Z")

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

    def stash_eidetic_memory(self, speaker, content, timestamp = time()):
        eidetic_memory, unique_id = generate_eidetic_memory(speaker, content, timestamp)
        ## Load to pinecone

    ## Assemble and index eidetic memories from a message by a speaker
    ## Eidetic memory is the base form of all episodic memory
    def generate_eidetic_memory(self, speaker, content, timestamp = time()):
        unique_id = str(uuid4())

        summary, keywords, tokens = self.generate_memory_elements(content, 0)
        # summary, tokens, keywords, content_vector = self.generate_eidetic_summary(content)
        timestring = self.timestamp_to_datetime(timestamp)
        eidetic_memory = {
            "id": unique_id,
            "episodic_parent_id":"",
            "speaker": speaker,
            "content": content,
            "original_summary": summary,
            "original_timestamp": timestamp,
            "original_timestring": timestring,
            "original_tokens":int(tokens),
            "is_retconned": False,
            "retcon_summary": "",
            "retcon_rank":0.0,
            "retcon_time": "",
            "retcon_tokens":0,
            "keywords": keywords,
            "depth":0
        }

        print(eidetic_memory)
        ## Stash the memory in Pinecone and save it locally
        # self.index_memory(eidetic_memory_path, eidetic_memory_type, eidetic_memory, unique_id, content_vector)
        return eidetic_memory, unique_id

    ## Assemble an eposodic memory from a collection of eidetic or lower-depth episodic memories.
    def generate_episodic_memory(self, memories, depth, timestamp = time()):
        unique_id = str(uuid4())
        timestring = self.timestamp_to_datetime(timestamp)

        ## Send collection of memories to be summarized, pass the unique id to set episodic_parent_id
        # summary, tokens, keywords, episodic_vector, memory_ids = self.generate_episodic_summary(memories, unique_id)
        content = ''
        memory_ids = ()
        for memory in memories:
            content += '%s\n' % memory['summary']
            memory_ids.append(str(memory['id']))
        content = "\n".join(result_list)
        summary, keywords, tokens = self.generate_memory_elements(content)

        episodic_memory = {
            "id": unique_id,
            "episodic_parent_id":"",
            "lower_memory_ids": memory_ids,
            "original_summary": summary,
            "original_timestamp": timestamp,
            "original_timestring": timestring,
            "original_tokens":int(tokens),
            "is_retconned": False,
            "retcon_summary": "",
            "retcon_rank":0.0,
            "retcon_time": "",
            "retcon_tokens":0,
            "keywords": keywords,
            "depth": int(depth)
        }
        ## TODO: update config file to have different memory paths and pinecone memory metadata
        print(episodic_memory)
        ## Stash the memory in Pinecone and save it locally
        # self.index_memory(episodic_memory_path, episodic_memory_type, episodic_memory, unique_id, episodic_vector)
        return episodic_memory, unique_id

    ## Format eidetic memory with a speaker and timestamp for context
    def format_eidetic_memory(self, memory):
        message = '%s: %s - %s' % (memory['speaker'], memory['original_timestring'], memory['content'])
        return message

    ## Format episodic memory with a timestamp for context
    def format_episodic_memory(self, memory):
        message = '%s: %s' % (memory['original_timestring'], memory['original_summary'])
        return message

    def generate_memory_elements(self, content, depth):
        if int(depth) == 0:
            prompt_name = 'eidetic_memory'
        elif int(depth) == 1:
            prompt_name = 'eidetic_to_episodic_memory'
        else:
            prompt_name = 'episodic_to_episodic_memory'

        # vector = gpt3_embedding(content)

        prompt_obj = self.load_json('%s/%s.json' % (self.__config['memory_management']['memory_prompts_dir'], prompt_name))

        messages = list()
        temperature = float(prompt_obj['summary']['temperature'])
        response_tokens = int(prompt_obj['summary']['response_tokens'])
        messages.append(self.compose_gpt_message(prompt_obj['summary']['system_message'],'system'))
        messages.append(self.compose_gpt_message(content,'user'))
        summary_obj, total_tokens = self.gpt_completion(messages, temperature, response_tokens)

        expected_return_type = str(prompt_obj['summary']['expected_return_type'])

        summary = self.process_gpt_memory_response('summary',summary_obj, expected_return_type)

        messages.clear()

        temperature = float(prompt_obj['keywords']['temperature'])
        response_tokens = int(prompt_obj['keywords']['response_tokens'])
        messages.append(self.compose_gpt_message(prompt_obj['keywords']['system_message'],'system'))
        messages.append(self.compose_gpt_message(content,'user'))
        keywords_obj, total_tokens = self.gpt_completion(messages, temperature, response_tokens)

        expected_return_type = str(prompt_obj['keywords']['expected_return_type'])

        keywords = self.process_gpt_memory_response('keywords',keywords_obj, expected_return_type)

        return summary, keywords, total_tokens

    ## Generate a summary for a single message.
    ## These summaries are typically shorter than episodic summaries.
    def generate_eidetic_summary(self, content):
        prompt_obj = self.load_json('%s/%s.json' % (self.__config['memory_management']['memory_prompts_dir'], 'eidetic_memory'))

        messages = list()
        temperature = float(prompt_obj['summary']['temperature'])
        response_tokens = int(prompt_obj['summary']['response_tokens'])
        messages.append(compose_gpt_message(prompt_obj['summary']['system_message'],'system'))
        messages.append(compose_gpt_message(content,'user'))
        summary_obj = self.gpt_completion(messages, temperature, response_tokens)
        expected_return_type = str(prompt_obj['summary']['expected_return_type'])

        summary = self.process_gpt_memory_response('summary',summary_obj, expected_return_type)

        messages.clear()
        
        temperature = float(prompt_obj['keywords']['temperature'])
        response_tokens = int(prompt_obj['keywords']['response_tokens'])
        messages.append(compose_gpt_message(prompt_obj['keywords']['system_message'],'system'))
        messages.append(compose_gpt_message(content,'user'))
        keywords_obj = self.gpt_completion(messages, temperature, response_tokens)

        expected_return_type = str(prompt_obj['keywords']['expected_return_type'])

        keywords = self.process_gpt_memory_response('keywords',keywords_obj, expected_return_type)
        
        # content_vector = gpt3_embedding(content)
        ## TODO: summarize message content in very short form
        # summary, tokens = self.generate_short_summary(content)
        # keywords = generate_keywords(content)
        # return summary, tokens, keywords, content_vector
        return ""

    ## Generate a summary for a collection of eidetic or episodic memories.
    ## These summaries are typically shorter than episodic summaries.
    ## Depth is the level of the memories being summarized
    def generate_episodic_summary(self, memories, depth, unique_id):
        memory_ids = list() ## TODO: Get each memory id
        content = '' ## TODO: Extract summaries
        ## TODO: for each memory in memory id list, update episodic id
        for memory in memories:
            memory_ids.append(memory['id'])
            set_episodic_id(memory, unique_id)
            if depth == 0:
                content += '%s\n\n' % self.format_eidetic_memory(memory)
            else:
                content += '%s\n\n' % self.format_episodic_memory(memory)

        token_estimate = get_token_count(content) ## Do something with this estimate
        episodic_vector = gpt3_embedding(content)
        ## TODO: summarize message content
        summary, tokens = generate_summary(content)
        keywords = generate_keywords(content)

        return summary, tokens, keywords, episodic_vector, memory_ids

    def process_gpt_memory_response(self, response_name, response_obj, expected_return_type):
        result = ''
        if expected_return_type == "string":
            result = response_obj['response']
        elif expected_return_type == "list":
            result_list = list(response_obj['response'])
            result = "\n".join(result_list)

        result = result.strip()
        result = re.sub('[\r\n]+', '\n', result)
        result = re.sub('[\t ]+', ' ', result)
        return result

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

######################################################
## Open AI stuff... might move later...

    def gpt3_embedding(self, content):
        engine = config['open_ai']['input_engine']
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

                self.print_response_stats(response)
                response_id = str(response['id'])
                prompt_tokens = int(response['usage']['prompt_tokens'])
                completion_tokens = int(response['usage']['completion_tokens'])
                total_tokens = int(response['usage']['total_tokens'])
                print(response['choices'][0]['message']['content'].strip())
                response_obj = json.loads(response['choices'][0]['message']['content'].strip())
                return response_obj, total_tokens
            except Exception as oops:
                retry += 1
                if retry >= max_retry:
                    return "GPT3.5 error: %s" % oops
                print('Error communicating with OpenAI:', oops)
                sleep(2)

    def print_response_stats(self, response):
        response_id = ('\nResponse %s' % str(response['id']))
        prompt_tokens = ('\nPrompt Tokens: %s' % (str(response['usage']['prompt_tokens'])))
        completion_tokens = ('\nCompletion Tokens: %s' % str(response['usage']['completion_tokens']))
        total_tokens = ('\nTotal Tokens: %s\n' % (str(response['usage']['total_tokens'])))
        print(response_id + prompt_tokens + completion_tokens + total_tokens)
######################################################


    # ## Update memory with the episodic id.
    # ## Each memory will have a parent (episodic) id to allow for
    # ## recursive retcon updates from either depth direction.
    # def set_episodic_id(self, memory_id, episodic_id, memory_path, memory_type):
    #     ## TODO: Open memory file, if the eposodic id is empty, add it and overwrite the old content.
    #     ## Don't update if the episodic_id has data. Throw error or something.
    #     return ""

    # def index_memory(self, memory_path, memory_type, memory, unique_id, vector):
    #     save_json('%s/%s.json' % (memory_path,unique_id), memory)
    #     ## TODO:
    #     metadata = {'memory_type': memory_type}
    #     save_vector(vector, unique_id, metadata, memory_type)
    #     return unique_id

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