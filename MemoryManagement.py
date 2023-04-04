import configparser
import os
import json
import glob
import datetime
from uuid import uuid4
import re
from UtilityFunctions import *

##### NOTE: Token counts should leave enough room for a variety of prompt instructions, since their use may vary. I am thinking of leaving a buffer of 1000 to ensure there is enough room, but I will generalize it so adjustments can be made easily

##### NOTE: Memories may need to be chunked if the token estimation is significantly off. This can be done on a prompt-by-prompt basis

## When a new message is added the memory manager will assess when memory compression should occur.
class MemoryManager:
    def __init__(self):
        self.__config = get_config()
        self.__cache_token_limit = int(self.__config['memory_management']['cache_token_limit'])
        self.__max_tokens = int(self.__config['open_ai']['max_token_input'])
        self.__episodic_memory_caches = [] # index will represent memory depth, useful for dynamic memory expansion
        self.__max_episodic_depth = 2 # will restrict memory expansion. 0 is unlimited depth.
        self.__pinecone_indexing_enabled = self.__config.getboolean('pinecone', 'pinecone_indexing_enabled')
        self.debug_messages_enabled = True

        ## When initialized, attempt to load cached state, otherwise make a new state
        if not (self.load_state()):
            self.create_state()

    ## Houses memories of a particular depth. Each change will trigger will be followed with a state save
    class MemoryCache:
        ## Depth should not be changed after initialization.
        ## If cache is not empty then load it, otherwise start fresh.
        def __init__(self, depth, cache_token_limit, max_tokens, cache=None):
            self.debug_messages_enabled = True
            self.__id = str(uuid4())
            self.__depth = int(depth)
            self.__config = configparser.ConfigParser()
            self.__config.read('config.ini')
            self.__cache_token_limit = cache_token_limit
            self.__max_tokens = int(max_tokens)
            self.__memories = list()
            self.__current_memory_ids = list()
            self.__previous_memory_ids = list()
            self.__token_count = 0
            self.__memory_stash_folder = (self.__config['memory_management']['stash_folder_template'] % self.__depth)
            self.__memory_dir = '%s/%s' % (self.__config['memory_management']['memory_stash_dir'], self.__memory_stash_folder)
            if cache is not None:
                self.load_cache(cache)

        @property
        def depth(self):
            return self.__depth

        @property
        def cache_token_limit(self):
            return self.self.__cache_token_limit

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
        def load_cache(self, cache):
            self.__id = str(cache['id'])
            self.__depth = int(cache['depth'])
            self.__cache_token_limit = int(cache['cache_token_limit'])
            self.__max_tokens = int(cache['max_tokens'])
            self.__token_count = int(cache['token_count'])
            self.__memories = list(cache['memories'])
            self.__current_memory_ids = list(cache['current_memory_ids'])
            self.__previous_memory_ids = list(cache['previous_memory_ids'])
            ## TODO: If cache is invalid, throw error.

        ## Return a JSON version of this object to save
        @property
        def cache(self):
            timestamp = time()
            timestring = timestamp_to_datetime(timestamp)
            cache = {
                "id":self.__id,
                "depth":self.__depth,
                "cache_token_limit":self.__cache_token_limit,
                "max_tokens":self.__max_tokens,
                "token_count":self.__token_count,
                "memories":self.__memories,
                "current_memory_ids": self.__current_memory_ids,
                "previous_memory_ids": self.__previous_memory_ids,
                "timestamp": timestamp,
                "timestring": timestring
            }
            return cache

        ## Memories are in the raven eidetic or episodic JSON format
        ## Memory space check is NOT enforced so check it before adding a memory
        def add_memory(self, memory, number_of_tokens):
            self.__token_count += int(number_of_tokens)
            memory_count = len(self.__memories)
            ## Link memory siblings
            if memory_count > 0:
                memory_id = memory['id']
                past_sibling_id = self.__memories[memory_count-1]['id']
                self.update_past_sibling(memory_id, past_sibling_id)
                memory['past_sibling'] = past_sibling_id
            self.__memories.append(memory)
            self.__current_memory_ids.append(memory['id'])
            self.parent_local_child_memories(memory)
            self.save_memory_locally(memory)

        ## Before adding a memory, check to see if there will be space with next memory.
        def has_memory_space(self, next_number_of_tokens):
            if self.__token_count + int(next_number_of_tokens) <= self.__cache_token_limit:
                return True
            else:
                return False

        def transfer_memory_ids(self):
            ## Cache the current caches memory ids for conversation loading purposes
            self.__previous_memory_ids = self.__current_memory_ids.copy()
            self.__current_memory_ids.clear()

        def flush_memory_cache(self):
            ## Clear memory cache and reset token count
            self.__memories.clear()
            self.__token_count = 0
            self.__id = str(uuid4())

        ## Add memory id to past sibling's 'next_sibling' id
        def update_past_sibling(self, memory_id, past_sibling_id):
            sibling_path = '%s/%s.json' % (self.__memory_dir, past_sibling_id)
            if os.path.isfile(sibling_path):
                content = load_json(sibling_path)
                content['next_sibling'] = memory_id
                save_json(sibling_path, content)

        ## Stash the memory in the appropriate folder locally
        def save_memory_locally(self, memory):
            debug_message('saving memory locally...', self.debug_messages_enabled)
            memory_id = memory['id']
            memory_path = '%s/%s.json' % (self.__memory_dir, memory_id)
            if not os.path.exists(self.__memory_dir):
                os.mkdir(self.__memory_dir)
            save_json(memory_path, memory)

        ## Update all lower-depth memories with higher-depth memory id
        def parent_local_child_memories(self, memory):
            debug_message('Parenting child memories...', self.debug_messages_enabled)
            memory_id = memory['id']
            if self.__depth > 0 in memory:
                child_ids = list(memory['lower_memory_ids'])
                for id in child_ids:
                    child_path = '%s/%s.json' % (self.__memory_dir, id)
                    if os.path.isfile(child_path):
                        content = load_json(child_path)
                        content['episodic_parent_id'] = memory_id
                        save_json(child_path, content)
                    

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
            debug_message("no state backups found...", self.debug_messages_enabled)
            return False

        debug_message("state backups loaded...", self.debug_messages_enabled)
        state_path = list(files)[0].replace('\\','/')
        state = load_json(state_path)

        for cache in state['memory_caches']:
            depth = cache['depth']
            self.__episodic_memory_caches.append(self.MemoryCache(depth, self.__cache_token_limit, self.__max_tokens, cache))

        ## TODO:
        ## load state, if that fails, try backup files
        ## If memory caches exceed max episodic depth then throw error or set new max
        ## initialize missing memory caches
        return True

    def create_state(self):
        for i in range(self.__max_episodic_depth+1):
            ## Initialize all episodic memory caches
            self.__episodic_memory_caches.append(self.MemoryCache(i, self.__cache_token_limit, self.__max_tokens))
        self.save_state()

    def save_state(self):
        ## TODO: Restrict backups to max_backup_states
        timestamp = time()
        timestring = timestamp_to_datetime(timestamp)
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
        save_json(filepath, state)
        debug_message('State saved...', self.debug_messages_enabled)

    ## Assemble and index eidetic memories from a message by a speaker
    ## Eidetic memory is the base form of all episodic memory
    def generate_eidetic_memory(self, speaker, content):
        timestamp = time()
        debug_message('Generating eidetic memory...', self.debug_messages_enabled)
        unique_id = str(uuid4())
        timestring = timestamp_to_datetime(timestamp)
        depth = 0

        content_tokens = get_token_estimate(content)

        summary_result = self.summarize_content('%s: %s' % (speaker, content), depth, speaker, content_tokens = content_tokens)
        summary = self.cleanup_response(summary_result)
        summary_tokens = get_token_estimate(summary)

        ## Build episodic memory object
        eidetic_memory = {
            "id": unique_id,
            "episodic_parent_id":"",
            "speaker": speaker,
            "content": content,
            "content_tokens":content_tokens,
            "summary": summary,
            "summary_tokens":int(summary_tokens),
            "next_sibling":None,
            "past_sibling":None,
            "theme_links":[],
            "timestamp": timestamp,
            "timestring": timestring,
            "depth":int(depth)
        }
        return eidetic_memory, summary_tokens

    ## Assemble an eposodic memory from a collection of eidetic or lower-depth episodic memories
    def generate_episodic_memory(self, memories, depth):
        timestamp = time()
        debug_message('Generating episodic memory of cache depth (%s)...' % str(depth), self.debug_messages_enabled)
        unique_id = str(uuid4())
        timestring = timestamp_to_datetime(timestamp)

        ## Append all summaries together, get a token total, and get a list of memory ids
        content = ''
        content_tokens = 0
        memory_ids = []
        for memory in memories:
            content += '%s\n' % (memory['summary'])
            content_tokens += int(memory['summary_tokens'])
            memory_ids.append(str(memory['id']))

        summary_result = self.summarize_content(content, depth, content_tokens = content_tokens)
        summary = self.cleanup_response(summary_result)
        summary_tokens = get_token_estimate(summary)
        
        ## Extract the themes of this memory
        themes = self.extract_themes(content)
        theme_namespace = self.__config['memory_management']['theme_namespace_template']
        theme_folderpath = self.__config['memory_management']['theme_stash_dir']

        ## Create new theme links for all lower memories and add links to theme objects
        theme_links = list()
        for theme_id in themes:
            theme_filepath = '%s/%s.json' % (theme_folderpath, theme_id)
            theme_obj = load_json(theme_filepath)
            theme_memory_ids = map(lambda l: l['memory_id'], theme_obj['links'])
            for memory_id in memory_ids:
                if memory_id not in theme_memory_ids:
                    memory_link = self.generate_theme_link(theme_id, memory_id, (int(depth)-1))
                    theme_links.append(memory_link)
                    theme_obj['links'].append(memory_link)
            save_json(theme_filepath, theme_obj)

        ## Add theme links to lower memories
        if len(theme_links) > 0:
            memory_stash_folder = (self.__config['memory_management']['stash_folder_template'] % (int(depth)-1))
            memory_folderpath = '%s/%s' % (self.__config['memory_management']['memory_stash_dir'], memory_stash_folder)
            for memory_id in memory_ids:
                memory_links = list(filter(lambda l: l['memory_id']==memory_id, theme_links))
                if len(memory_links) > 0:
                    memory_filepath = '%s/%s.json' % (memory_folderpath, memory_id)
                    memory_obj = load_json(memory_filepath)
                    memory_obj['theme_links'] += memory_links
                    save_json(memory_filepath, memory_obj)

        ## Build episodic memory object
        episodic_memory = {
            "id": unique_id,
            "episodic_parent_id":"",
            "lower_memory_ids": memory_ids,
            "summary": summary,
            "summary_tokens":int(summary_tokens),
            "anticipation": "",
            "anticipation_tokens": "",
            "next_sibling":None,
            "past_sibling":None,
            "theme_links":[],
            "timestamp": timestamp,
            "timestring": timestring,
            "depth": int(depth)
        }
        return episodic_memory, summary_tokens

    ## Strip away unwanted characters from gpt response
    def cleanup_response(self, response):
        response = response.strip()
        response = re.sub('[\r\n]+', '\n', response)
        response = re.sub('[\t ]+', ' ', response)
        return response

    ## Caching memories may cascade and compress higher depth caches
    ## Check to see if cache as room, if so then add memory, otherwise compress that cache before adding
    def cache_memory(self, memory, tokens, depth):
        debug_message('adding memory to cache (%s)' % str(depth), self.debug_messages_enabled)
        if self.__episodic_memory_caches[int(depth)].has_memory_space(tokens):
            debug_message('There is enough space in the cache...', self.debug_messages_enabled)
            self.__episodic_memory_caches[int(depth)].add_memory(memory, tokens)
        else:
            debug_message('There is not enough space in the cache (%s), compressing...' % str(depth), self.debug_messages_enabled)    
            self.compress_memory_cache(depth)
            debug_message('Flushing cache of depth (%s)...' % str(depth), self.debug_messages_enabled)
            self.__episodic_memory_caches[int(depth)].flush_memory_cache()
            # Add the new message to the recently flushed memory cache
            self.__episodic_memory_caches[int(depth)].add_memory(memory, tokens)
        debug_message('Saving state...', self.debug_messages_enabled)
        self.index_memory(memory)
        self.save_state()

    def create_new_memory(self, speaker, content):
        episodic_memory = None
        episodic_tokens = 0
        depth = 0
        memory, tokens = self.generate_eidetic_memory(speaker, content)
        debug_message('adding memory to cache (%s)' % str(depth), self.debug_messages_enabled)
        if self.__episodic_memory_caches[depth].has_memory_space(tokens):
            debug_message('There is enough space in the cache...', self.debug_messages_enabled)
            self.__episodic_memory_caches[depth].add_memory(memory, tokens)
        else:
            debug_message('There is not enough space in the cache, compressing...', self.debug_messages_enabled)    
            episodic_memory, episodic_tokens = self.compress_memory_cache(depth)
            self.__episodic_memory_caches[depth].flush_memory_cache()
            self.__episodic_memory_caches[depth].add_memory(memory, tokens)
        
        self.index_memory(memory)
        if speaker == 'RAVEN':
            debug_message('Saving state...', self.debug_messages_enabled)
            self.save_state()
        return memory, tokens, episodic_memory, episodic_tokens

    ## Summarize all active memories and return the new memory id
    ## TODO: This process could cascade several memory caches, but only depth 1 caches will be returned.
    ## Need to account for this by passing a list down the line, then let the conversation manager
    ## decide how to utilize the returned memories. will figure that out when I am less drunk.
    def compress_memory_cache(self, depth):
        debug_message('Compressing cache of depth (%s)...' % str(depth), self.debug_messages_enabled)
        ## Get a copy of the cached memories
        memories = self.__episodic_memory_caches[int(depth)].memories
        ## Ensure the memory cache current memory ids transfer to the previous memory ids list
        self.__episodic_memory_caches[int(depth)].transfer_memory_ids()


        debug_message('Pushing compressed memory to cache of depth (%s)...' % str(int(depth)+1), self.debug_messages_enabled)
        ## Generate a higher depth memory
        episodic_memory, episodic_tokens = self.generate_episodic_memory(memories, int(depth)+1)

        ## Add the new memory to the cache
        self.cache_memory(episodic_memory, episodic_tokens, int(depth)+1)
        debug_message('Flushing cache of depth (%s)...' % str(depth), self.debug_messages_enabled)
        self.__episodic_memory_caches[int(depth)].flush_memory_cache()
        return episodic_memory, episodic_tokens


    def summarize_content(self, content, depth, speaker = '', content_tokens = 0):
        ## Choose which memory processing prompt to use
        if int(depth) == 0:
            prompt_name = 'eidetic_memory'
        elif int(depth) == 1:
            prompt_name = 'eidetic_to_episodic_memory'
        else:
            prompt_name = 'episodic_to_episodic_memory'

        ## Load the prompt from a .json file
        prompt_obj = load_json('%s/%s.json' % (self.__config['memory_management']['memory_prompts_dir'], prompt_name))
        
        ## Sum the token count for the content with the intended prompt token count
        response_tokens = int(prompt_obj['summary']['response_tokens']) + int(content_tokens)
        if response_tokens > self.__max_tokens:
            response_tokens = self.__max_tokens
        temperature = float(prompt_obj['summary']['temperature'])

        ## Generate memory element
        messages = list()
        if int(depth) == 0:
            prompt_content = prompt_obj['summary']['system_message'] % (speaker, content)
        else:
            prompt_content = prompt_obj['summary']['system_message'] % content
        messages.append(self.compose_gpt_message(prompt_content,'user'))
        memory_element, total_tokens = gpt_completion(messages, temperature, response_tokens)
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
        if not self.__pinecone_indexing_enabled:
            return
        depth = int(memory['depth'])
        memory_id = memory['id']
        debug_message('indexing memory (%s)' % memory_id, self.debug_messages_enabled)

        vector = gpt3_embedding(str(memory['summary']))
        ## The metadata and namespace are redundant but I need the data split for later
        metadata = {'memory_type': 'episodic', 'depth': str(depth)}
        namespace = self.__config['memory_management']['memory_namespace_template'] % depth
        save_vector_to_pinecone(vector, memory_id, metadata, namespace)

#####################################################
## THEMES ##
## Theme objects are just an intuition I have about how these memories should be organized
## Themes are extracted from episodic memory contents before they are summarized
## Themes will be embedded
##      If the embedding doesn't meet a query score falls under the threshold a new Theme object and pinecone vector will be created
##      If the query score is over the query threshold then the Themes will merge with the pre-existing Theme object
## Lower memories (depth of current memory - 1) are linked to the Theme via an ID on both entities
## Later on, random memories will be selected from the Theme objects
## New Theme elements will be extracted from the random sets and compared to their current Theme object.
## The comparison will have one of three results:
##      The new Themes match the Theme object the set was pulled from.
##          The random memories' connection strength to the original Theme will be strengthened and all other connections will be weakened
##      The new Themes match a different pre-existing Theme object.
##          The random memories'connection strength to the original Theme and the new Theme will be strengthened
##      The new Themes don't match any pre-existing Theme objects.
##          The random memories'connection strength to the original Theme will be weakened
## Each random memories selected will be disqualified for random selection for a period of time
## This will replicate a reienforcement mechanism. Random selections will help extract memories with complex or multiple themes.

    ## Get a list of themes from a summary of a memory and prepare the theme embedding objects
    def extract_themes(self, content):
        print('Extracting themes...')
        extracted_themes = list()
        ## Prompt for themes
        themes_result = self.extract_content_themes(content)
        ## Cleanup themes
        themes, has_error = self.cleanup_theme_response(themes_result)
        if has_error:
            breakpoint('there was a thematic extraction error')
            return extracted_themes
        theme_namespace = self.__config['memory_management']['theme_namespace_template']
        theme_folderpath = self.__config['memory_management']['theme_stash_dir']
        theme_match_threshold = float(self.__config['memory_management']['theme_match_threshold'])
        print('themes extracted...')
        for theme in themes:
            theme = str(theme)
            vector = gpt3_embedding(theme)
            theme_matches = query_pinecone(vector, 1, namespace=theme_namespace)
            print('these are the theme matches:')
            print(theme_matches)
            if theme_matches is not None:
                if len(theme_matches['matches']) > 0:
                    ## There is a match so check the threshold
                    match_score = float(theme_matches['matches'][0]['score'])
                    print('the score for theme match {%s} was %s' % (theme, str(match_score)))
                    if match_score >= theme_match_threshold:
                        ## Theme score is above the threshold, update an existing theme
                        existing_theme_id = theme_matches['matches'][0]['id']
                        ## Load, update, and save theme
                        theme_filepath = '%s/%s.json' % (theme_folderpath, existing_theme_id)
                        existing_theme = load_json(theme_filepath)
                        if theme not in existing_theme['themes']:
                            existing_theme['themes'].append(theme)
                            theme_string = ','.join(existing_theme['themes'])
                            self.update_theme_vector(existing_theme_id, theme_string, theme_namespace)
                            existing_theme['theme_count'] = len(existing_theme['themes'])
                            save_json(theme_filepath, existing_theme)
                        ## Add to extracted theme list
                        if existing_theme_id not in extracted_themes:
                            extracted_themes.append(existing_theme_id)
                        continue
            print('Making new themes')
            ## The theme score falls under the threshold or there was no match, make a new theme
            unique_id = str(uuid4())

            
            ## Add the theme to pinecone before making it so that similar themes in the current list can be merged
            payload = [{'id': unique_id, 'values': vector}]
            save_payload_to_pinecone(payload, theme_namespace)

            ## Make theme and save it locally
            new_theme = self.generate_theme(unique_id)
            new_theme['themes'].append(theme)
            new_theme['theme_count'] = 1
            theme_filepath = '%s/%s.json' % (theme_folderpath, unique_id)
            save_json(theme_filepath, new_theme)

            ## Add to extracted theme list
            extracted_themes.append(unique_id)

        return extracted_themes

    ## Get themes
    def extract_content_themes(self, content):
        prompt_name = 'memory_theme'
        ## Load the prompt from a .json file
        prompt_obj = load_json('%s/%s.json' % (self.__config['memory_management']['memory_prompts_dir'], prompt_name))
        
        temperature = prompt_obj['summary']['temperature']
        response_tokens = prompt_obj['summary']['response_tokens']

        ## Generate memory element
        prompt_content = prompt_obj['summary']['system_message'] % (content)
        prompt = [self.compose_gpt_message(prompt_content,'user')]
        response, tokens = gpt_completion(prompt, temperature, response_tokens)
        return response

    ## Ensure the theme extraction has been cleaned up
    def cleanup_theme_response(self, themes):
        has_error = True
        if type(themes) == list:
                has_error = False
                return themes, has_error
        themes_obj = {}
        try:
            themes_obj = json.loads(themes)
        except Exception as err:
            print('ERROR: unable to parse the json object when extracting themes')
            print('Value from GPT:\n\n%s' % themes)
            breakpoint('\n\npausing...')
            return [], has_error

        dict_keys = list(themes_obj.keys())
        if len(dict_keys) > 1:
            print('ERROR: unknown response for theme extraction')
            print('Value from GPT:\n\n%s' % themes)
            print(themes_obj)
            breakpoint('\n\npausing...')
            return [], has_error
        else:
            key = dict_keys[0]
            if type(themes_obj[key]) == list:
                has_error = False
                return themes_obj[key], has_error
            else:
                print('ERROR: unknown response for theme extraction')
                print('Value from GPT:\n\n%s' % themes)
                print(themes_obj)
                breakpoint('\n\npausing...')
                return [], has_error
        return [], has_error

    ## Load the theme, regenerate the embedding string, get embedding, and update the pinecone record
    def update_theme_vector(self, theme_id, theme_string, namespace):
        if not self.__pinecone_indexing_enabled:
            return
        vector = gpt3_embedding(theme_string)
        update_response = update_pinecone_vector(theme_id, vector, namespace)

    def generate_theme(self, theme_id):
        timestamp = time()
        timestring = timestamp_to_datetime(timestamp)
        theme_obj = {
            'id':theme_id,
            'themes':[],
            'theme_count':0,
            'links':[],
            'timestamp':timestamp,
            'timestring':timestring,
            'update_embedding':False
        }
        return theme_obj

    def generate_theme_link(self, theme_id, memory_id, memory_depth):
        link = {
            'theme_id':theme_id,
            'memory_id':memory_id,
            'depth':int(memory_depth),
            'weight':0.0,
            'cooldown_count':0,
            'repeat_theme_count':0
        }
        return link