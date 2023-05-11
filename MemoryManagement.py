import configparser
import os
import json
import glob
import datetime
from uuid import uuid4
import re
import sqlite3
from ThemeManagement import ThemeManager
from PromptManagement import PromptManager
from UtilityFunctions import *

##### NOTE: Token counts should leave enough room for a variety of prompt instructions, since their use may vary. I am thinking of leaving a buffer of 1000 to ensure there is enough room, but I will generalize it so adjustments can be made easily

##### NOTE: Memories may need to be chunked if the token estimation is significantly off. This can be done on a prompt-by-prompt basis

## When a new message is added the memory manager will assess when memory compression should occur.
class MemoryManager:
    def __init__(self):
        self.__config = get_config()
        self.__prompts = PromptManager()
        self.__themes = ThemeManager()
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
    class _MemoryCache:
        ## Depth should not be changed after initialization.
        ## If cache is not empty then load it, otherwise start fresh.
        def __init__(self, depth, cache_token_limit, max_tokens, cache = None):
            self.debug_messages_enabled = True
            self.__config = get_config()
            if cache is not None:
                self.__cache = cache
                self.__depth = self.__cache['depth']
                self.__cache_token_limit = self.__cache['cache_token_limit']
                self.__max_tokens = self.__cache['max_tokens']
            else:
                self.__depth = int(depth)
                self.__cache_token_limit = int(cache_token_limit)
                self.__max_tokens = int(max_tokens)
                self.__cache = self.get_new_cache()

        @property
        def id(self):
            return self.__cache['id']

        @property
        def depth(self):
            return self.__depth

        @property
        def cache_token_limit(self):
            return self.__cache_token_limit

        ## Returns a copy of memories; useful for compression and will not bork stuff when flushed
        @property
        def memories(self):
            memories = sql_query_by_ids('Memories','id',self.__cache['memory_ids'])
            return memories

        @property
        def memory_count(self):
            return len(self.__cache['memory_ids'])

        @property
        def token_count(self):
            return self.__cache['token_count']

        ## Return a list of all ids in memories tracked by the memory cache
        @property
        def memory_ids(self):
            return self.__cache['memory_ids']
        
        # @property
        # def previous_memory_ids(self):
        #     return self.__previous_memory_ids

        def get_new_cache(self, new_cache_id = str(uuid4())):
            new_cache = create_row_object(
                table_name='Memory_Caches',
                id=new_cache_id,
                depth=self.__depth,
                cache_token_limit=self.__cache_token_limit,
                token_count=0,
                max_tokens=self.__max_tokens,
                memory_ids=list(),
                created_on=time(),
                modified_on=time()
            )
            return new_cache

        ## Memories are in the raven eidetic or episodic JSON format
        ## Memory space check is NOT enforced so check it before adding a memory
        def add_memory(self, memory, number_of_tokens):
            ## Increment the token count
            self.__cache['token_count'] += int(number_of_tokens)
            ## Link memory siblings
            memory_count = len(self.__cache['memory_ids'])
            memory_id = memory['id']
            past_sibling_id = None
            if memory_count > 0:
                past_sibling_id = self.__cache['last_memory_id']
            else:
                self.__cache['first_memory_id'] = memory['id']
                if self.__cache['past_cache_id'] is not None:
                    ## Update cache's first memory id
                    past_cache_results = sql_query_by_ids('Memory_Caches','id',self.__cache['past_cache_id'])
                    if len(past_cache_results) <= 0:
                        debug_message(f"Unable to load past memory cache {self.__cache['past_cache_id']}")
                    else:
                        past_sibling_id = past_cache_results[0]['last_memory_id']
            self.update_past_sibling(memory_id, past_sibling_id)
            ## Link memory children
            self.parent_local_child_memories(memory)
            ## Update memory
            memory['past_sibling_id'] = past_sibling_id
            self.save_memory(memory)
            ## Update cache
            self.__cache['last_memory_id'] = memory['id']
            self.__cache['memory_ids'].append(memory['id'])
            self.save_memory_cache()

        ## Before adding a memory, check to see if there will be space with next memory.
        def has_memory_space(self, next_number_of_tokens):
            if self.__cache['token_count'] + int(next_number_of_tokens) <= self.__cache['cache_token_limit']:
                return True
            else:
                return False
            
        def flush_memory_cache(self):
            ## Get copy of current cache id and make a new cache id
            old_cache_id = self.__cache['id']
            new_cache_id = str(uuid4())
            ## Update current cache with next cache id then save
            self.__cache['next_cache_id'] = new_cache_id
            self.save_memory_cache()
            ## Clear memory cache, set past cache id, then save
            self.__cache = self.get_new_cache(new_cache_id)
            self.__cache['past_cache_id'] = old_cache_id
            self.save_memory_cache()

        def save_memory_cache(self):
            sql_insert_row('Memory_Caches','id',self.__cache)

        ## Add memory id to past sibling's 'next_sibling' id
        def update_past_sibling(self, memory_id, past_sibling_id):
            sql_update_row('Memories','id',{'id':past_sibling_id,'next_sibling_id':memory_id})
            
        ## Save the memory in the sql database
        def save_memory(self, memory):
            sql_insert_row('Memories', 'id', memory)

        ## Update all lower-depth memories with higher-depth memory id
        def parent_local_child_memories(self, memory):
            debug_message('Parenting child memories...', self.debug_messages_enabled)
            memory_id = memory['id']
            if self.__depth > 0:
                if memory['episodic_children_ids'] is None:
                    memory['episodic_children_ids'] = []
                child_ids = list(memory['episodic_children_ids'])
                children_memories = sql_query_by_ids('Memories', 'id', child_ids)
                for child_memory in children_memories:
                    child_memory['episodic_parent_id'] = memory_id
                    sql_update_row('Memories','id',child_memory)

    ## Return the number of memories of a given cache
    def get_cache_memory_count(self,depth):
        if int(depth) > len(self.__episodic_memory_caches):
            return -1
        return self.__episodic_memory_caches[int(depth)].memory_count

    ## Return the list of memories currently in cache
    def get_memories_from_cache(self, depth):
        return self.__episodic_memory_caches[int(depth)].memories

    # ## Return id list of memories in previous cache before it was flushed.
    # def get_previous_memory_ids_from_cache(self, depth):
    #     return self.__episodic_memory_caches[int(depth)].previous_memory_ids.copy()

    @property
    def cache_count(self):
        return len(self.__episodic_memory_caches)

    ## Load JSON object representing state. State is all memory caches not yet summarized, active tasks, and active context.
    def load_state(self):
        ## Get most recent memory state from database
        query = f"select * from Memory_States order by created_on desc limit 1"
        states = sql_custom_query(query)
        # there were no state backups so return false to make a new one
        if len(states) <= 0:
            debug_message("no state backups found...", self.debug_messages_enabled)
            return False
        ## Append a new cache
        debug_message("state backups loaded...", self.debug_messages_enabled)
        memory_caches = sql_query_by_ids('Memory_Caches','id', states[0]['memory_cache_ids'])
        for cache in memory_caches:
            depth = cache['depth']
            self.__episodic_memory_caches.append(self._MemoryCache(depth, self.__cache_token_limit, self.__max_tokens, cache))
        return True

    def create_state(self):
        for i in range(self.__max_episodic_depth+1):
            ## Initialize all episodic memory caches
            self.__episodic_memory_caches.append(self._MemoryCache(i, self.__cache_token_limit, self.__max_tokens))
        self.save_state()

    def save_state(self):
        unique_id = str(uuid4())

        memory_cache_ids = list()
        for cache in self.__episodic_memory_caches:
            memory_cache_ids.append(cache.id)
            cache.save_memory_cache()
        
        ## Insert Memory State
        state = create_row_object(
            table_name='Memory_States',
            id=unique_id,
            memory_cache_ids=memory_cache_ids,
            created_on=time(),
            modified_on=time()
        )
        success = sql_insert_row('Memory_States','id',state)
        if success:
            debug_message('State saved.', self.debug_messages_enabled)
        else:
            debug_message('Unable to save state!', self.debug_messages_enabled)

    ## Assemble and index eidetic memories from a message by a speaker
    ## Eidetic memory is the base form of all episodic memory
    def generate_eidetic_memory(self, speaker, content):
        debug_message('Generating eidetic memory...', self.debug_messages_enabled)
        
        unique_id = str(uuid4())
        depth = 0
        content_tokens = get_token_estimate(content)
        summary = ('%s: %s') % (speaker, content)
        summary_tokens = get_token_estimate(summary)

        ## Build episodic memory object
        eidetic_memory = create_row_object(
            table_name='Memories',
            id=unique_id,
            depth=int(depth),
            speaker=speaker,
            content=content,
            content_tokens=content_tokens,
            summary=summary,
            summary_tokens=int(summary_tokens),
            total_themes=0,
            created_on=time(),
            modified_on=time()
        )
        return eidetic_memory, summary_tokens

    ## Assemble an eposodic memory from a collection of eidetic or lower-depth episodic memories
    def generate_episodic_memory(self, memories, depth):
        timestamp = time()
        debug_message('Generating episodic memory of cache depth (%s)...' % str(depth), self.debug_messages_enabled)
        unique_id = str(uuid4())
        timestring = timestamp_to_datetime(timestamp)

        ## Append all summaries together, get a token total, and get a list of memory ids
        contents = []
        content_tokens = 0
        memory_ids = []
        for memory in memories:
            contents.append(memory['summary'])
            content_tokens += int(memory['summary_tokens'])
            memory_ids.append(str(memory['id']))
        content = '\n'.join(contents)
        summary_result = self.summarize_content(content, depth, content_tokens = content_tokens)
        summary = self.cleanup_response(summary_result)
        summary_tokens = get_token_estimate(summary)
        
        ## Extract the themes of this memory
        themes = self.__themes.extract_themes(content)

        ## Make new theme links and update old memories
        for memory in memories:
            total_themes = 0
            for theme_id in themes.keys():
                recurrence = themes[theme_id]['recurrence']
                total_themes += recurrence
                ## Create a new theme link record
                new_theme_id = str(uuid4())
                new_theme = self.__themes.create_theme_link_object(
                    id=new_theme_id,
                    depth=int(memory['depth']),
                    memory_id=memory['id'],
                    theme_id=theme_id,
                    recurrence=recurrence,
                    cooldown=0,
                    created_on=time(),
                    modified_on=time()
                )
                ## Insert new theme link record
                sql_insert_row('Theme_Links','id',new_theme)
            ## Update memory with new total_themes:
            new_total_themes = memory['total_themes']
            if new_total_themes < 0 or type(new_total_themes) != int:
                new_total_themes = 0
            new_total_themes += total_themes
            sql_update_row('Memories', 'id', {'id':memory['id'], 'total_themes':new_total_themes})

        ## Build episodic memory object
        episodic_memory = create_row_object(
            table_name='Memories',
            id=unique_id,
            depth=int(depth),
            summary=summary,
            summary_tokens=int(summary_tokens),
            episodic_children_ids=memory_ids,
            total_themes=0,
            created_on=time(),
            modified_on=time()
        )
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
        compression_occurred = False
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
            self.compress_memory_cache(depth)
            self.__episodic_memory_caches[depth].flush_memory_cache()
            self.__episodic_memory_caches[depth].add_memory(memory, tokens)
            compression_occurred = True
        self.index_memory(memory)
        if speaker == 'RAVEN':
            debug_message('Saving state...', self.debug_messages_enabled)
            self.save_state()

        return memory, tokens, compression_occurred

    ## Summarize all active memories and return the new memory id
    ## TODO: This process could cascade several memory caches, but only depth 1 caches will be returned.
    ## Need to account for this by passing a list down the line, then let the conversation manager
    ## decide how to utilize the returned memories. will figure that out when I am less drunk.
    def compress_memory_cache(self, depth):
        debug_message('Compressing cache of depth (%s)...' % str(depth), self.debug_messages_enabled)
        ## Get a copy of the cached memories
        memories = self.__episodic_memory_caches[int(depth)].memories

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
            prompt = self.__prompts.EideticSummary.get_prompt(speaker, content)
            response_tokens = self.__prompts.EideticSummary.response_tokens
            temperature = self.__prompts.EideticSummary.temperature
        elif int(depth) == 1:
            prompt = self.__prompts.EideticToEpisodicSummary.get_prompt(content)
            response_tokens = self.__prompts.EideticToEpisodicSummary.response_tokens
            temperature = self.__prompts.EideticToEpisodicSummary.temperature
        else:
            prompt = self.__prompts.EpisodicSummary.get_prompt(content)
            response_tokens = self.__prompts.EpisodicSummary.response_tokens
            temperature = self.__prompts.EpisodicSummary.temperature

        messages = [compose_gpt_message(prompt,'user')]
        memory_element, total_tokens = gpt_completion(messages, temperature, response_tokens)
        ## Save prompt and response
        prompt_row = create_row_object(
            table_name='Prompts',
            id=str(uuid4()),
            prompt=prompt,
            response=memory_element,
            tokens=total_tokens,
            temperature=temperature,
            comments=f"Summarization of memories at depth {depth}.",
            created_on=time()
        )
        sql_insert_row('Prompts','id',prompt_row)
        return memory_element

    ## Save memory locally, update local child memories, and save memory vector to pinecone
    def index_memory(self, memory):
        if not self.__pinecone_indexing_enabled:
            return
        depth = int(memory['depth'])
        memory_id = memory['id']
        memory_datetime = timestamp_to_detailed_datetime(memory['created_on'])
        memory_summary = f"[{memory_datetime}]\n{str(memory['summary'])}"
        debug_message('indexing memory (%s)' % memory_id, self.debug_messages_enabled)
        vector = gpt3_embedding(memory_summary)
        ## The metadata and namespace are redundant but I need the data split for later
        metadata = {'memory_type': 'episodic', 'depth': str(depth)}
        namespace = self.__config['memory_management']['memory_namespace_template'] % depth
        save_vector_to_pinecone(vector, memory_id, metadata, namespace)

    ## TODO: Raven will need a form of temporal recall as well
        # Temporal categorization can mostly be done programatically but some aspects will need subprompt processing
        # This will get tricky when it comes to decoding phrases like 'yesterday' or 'last time' considering each message
        # by the User could be days apart but still consolidated into the same context

    ## Memory recall happens in two phases (for now). Direct recall and thematic recall.
    ## Direct recall is a vector search on depth-0 memories, almost like a keyword search.
    ## Thematic recall is a vector search on Themes, which are generated in batches as memories transition from a lower depth to a higher depth.
    ## The results from either recalls will affect the end result. I imagine the following scenarios:
        # Direct and Theme recall produce strong candidates
            # The sets are similar
                # Choose the strongest candidates which match both sets
            # The sets mixed
                # Choose the strongest candidate from each set, even if they don't match
            # The sets are different
                # Choose the strongest candidate from a single set
        # Direct xor Theme recall produce strong candidates
            # Choose the stongest candidate
        # Neither recalls produce strong candidates
            # Recall was not successful, leave the section blank or notify Raven

    ## Return recalled memory ids and a boolean if the recall returned results
    def memory_recall(self, most_recent_message, conversation_log):
        debug_message('Beginning memory recall.', self.debug_messages_enabled)
        if conversation_log == '':
            breakpoint('Conversation log is empty. Skipping memory recall...')
            return [], False
        ## Determine if memory recall is necessary:
        recall_prompt = self.__prompts.RecallExtraction.get_prompt(conversation_log)
        recall_response_tokens = self.__prompts.RecallExtraction.response_tokens
        recall_temperature = self.__prompts.RecallExtraction.temperature
        recall_instructions = self.__prompts.RecallExtraction.system_instructions
        recall_messages = [compose_gpt_message(recall_instructions,'system'), compose_gpt_message(recall_prompt,'user')]
        recall_element, recall_total_tokens = gpt_completion(recall_messages, recall_temperature, recall_response_tokens)
        ## Save recall prompt and response
        recall_prompt_row = create_row_object(
            table_name='Prompts',
            id=str(uuid4()),
            prompt=recall_prompt,
            response=recall_element,
            system_message=recall_instructions,
            tokens=recall_total_tokens,
            temperature=recall_temperature,
            comments='Checking if memory recall is needed.',
            created_on=time()
        )
        sql_insert_row('Prompts','id',recall_prompt_row)
        
        recall_obj = string_to_json(recall_element)
        if 'sufficient_information' not in recall_obj:
            debug_message('Issue processing memory recall extraction object.', self.debug_messages_enabled)
            return [], False

        if recall_obj['sufficient_information']:
            debug_message('Sufficient information available, skipping memory recall...', self.debug_messages_enabled)
            return [], False

        ## Recall determined that more information is needed. Perform an explict search against the user's most recent message
        recalled_hyde = recall_obj['reasoning'] + ('' if (recall_obj['required_information'] == '') else '\n%s' % recall_obj['required_information'])
        relevant_result_obj = self.explicit_memory_recall(recalled_hyde, most_recent_message)

        if 'pertinent_information_present' not in relevant_result_obj:
            debug_message('Memory relevancy didn''t return any results...', self.debug_messages_enabled)
            return [], False
        elif relevant_result_obj['pertinent_information_present']:
            return relevant_result_obj['relevant_information_ids'], True
        return [], False

    ## TODO: If the explicit memory recall fails to produce results then thematic search and a lower threshold explicit search will be needed
    def explicit_memory_recall(self, hyde_query, most_recent_message, threshold = 0.8):
        relevant_obj = {}
        relevant_results = ''

        query_string = '%s\nUSER:\n%s' % (hyde_query, most_recent_message)
        query_vector = gpt3_embedding(query_string)
        query_namespace = (self.__config['memory_management']['memory_namespace_template']) % 0
        query_results = query_pinecone(vector = query_vector, return_n = 10, namespace = query_namespace)

        if query_results is not None:
            if len(query_results['matches']) > 0:
                relevant_content = ''
                recalled_memories = {}
                for match_memory in query_results['matches']:
                    if match_memory['score'] < 0.8:
                        continue
                    ## Load memory file
                    memories = sql_query_by_ids('Memories', 'id', [match_memory['id']])
                    if len(memories) == 0:
                        breakpoint(f"Unable to find memory {match_memory['id']}")
                        continue
                    memory_obj = memories[0]
                    ## Get memory contents
                    memory_id = memory_obj['id']
                    memory_date = timestamp_to_datetime(memory_obj['created_on'])
                    memory_content = memory_obj['content']
                    memory_speaker = memory_obj['speaker']
                    relevant_information_content = f"[\nINFORMATION ID: {memory_id}\nRECORDED ON: {memory_date}\nFROM: {memory_speaker}\nCONTENT: {memory_content}\n]\n"

                    recalled_memories[memory_id] = memory_obj
                    ## Append for relevant content body
                    relevant_content += relevant_information_content
                ## Determine if recalled memories are relevant:
                relevant_prompt = self.__prompts.RecallRelevancy.get_prompt(most_recent_message, hyde_query, relevant_content)
                relevant_response_tokens = self.__prompts.RecallRelevancy.response_tokens
                relevant_temperature = self.__prompts.RecallRelevancy.temperature
                relevant_instructions = self.__prompts.RecallRelevancy.system_instructions
                relevant_messages = [compose_gpt_message(relevant_instructions,'system'), compose_gpt_message(relevant_prompt,'user')]
                relevant_element, relevant_total_tokens = gpt_completion(relevant_messages, relevant_temperature, relevant_response_tokens)
                ## Save relevant memory prompt and response
                relevant_prompt_row = create_row_object(
                    table_name='Prompts',
                    id=str(uuid4()),
                    prompt=relevant_prompt,
                    response=relevant_element,
                    system_message=relevant_instructions,
                    tokens=relevant_response_tokens,
                    temperature=relevant_temperature,
                    comments='Checking if recalled memories are relevant.',
                    created_on=time()
                )
                sql_insert_row('Prompts','id',relevant_prompt_row)

                relevant_obj = string_to_json(relevant_element)
        return relevant_obj

    ## TODO: I need a "memory pruning" feature which properly removes memories from all places (pinecone, memory caches, ). If it triggered a memory compression it should remove those compressions. This will be tricky...


# breakpoint('Here is the hyde:')
        # print(recalled_hyde)
        # breakpoint('Here is the response: ')
        # print(explicit_query_results)
        # breakpoint('press any key to continue...')

        ## Prompt for themes
        # themes_result = self.__themes.extract_recall_themes(conversation_log)
        ## Cleanup themes
        # themes, theme_error = self.__themes.cleanup_theme_response(themes_result)

        ######################################################
        # theme_error = False
        # if theme_error:
        #     breakpoint('there was a thematic extraction error')
        # else:
        #     # theme_query_string = ','.join(themes)
        #     theme_query_string = explicit_query_string
        #     theme_query_vector = gpt3_embedding(theme_query_string)
        #     theme_query_namespace = self.__config['memory_management']['theme_namespace_template']
        #     theme_query_results = query_pinecone(theme_query_vector, 10, theme_query_namespace)

        #     breakpoint('Here is the theme search:')
        #     print(theme_query_string)
        #     breakpoint('Here is the response: ')
        #     print(theme_query_results)
        #     breakpoint('press any key to continue...')
        # ## Analyze the search results
        # theme_memory_ids = []
        # theme_ids = {}
        # theme_links = []
        # ## Each theme will be opened, memory ids extracted, secondary search on all memory ids, top results chosen.
        # if not theme_error and theme_query_results is not None:
        #     if len(theme_query_results['matches']) > 0:
        #         for theme_match in theme_query_results['matches']:
        #             ## Cache the theme ids, scores, and objects
        #             match_score = float(theme_match['score'])
        #             theme_id = theme_match['id']
        #             theme_ids[theme_id] = {}
        #             theme_ids[theme_id]['score'] = match_score

        #             ## Load theme files
        #             theme_filepath = self.__themes.get_theme_path(theme_id)
        #             theme_obj = load_json(theme_filepath)
        #             theme_ids[theme_id]['object'] = theme_obj
        #             for mem_id in theme_obj['links']:
        #                 theme_links.append(theme_obj['links'][mem_id])
        #         theme_memory_ids = list(set(map(lambda m: m['memory_id'], theme_links)))
        
        # explicit_memory_ids = []
        # if explicit_query_results is not None:
        #         if len(explicit_query_results['matches']) > 0:
        #             explicit_memory_ids = list(set(map(lambda m: m['id'], explicit_query_results['matches'])))
        #             ## These memory ids will be directly compared to the themeatic search

        # # breakpoint('Done recalling memories. Here are the intersection ids...')
        # breakpoint('Done recalling memories.')
        # # intersection_ids = list((set(theme_links)).intersection(set(explicit_memory_ids)))
        # # print(intersection_ids)
        # breakpoint('Here are the theme links...')
        # print(theme_links)
        # breakpoint('Here are the theme memory ids...')
        # print(theme_memory_ids)
        # breakpoint('Here are the explicit memory ids...')
        # print(explicit_query_results)
        # breakpoint('press any key to continue...')