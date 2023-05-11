import configparser
import os
import json
import glob
from time import time,sleep
from uuid import uuid4
from UtilityFunctions import *
from MemoryManagement import MemoryManager
from PromptManagement import PromptManager

class ConversationManager:
    def __init__(self):
        self.__config = get_config()
        self.__memory_manager = MemoryManager()
        self.__prompts = PromptManager()
        self.__eidetic_memory_log = self.MemoryLog(750,4,0)
        self.__episodic_memory_log = self.MemoryLog(750,4,1)
        self.make_required_directories()

    class MemoryLog:
        def __init__(self, max_log_tokens, min_log_count, depth):
            self.__depth = int(depth)
            self.__max_log_tokens = int(max_log_tokens)
            self.__min_log_count = int(min_log_count)
            self.__memory_ids = list()
            self.__token_count = 0
            self.__memory_string = ''
            
        def add(self, memory_id, tokens):
            self.__memory_ids.append(memory_id)
            self.__token_count += int(tokens)
            memory_count = len(self.__memory_ids)
            ## If the memory count is equal to the min log count then don't rebuild the list
            if self.__token_count > self.__max_log_tokens and memory_count > self.__min_log_count:
                self.refresh()
            else:
                self.__generate_memory_string()

        ## Rebuild memory ids list and update token_count
        def refresh(self):
            rolling_memory_log = self.get_rolling_memory_log()
            if len(rolling_memory_log) < self.__min_log_count:
                query = f"select id, created_on, sum(summary_tokens) over (partition by 1) as running_total from (select id, created_on, summary_tokens from Memories where depth = {self.__depth} order BY created_on desc limit {self.__min_log_count}) order by created_on"
                rolling_memory_log = sql_custom_query(query)

            if len(rolling_memory_log) <= 0:
                debug_message(f"Unable to load memories into rolling memory log of depth {self.__depth}...")
                self.__token_count = 0
                self.__memory_ids = list()
            else:
                self.__token_count = rolling_memory_log[0]['running_total']
                self.__memory_ids = list(map(lambda m: m['id'], rolling_memory_log))
            self.__generate_memory_string()
        
        def __generate_memory_string(self):
            memories = self.memories
            if len(memories) > 0:
                self.__memory_string = '\n'.join(m['summary'] for m in memories)
            else:
                self.__memory_string = ''

        @property
        def memory_string(self):
            return self.__memory_string

        @property
        def memory_count(self):
            return len(self.__memory_ids)
        
        @property
        def memories(self):
            results = []
            if len(self.__memory_ids) > 0:
                id_list = ','.join('?' for _ in self.__memory_ids)
                query = f"select * from Memories where depth = {self.__depth} and id in ({id_list}) order by created_on"
                results = sql_custom_query(query, self.__memory_ids)
            return results
        
        @property
        def memory_ids(self):
            return self.__memory_ids.copy()

        ## Query database for the most recent memories of a particular depth up to a token threshold
        def get_rolling_memory_log(self):
            query = f"""
            with recursive
            ranked_memories as
            (
                select
                    id
                    ,depth
                    ,summary_tokens
                    ,created_on
                    ,row_number() over (order by created_on desc) as row_num
                from
                    Memories
                where
                    depth = ?
            ),
            cumulative_sum as
            (
                select
                    id
                    ,depth
                    ,created_on
                    ,row_num
                    ,summary_tokens as running_total
                from
                    ranked_memories
                where
                    row_num = 1
                union all
                select
                    rm.id
                    ,rm.depth
                    ,rm.created_on
                    ,rm.row_num
                    ,cs.running_total + rm.summary_tokens as running_total
                from
                    ranked_memories as rm
                join 
                    cumulative_sum as cs on rm.row_num = cs.row_num + 1 
                    and rm.depth = cs.depth
            )
            select
                id
                ,depth
                ,created_on
                ,max(running_total) over (partition by 1) as running_total
            from
                cumulative_sum
            where
                running_total <= ?
            order by
                created_on
            """
            results = sql_custom_query(query, (self.__depth, self.__max_log_tokens))
            return results

    def make_required_directories(self):
        for d in self.__config['required_directories']:
            try:
                val = self.__config['required_directories'][d]
                path = ".\%s" % val.replace("/", "\\") 
                if not os.path.exists(path):
                    print('creating folder path: %s' % val)
                else:
                    print('folder path exists: %s' % val)
                os.makedirs(path, exist_ok=True)
            except OSError as err:
                print(err)

    ## Load rolling memory logs and chat history; -1 chat history load all past messages
    def load_state(self, chat_history_count = -1):
        self.__memory_manager.load_state()
        chat_history = []
        if chat_history_count == 0:
            return []
        elif chat_history_count > 0:
            ## Load some
            query = f"select * from (select * from Memories where depth = 0 order BY created_on desc limit {chat_history_count}) order by created_on"
            chat_history = sql_custom_query(query)
        else:
            query = 'select * from Memories where depth = 0 order by created_on'
            chat_history = sql_custom_query(query)
        self.__eidetic_memory_log.refresh()
        self.__episodic_memory_log.refresh()
        return chat_history
            
    def log_message(self, speaker, content):
        memory, tokens, compression_occurred = self.__memory_manager.create_new_memory(speaker, content)
        ## If there was a compression refresh all memory logs
        if compression_occurred:
            self.__eidetic_memory_log.refresh()
            self.__episodic_memory_log.refresh()
        else:
            ## Otherwise simply add the new memory
            self.__eidetic_memory_log.add(memory['id'], tokens)
    
    def generate_response(self):
        conversation = self.__eidetic_memory_log.memory_string
        if conversation == '':
            debug_message('Conversation is blank. Skipping generate response...')
            return ''

        ## Get anticipation
        anticipation = ''
        anticipation_tokens = 0
        anticipation_prompt = self.__prompts.Anticipation.get_prompt(conversation)
        anticipation_response_tokens = self.__prompts.Anticipation.response_tokens
        anticipation_temperature = self.__prompts.Anticipation.temperature
        anticipation, anticipation_tokens = gpt_completion([compose_gpt_message(anticipation_prompt,'user')], anticipation_temperature, anticipation_response_tokens)
        ## Save anticipation prompt and response
        anticipation_prompt_row = create_row_object(
            table_name='Prompts',
            id=str(uuid4()),
            prompt=anticipation_prompt,
            response=anticipation,
            tokens=anticipation_tokens,
            temperature=anticipation_temperature,
            comments='Anticipate user needs.',
            created_on=time()
        )
        sql_insert_row('Prompts','id',anticipation_prompt_row)


        ## Perform memory recall
        recalled = ''
        active_memory_count = self.__eidetic_memory_log.memory_count
        if active_memory_count > 1:
            active_memory_ids = self.__eidetic_memory_log.memory_ids
            active_memories = self.__eidetic_memory_log.memories
            most_recent_message = active_memories[active_memory_count-1]
            recent_conversation = '\n'.join(list(map(lambda x: x['summary'], active_memories)))
            recall_results, successful_recall = self.__memory_manager.memory_recall(most_recent_message['content'], conversation)
            recalled_memories = sql_query_by_ids('Memories', 'id', recall_results)
            recalled_memory_summaries = []
            for recalled_memory in recalled_memories:
                if recalled_memory['id'] in active_memory_ids:
                    continue
                recalled_memory_date = timestamp_to_datetime(recalled_memory['created_on'])
                recalled_memory_content = recalled_memory['content']
                recalled_memory_speaker = recalled_memory['speaker']
                recalled_memory_summaries.append(f"[\nRECORDED ON: {recalled_memory_date}\nFROM: {recalled_memory_speaker}\nCONTENT: {recalled_memory_content}\n]")
            recalled = '\n'.join(recalled_memory_summaries)
        
        ## Prompt conversation
        prompt_sections_list = []
        prompt_content_list = []
        compiled_conversation_content = ''
        if anticipation != '':
            prompt_content_list.append(f"ANTICIPATED USER NEEDS:\n{anticipation}")
        if recalled != '':
            prompt_sections_list.append('CONVERSATION HISTORY')
            prompt_content_list.append(f"CONVERSATION HISTORY:\n{recalled}")
        notes = self.__episodic_memory_log.memory_string
        if notes != '':
            prompt_sections_list.append('CONVERSATION NOTES')
            prompt_content_list.append(f"CONVERSATION NOTES:\n{notes}")
        prompt_sections_list.append('CONVERSATION LOG')
        prompt_content_list.append(f"CONVERSATION LOG:\n{conversation}")

        prompt_sections = ' and '.join(prompt_sections_list)
        prompt_content = '\n'.join(prompt_content_list)

        conversation_prompt = self.__prompts.Conversation.get_prompt(prompt_content, prompt_sections)
        conversation_response_tokens = self.__prompts.Conversation.response_tokens
        conversation_temperature = self.__prompts.Conversation.temperature
        conversation_response, conversation_tokens = gpt_completion([compose_gpt_message(conversation_prompt,'user')], conversation_temperature, conversation_response_tokens)
        ## Save conversation prompt and response
        conversation_prompt_row = create_row_object(
            table_name='Prompts',
            id=str(uuid4()),
            prompt=conversation_prompt,
            response=conversation_response,
            tokens=conversation_tokens,
            temperature=conversation_temperature,
            comments='Present conversation prompt.',
            created_on=time()
        )
        sql_insert_row('Prompts','id',conversation_prompt_row)        
        return conversation_response
