import configparser
import os
import json
import glob
from time import time,sleep
from uuid import uuid4
from UtilityFunctions import *
from MemoryManagement import MemoryManager

class ConversationManager:
    def __init__(self):
        self.__config = get_config()
        self.__memory_manager = MemoryManager()
        self.__eidetic_memory_log = self.MemoryLog(750,4)
        self.__episodic_memory_log = self.MemoryLog(750,4)
        self.make_required_directories()

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
    
        ## Reload memory log from a list of memories
        def load_memory_list(self, memories):
            if len(memories) > 0:
                if int(memories[0]['depth']) == 0:
                    ## Eideitc memory list
                    self.__memories = list(map(lambda m: (m, m['content_tokens']), memories))
                else:
                    ## Episodic memory list
                    self.__memories = list(map(lambda m: (m, m['summary_tokens']), memories))
                self.__check_memory_log()
                self.__generate_memory_string()


        ## Checks to see if the memory list needs rebuilt
        def __check_memory_log(self):
            memory_count = len(self.__memories)
            ## If the memory count is equal to the min log count then don't rebuild the list
            if self.__token_count > self.__max_log_tokens and memory_count > self.__min_log_count:
                ## Rebuild the memory list with a minumum number of memories, but continue to add memories until token count is reached
                memories = self.__memories.copy()
                ## Sort list so most recent memories are first
                memories = sorted(memories,key=lambda x: x[0]['created_on'],reverse=True)
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
                ## The memory list is backwards after this process so reverse it
                self.__memories.reverse()
                ## Once complete, update token count
                self.__token_count = token_count
        
        def __generate_memory_string(self):
            if len(self.__memories) > 0:
                if int(self.__memories[0][0]['depth']) == 0:
                    self.__memory_string = '\n'.join('%s: %s' % (m[0]['speaker'],m[0]['content']) for m in self.__memories)
                else:
                    self.__memory_string = '\n'.join(m[0]['summary'] for m in self.__memories)

        @property
        def memory_string(self):
            return self.__memory_string

        @property
        def memory_count(self):
            return len(self.__memories)
        
        @property
        def memories(self):
            return self.__memories.copy()
        
        @property
        def memory_ids(self):
            return list(map(lambda x: x[0], self.__memories))

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

    ## Load recent conversation log and conversation notes
    def load_state(self, recent_message_count = 2):
        self.reload_memory_log(0)
        self.reload_memory_log(1)
        return self.get_recent_messages(recent_message_count)

    def reload_memory_log(self, depth=0):
            depth = int(depth)
            cached_memories = self.__memory_manager.get_memories_from_cache(depth)
            memory_count = len(cached_memories)
            memories_to_reload = self.__memory_manager.get_previous_memory_ids_from_cache(depth)
            past_memory_count = len(memories_to_reload)
            if memory_count == 0 and past_memory_count == 0:
                ## No message history to load...
                return
            else:
                ## There may be a mix of currently cached memories and past memories so we will load the mixture
                if memory_count > 0:
                    for m in range(memory_count):
                        ## Add id of currently cached memory
                        memories_to_reload.append(cached_memories[m]['id'])
                ## Reload the memories from file instead of from the cache so you can do all at once
                memory_folder = self.__config['memory_management']['stash_folder_template'] % depth
                memory_file_path = '%s/%s' % (self.__config['memory_management']['memory_stash_dir'], memory_folder)
                reloaded_memories = list()
                for mem_id in memories_to_reload:
                    memory_path = '%s/%s.json' % (memory_file_path, mem_id)
                    try:
                        memory = load_json(memory_path)
                        reloaded_memories.append(memory)
                    except Exception as err:
                        print('ERROR: unable to load memory %s.json from %s' % (mem_id, memory_path))
                if depth == 0:
                    self.__eidetic_memory_log.load_memory_list(reloaded_memories)
                else:
                    self.__episodic_memory_log.load_memory_list(reloaded_memories)

    def get_recent_messages(self, message_count = 2):
        message_count = min(int(message_count), self.__eidetic_memory_log.memory_count)
        if message_count <= 0:
            ## No messages to display
            return list()
        ## Display recent messages with dates and speaker stamps for context.
        eidetic_memories = list(map(lambda m: m[0], self.__eidetic_memory_log.memories))
        ## Sort list so most recent message is first
        eidetic_memories = sorted(eidetic_memories, key=lambda x: x['timestamp'], reverse=True)
        eidetic_memories = eidetic_memories[0:message_count]
        eidetic_memories.reverse()
        return eidetic_memories
            
    def log_message(self, speaker, content):
        eidetic_memory, eidetic_tokens, episodic_memory, episodic_tokens = self.__memory_manager.create_new_memory(speaker, content)
        self.__eidetic_memory_log.add(eidetic_memory, eidetic_tokens)
        if episodic_memory is not None:
            self.__episodic_memory_log.add(episodic_memory, episodic_tokens)
    
    def generate_response(self):
        prompt_obj = load_json('%s/%s.json' % (self.__config['conversation_management']['conversation_management_dir'], 'conversation_prompt'))

        notes_body = self.__episodic_memory_log.memory_string
        conversation_body = self.__eidetic_memory_log.memory_string
        anticipation_body = self.__get_anticipation_response(conversation_body, prompt_obj)
        
        recalled_body = ''
        if self.__eidetic_memory_log.memory_count > 0 and self.__episodic_memory_log.memory_count > 0:
            most_recent_memory = (self.__eidetic_memory_log.memories[self.__eidetic_memory_log.memory_count-1])[0]
            most_recent_message = most_recent_memory['content']
            
            recalled_ids, memory_recall_performed, memory_recall_successful = self.__memory_manager.memory_recall(most_recent_message, conversation_body, notes_body)
            active_memory_ids = self.__eidetic_memory_log.memory_ids
            
            if len(recalled_ids) > 0:
                memory_stash_folder = self.__config['memory_management']['memory_stash_dir']
                memory_depth_folder = (self.__config['memory_management']['stash_folder_template']) % 0

                for recalled_memory_id in recalled_ids:
                    if recalled_memory_id in active_memory_ids:
                        continue
                    memory_filepath = '%s/%s/%s.json' % (memory_stash_folder, memory_depth_folder, recalled_memory_id)
                    recalled_memory_obj = load_json(memory_filepath)
                    recalled_memory_date = recalled_memory_obj['timestring']
                    recalled_memory_content = recalled_memory_obj['content']
                    recalled_memory_speaker = recalled_memory_obj['speaker']
                    recalled_memory_summary = f"[\nRECORDED ON: {recalled_memory_date}\nFROM: {recalled_memory_speaker}\nCONTENT: {recalled_memory_content}\n]\n"
                    recalled_body += recalled_memory_summary

        prompt_instruction_sections = ['conversation log']
        prompt_body = 'ANTICIPATED USER NEEDS:\n%s\n' % anticipation_body
        if recalled_body != '':
            prompt_instruction_sections.append('conversation history')
            prompt_body +='CONVERSATION HISTORY:\n%s\n' % recalled_body

        if self.__episodic_memory_log.memory_count > 0:
            prompt_instruction_sections.append('conversation notes')
            prompt_body +='CONVERSATION NOTES:\n%s\n' % notes_body
        prompt_body += 'CONVERSATION LOG:\n%s' % conversation_body
        
        prompt_instructions = ' and '.join(prompt_instruction_sections)
        response = self.__get_conversation_response(prompt_body, prompt_instructions, prompt_obj)
        
        return response

    def __get_conversation_response(self, prompt_body, prompt_instructions, prompt_obj):
        conversation_log_obj = prompt_obj['conversation']

        prompt = conversation_log_obj['prompt'] % (prompt_instructions, prompt_body)
        temperature = float(conversation_log_obj['temperature'])
        response_tokens = int(conversation_log_obj['response_tokens'])

        timestring = str(time())
        conversation_stash_dir = self.__config['conversation_management']['conversation_stash_dir']
        ## Save prompt for debug
        prompt_path = '%s/%s_prompt.txt' % (conversation_stash_dir, timestring)
        save_file(prompt_path, prompt)

        gpt_messages = list()
        gpt_messages.append(self.compose_gpt_message(prompt, 'user'))

        response, tokens = gpt_completion(gpt_messages, temperature, response_tokens)

        ## Save response for debug
        response_path = '%s/%s_response.txt' % (conversation_stash_dir, timestring)
        save_file(response_path, response)

        return response

    ## Generate the conversation's anticipation section
    def __get_anticipation_response(self, conversation_body, prompt_obj):
        anticipation_obj = prompt_obj['anticipation']

        notes_body = self.__episodic_memory_log.memory_string
        if self.__episodic_memory_log.memory_count > 0:
            prompt_instructions = 'chat notes and chat log'
            prompt_body = 'CHAT NOTES:\n%s\nCHAT LOG:\n%s' % (notes_body, conversation_body)
        else:
            prompt_instructions = 'chat log'
            prompt_body = conversation_body

        prompt = anticipation_obj['prompt'] % (prompt_instructions, prompt_body)
        temperature = float(prompt_obj['anticipation']['temperature'])
        response_tokens = int(prompt_obj['anticipation']['response_tokens'])

        timestring = str(time())
        anticipation_stash_dir = self.__config['conversation_management']['anticipation_stash_dir']
        ## Save prompt for debug
        prompt_path = '%s/%s_prompt.txt' % (anticipation_stash_dir, timestring)
        save_file(prompt_path, prompt)

        gpt_messages = list()
        gpt_messages.append(self.compose_gpt_message(prompt, 'user'))
        response, tokens = gpt_completion(gpt_messages, temperature, response_tokens)

        ## Save response for debug
        response_path = '%s/%s_response.txt' % (anticipation_stash_dir, timestring)
        save_file(response_path, response)

        return response

    def compose_gpt_message(self, content, role, name=''):
        if name == '':
            return {"role":role, "content": content}
        else:
            role = 'system'
            return {"role":role,"name":name,"content": content}
