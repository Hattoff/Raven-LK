import configparser
import os
import json
import glob
import random
from time import time,sleep
import datetime
from uuid import uuid4
# import pinecone
# import tiktoken
import re
from PromptManagement import PromptManager
from UtilityFunctions import *

## NOTE:Likely will move the OpenAi stuff later, not sure.
# import openai

## The theme manager will extract themes from given content, create or merge existing Theme object, 
## create and manage links between memories and themes, manage the iterative theming of memories, and
## facilitate the splitting of themes as they diverge.

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

class ThemeManager:
    def __init__(self):
        self.__config = get_config()
        self.debug_messages_enabled = True
        self.__prompts = PromptManager()

    ## Get a list of themes from a summary of a memory and prepare the theme embedding objects
    def extract_themes(self, content):
        print('Extracting themes...')
        extracted_themes = {}
        ## Prompt for themes
        themes_result = self.extract_content_themes(content)
        ## Cleanup themes
        themes, has_error = self.cleanup_theme_response(themes_result)
        if has_error:
            breakpoint('there was a thematic extraction error')
            return extracted_themes
        theme_namespace = self.__config['memory_management']['theme_namespace_template']
        theme_folderpath = self.__prompts.ThemeExtraction.stash_path
        theme_match_threshold = float(self.__config['memory_management']['theme_match_threshold'])
        print('themes extracted...')
        for theme in themes:
            theme = str(theme)
            ## Embed this theme and check for the most similar Theme Object
            vector = gpt3_embedding(theme)
            theme_matches = query_pinecone(vector, 1, namespace=theme_namespace)
            if theme_matches is not None:
                if len(theme_matches['matches']) > 0:
                    ## There is a match so check the threshold
                    match_score = float(theme_matches['matches'][0]['score'])
                    if match_score >= theme_match_threshold:
                        ## Theme score is above the threshold, update an existing theme
                        existing_theme_id = theme_matches['matches'][0]['id']
                        ## Load, update, and save theme
                        theme_filepath = '%s/%s.json' % (theme_folderpath, existing_theme_id)
                        existing_theme = load_json(theme_filepath)
                        if theme not in existing_theme['themes']:
                            ## Add new theme to the themes list
                            existing_theme['themes'].append(theme)
                            ## Start tracking the history of this newly added theme
                            existing_theme['theme_history'][theme] = [self.generate_theme_history(0, match_score)]
                            ## Embed the new collection of themes
                            new_theme_string = ','.join(existing_theme['themes'])
                            new_theme_vector = gpt3_embedding(new_theme_string)
                            ## Update existing pinecone record's vector
                            update_pinecone_vector(existing_theme_id, new_theme_vector, theme_namespace)
                            existing_theme['theme_count'] = len(existing_theme['themes'])
                        else:
                            ## Get the existing theme history, the count elements is the new iteration number
                            history_count = len(existing_theme['theme_history'][theme])
                            ## Track the match_score of this theme
                            existing_theme['theme_history'][theme].append(self.generate_theme_history(history_count, match_score))
                        
                        ## Update the existing Theme Object
                        save_json(theme_filepath, existing_theme)
                            
                        ## Add to extracted theme id to list and keep track of how many times a similar theme has been extracted
                        if existing_theme_id not in extracted_themes:
                            extracted_themes[existing_theme_id] = {'recurrence':1, 'new_theme':False}
                        else:
                            extracted_themes[existing_theme_id]['recurrence'] += 1
                        continue
            ## The theme score falls under the threshold or there was no match, make a new theme
            unique_id = str(uuid4())

            ## Add the theme to pinecone before making it so that similar themes in the current list can be merged
            payload = [{'id': unique_id, 'values': vector}]
            save_payload_to_pinecone(payload, theme_namespace)

            ## Make theme and save it locally
            new_theme = self.generate_theme(unique_id)
            ## Add the theme to the list
            new_theme['themes'].append(theme)
            new_theme['theme_count'] = 1
            ## Initialize the theme history tracking. The similarity score is 1 (100%)
            new_theme['theme_history'][theme] = [self.generate_theme_history(0, 1.0)]
            theme_filepath = '%s/%s.json' % (theme_folderpath, unique_id)
            save_json(theme_filepath, new_theme)

            ## Add to extracted theme list
            extracted_themes[unique_id] = {'recurrence':1, 'new_theme':True}

        return extracted_themes

    ## Get themes
    def extract_content_themes(self, content):
        prompt = self.__prompts.ThemeExtraction.get_prompt(content)
        temperature = self.__prompts.ThemeExtraction.temperature
        response_tokens = self.__prompts.ThemeExtraction.response_tokens

        message = [self.compose_gpt_message(prompt,'user')]
        response, tokens = gpt_completion(message, temperature, response_tokens)
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
    
    ## Create a Theme object and save it locally.
    def generate_theme(self, theme_id, recurrence = 0):
        timestamp = time()
        timestring = timestamp_to_datetime(timestamp)
        theme_obj = {
            'id':theme_id,
            'themes':[],
            'theme_count':0,
            'links':{},
            'theme_history':{},
            'recurrence': int(recurrence),
            'timestamp':timestamp,
            'timestring':timestring,
            'update_embedding':False
        }
        ## TODO: Save the Theme object locally
        return theme_obj
    
    ## Object used to track theme history. I intend to use this to analyze theme decoherence.
    def generate_theme_history(self, iteration = 0, similarity = 0.0):
        timestamp = time()
        timestring = timestamp_to_datetime(timestamp)
        theme_history_obj = {
            'iteration':iteration,
            'similarity':similarity,
            'timestamp':timestamp,
            'timestring':timestring
        }
        return theme_history_obj

    ## Create a Link which will be stored on both the Theme and Memory
    def generate_theme_link(self, theme_id, memory_id, memory_depth = 0, recurrence = 0):
        unique_id = str(uuid4())
        timestamp = time()
        timestring = timestamp_to_datetime(timestamp)
        link = {
            'id':unique_id,
            'theme_id':theme_id,
            'memory_id':memory_id,
            'depth':int(memory_depth),
            'recurrence': int(recurrence),
            'weight':0.0,
            'cooldown_count':0,
            'timestamp':timestamp,
            'timestring':timestring
        }
        return link

    def compose_gpt_message(self, content, role, name=''):
        content = content.encode(encoding='ASCII',errors='ignore').decode() ## Cheeky way to remove encoding errors
        if name == '':
            return {"role":role, "content": content}
        else:
            role = 'system'
            return {"role":role,"name":name,"content": content}

    ## Thematic Re-Classification
        ## Process of re-classification will be to select from various themes, get a random assortment of memories, get a random variation of memory ranges, and re-theme them. Link strength will need to be updated based on the results. We will calculate the Mediant for new weights. If that is too drastic then we can use a hyperparameter to adjust the rate of correction.

    def retheme(self):
        themes = self._get_all_theme_paths()
        memory_base_path = self.__config['memory_management']['memory_stash_dir']
        memory_stash_template = self.__config['memory_management']['stash_folder_template']

        ## Get up to between 2 and 5 random themes, get a random link and the adjacent memories
        ## Recombine the memories and retheme
        ## Calculate new link weights
        for t in range(0, min(len(themes), 5)):
            theme_path = themes[random.randint(0,len(themes)-1)]
            debug_message('Retheming in process...', self.debug_messages_enabled)
            memories = {}
            theme = load_json(theme_path)
            random_memory = {}
            retry = 4
            attempt = 0
            done = False
            random_link = {}
            while not done:
                random_link_id = random.choice(list(theme['links'].keys()))
                random_link = theme['links'][random_link_id]
                if self._link_is_updateable(random_link):
                    print(random_link)
                    done = True
                else:
                    attempt += 1
                    ## Reload theme object because the cooldown was updated
                    theme = load_json(theme_path)
                    if attempt > retry:
                        done = True
            
            if 'id' not in random_link:
                debug_message(f'Unable to get a memory link from theme {theme_path}', self.debug_messages_enabled)
                continue
            
            memory_id = random_link['memory_id']
            memory_path = self._get_memory_path(memory_id, random_link['depth'])
            random_memory = load_json(memory_path)
            memories[memory_id] = {'object':random_memory, 'path':memory_path}

            ## Get a random bool and set the search direction
            coin_toss = bool(random.getrandbits(1))

            if coin_toss:
                search_direction = 'next_sibling'
            else:
                search_direction = 'past_sibling'
            
            ## If that direction ends early then reverse the direction
            if random_memory[search_direction] is None:
                if not coin_toss:
                    search_direction = 'next_sibling'
                else:
                    search_direction = 'past_sibling'    

            ## Get up to 2 or 5 memories in the randomly chosen search direction:
            for i in range(random.randint(2, 5)):
                memory_id = random_memory[search_direction]
                if memory_id is None:
                    break
                memory_path = self._get_memory_path(memory_id, random_link['depth'])
                random_memory = load_json(memory_path)
                memories[memory_id] = {'object':random_memory, 'path':memory_path}

            if len(memories) == 1:
                debug_message('Only one memory found. Skipping retheme.', self.debug_messages_enabled)
                continue

            ## Sort the list of memories from oldest to most recent
            sorted(memories.items(), key=lambda x: x[1]['object']['timestamp'], reverse=True)
            dict(memories)
            memory_keys = list(memories.keys())

            ## Concatenate the content from the memories and retheme
            content = ''
            content_tokens = 0
            for memory_id in memory_keys:
                content += '%s\n' % (memories[memory_id]['object']['summary'])

            ## Extract the themes from the content
            rethemes = self.extract_themes(content)
            rethemes_keys = list(rethemes.keys())
            
            ## Create new theme links if the theme is new
            new_memory_links = {}
            new_theme_links = {}
            for memory_id in memory_keys:
                new_memory_links[memory_id] = {}
                existing_memory_links = (memories[memory_id]['object']['theme_links']).keys()
                additional_recurrence_count = 0
                for theme_id in rethemes_keys:
                    recurrence = rethemes[theme_id]['recurrence']
                    additional_recurrence_count += recurrence
                    if rethemes[theme_id]['new_theme'] or theme_id not in existing_memory_links:
                        ## Create a new link but set the recurrance to 0 so we can update all links at the same time
                        link_obj = self.generate_theme_link(theme_id, memory_id, int(memories[memory_id]['object']['depth']), 0)
                        if theme_id not in new_theme_links:
                            new_theme_links[theme_id] = {}
                        new_theme_links[theme_id][memory_id] = link_obj
                        new_memory_links[memory_id][theme_id] = link_obj
                
                debug_message('Memory %s currently has total theme count:%s, we will be adding stats for %s themes for a total of %s recurrences.' % (memory_id, str(memories[memory_id]['object']['total_theme_count']), str(len(rethemes_keys)), str(additional_recurrence_count)), self.debug_messages_enabled)

                ## Update memory with retheme recurrances
                total_theme_count = memories[memory_id]['object']['total_theme_count'] + additional_recurrence_count
                memories[memory_id]['object']['total_theme_count'] = total_theme_count

                debug_message('Memory %s now has total theme count: %s'% (memory_id, str((memories[memory_id]['object']['total_theme_count']))), self.debug_messages_enabled)

                ## Add new links to memory theme link list
                memories[memory_id]['object']['theme_links'].update(new_memory_links[memory_id])
                updated_link_ids = (memories[memory_id]['object']['theme_links']).keys()
                for updated_link_id in updated_link_ids:
                    ## Increment the recurrence of the rethemed links
                    if updated_link_id in rethemes_keys:
                        recurrence = rethemes[updated_link_id]['recurrence']
                        memories[memory_id]['object']['theme_links'][updated_link_id]['recurrence'] += recurrence
                    
                    ## Update the link weights and cooldown
                    link_cooldown = int(self.__config['memory_management']['theme_link_cooldown'])
                    memories[memory_id]['object']['theme_links'][updated_link_id]['weight'] = memories[memory_id]['object']['theme_links'][updated_link_id]['recurrence'] / total_theme_count
                    memories[memory_id]['object']['theme_links'][updated_link_id]['cooldown_count'] = link_cooldown
                    ## Save update memory
                    save_json(memories[memory_id]['path'], memories[memory_id]['object'])

                    ## Update theme with updated link information
                    updated_theme_path = self._get_theme_path(updated_link_id)
                    updated_theme = load_json(updated_theme_path)
                    updated_theme['links'].update({memory_id: memories[memory_id]['object']['theme_links'][updated_link_id]})
                    save_json(updated_theme_path, updated_theme)
        return

    ## Return a list of relative file paths for all themes in the theme stash
    def _get_all_theme_paths(self):
        theme_glob = glob.glob('./%s/*' % self.__config['memory_management']['theme_stash_dir'])
        themes = list(map(lambda x: x.replace('\\','/'), list(theme_glob)))
        return themes
    
    ## Return file path for particular memory and depth
    def _get_memory_path(self, memory_id, depth):
        memory_base_path = self.__config['memory_management']['memory_stash_dir']
        memory_stash_template = self.__config['memory_management']['stash_folder_template']
        memory_folder_path = memory_stash_template % int(depth)
        memory_path = '%s/%s/%s.json' % (memory_base_path, memory_folder_path, memory_id)
        return memory_path

    ## Return file path for particular theme
    def _get_theme_path(self, theme_id):
        theme_path = ('./%s/%s.json') % (self.__config['memory_management']['theme_stash_dir'], theme_id)
        return theme_path

    ## If link is on cooldown decrement the cooldown counter, update the memory and theme, and return False; otherwise return True
    def _link_is_updateable(self, link):
        theme_id = link['theme_id']
        memory_id = link['memory_id']
        ## If the link cooldown count is zero then it can be updated
        if link['cooldown_count'] == 0:
            return True
        
        ## Check if the link cooldown is less than zero. If it is then set it to zero. For now it will act as though it was not updatable.
        if link['cooldown_count'] < 0:
            debug_message('WARNING: Cooldown count on link: theme_id %s, memory_id: %s was less than zero. Setting it to zero.' % (theme_id, memory_id), self.debug_messages_enabled)
            new_cooldown = 0
        else:
            new_cooldown = int(link['cooldown_count']) - 1
        link['cooldown_count'] = new_cooldown

        ## Update theme link and save
        theme_path = self._get_theme_path(theme_id)
        theme = load_json(theme_path)
        theme['links'].update({memory_id: link})
        save_json(theme_path, theme)

        ## Update memory link and save
        memory_path = self._get_memory_path(memory_id, link['depth'])
        memory = load_json(memory_path)
        memory['theme_links'].update({theme_id: link})
        save_json(memory_path, memory)
        return False

    ## Thematic Searching
        ## Process of thematic searching will be to search for themes of the current conversation, focusing on the last user message, getting a number of top results, getting the memories referenced in those results, and comparing the returned memories to the semantic search results and the theme link strengths. The results of these comparisons will produce several memories which can then be extended to their most recent neighbor for context, affixed with the timestamp, and summarized in contast with the user's message. This recall will then be added as a section in the conversation prompt.
    

