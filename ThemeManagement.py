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

    def __theme_template(self):
        theme = {
            "id": None,
            "phrases":[],
            "theme_history_ids":[],
            "created_on":"",
            "modified_on":""
        }
        return theme

    def __theme_history_template(self):
        theme_history = {
            "id": None,
            "theme_id":None,
            "phrase":[],
            "iteration":-1,
            "similarity":-1.0,
            "created_on":"",
            "modified_on":""
        }
        return theme_history

    def __theme_link_template(self):
        theme_link = {
            "id": None,
            "depth":-1,
            "memory_id":None,
            "theme_id":None,
            "weight":-1.0,
            "recurrence":-1,
            "cooldown":-1,
            "created_on":"",
            "modified_on":""
        }
        return theme_link
    
    ## Build a theme object and pupulate it with none, some, or all columns/attributes
    def create_theme_object(self, **kwargs):
        return create_row_object('Themes', **kwargs)
    
    def create_theme_link_object(self, **kwargs):
        return create_row_object('Theme_Links', **kwargs)

    ## Get a list of themes from a summary of a memory and prepare the theme embedding objects
    def extract_themes(self, content):
        print('Extracting themes...')
        timestamp = str(time())
        extracted_themes = {}
        ## Prompt for themes
        themes_result = self.extract_content_themes(content)
        ## Cleanup themes
        themes, has_error = self.cleanup_theme_response(themes_result)
        if has_error:
            breakpoint('there was a thematic extraction error')
            return extracted_themes
        theme_namespace = self.__config['memory_management']['theme_namespace_template']
        theme_match_threshold = float(self.__config['memory_management']['theme_match_threshold'])
        print('themes extracted...')
        for phrase in themes:
            phrase = (str(phrase)).lower()
            ## Embed this theme and check for the most similar Theme Object
            vector = gpt3_embedding(phrase)
            theme_matches = query_pinecone(vector, 1, namespace=theme_namespace)
            if theme_matches is not None:
                if len(theme_matches['matches']) > 0:
                    ## There is a match so check the threshold
                    match_score = float(theme_matches['matches'][0]['score'])
                    if match_score >= theme_match_threshold:
                        ## Theme score is above the threshold, update an existing theme
                        existing_theme_id = theme_matches['matches'][0]['id']
                        ## Load, update, and save theme
                        query_themes = sql_query_by_ids('Themes','id',existing_theme_id)
                        if len(query_themes) == 0:
                            print(query_themes)
                            print(type(existing_theme_id))
                            breakpoint(f"Issue fetching existing theme from database: {existing_theme_id}")
                            continue
                        existing_theme = query_themes[0]
                        if phrase not in existing_theme['phrases']:
                            ## Add new phrase to the themes list
                            existing_theme['phrases'].append(phrase)
                             ## Start tracking the history of this newly added theme
                            if existing_theme['theme_history'] is None:
                                existing_theme['theme_history'] = {}
                            existing_theme['theme_history'].update({phrase: [self.generate_theme_history(0, match_score)]})
                            ## Embed the new collection of phrases
                            new_phrases_string = ','.join(existing_theme['phrases'])
                            new_phrases_vector = gpt3_embedding(new_phrases_string)
                            ## Update existing pinecone record's vector
                            update_pinecone_vector(existing_theme_id, new_phrases_vector, theme_namespace)
                        else:
                            ## Get the existing theme history, the count elements is the new iteration number
                            history_count = len(existing_theme['theme_history'][phrase])
                            ## Track the match_score of this theme
                            existing_theme['theme_history'][phrase].append(self.generate_theme_history(history_count, match_score))

                        ## Update the existing Theme Object
                        sql_update_row('Themes','id', existing_theme)
                            
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

            ## Make theme and save it to the database
            ## Initialize the theme history tracking. The similarity score is 1 (100%)
            new_theme = self.create_theme_object(
                    id=unique_id,
                    phrases=[phrase],
                    theme_history={phrase: [self.generate_theme_history(0, 1.0)]},
                    created_on=timestamp,
                    modified_on=timestamp
                )
            sql_insert_row('Themes','id', new_theme)

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

        ## Save anticipation prompt and response
        prompt_row = create_row_object(
            table_name='Prompts',
            id=str(uuid4()),
            prompt=prompt,
            response=response,
            tokens=tokens,
            temperature=temperature,
            comments='Extract themes.',
            created_on=str(time())
        )
        sql_insert_row('Prompts','id',prompt_row)

        return response

    ## Get themes
    def extract_recall_themes(self, content):
        prompt = self.__prompts.RecallThemeExtraction.get_prompt(content)
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
        
    ## Object used to track theme history. I intend to use this to analyze theme decoherence.
    def generate_theme_history(self, iteration = 0, similarity = 0.0):
        timestamp = str(time())
        theme_history_obj = {
            'iteration':iteration,
            'similarity':similarity,
            'created_on':timestamp
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
            'cooldown':0,
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
        ## Query all themes; we will choose from this list at random
        all_themes_query = sql_query_by_ids('Themes', 'id')
        if len(all_themes_query) <= 0:
            debug_message('No themes found, skipping retheme.')
            return
        all_themes = {x["id"]: x for x in all_themes_query}

        ## Get up to between 2 and 5 random themes to begin the process
        for t in range(0, min(len(all_themes.keys()), 5)):
            ## Get a random theme from the all_themes_list
            random_theme_id = random.choice(list(all_themes.keys()))
            debug_message(f"Retheming of theme {random_theme_id} in process...", self.debug_messages_enabled)
            random_theme = all_themes[random_theme_id]
            
            ## Pre-define some objects to avoid errors
            memories = {}
            random_memory = {}
            random_link = {}
            ## Limit retry attempts to 4
            retry = 4
            attempt = 0
            done = False
            while not done:
                ## Get all link ids associated with the random theme
                all_theme_links = {x["id"]: x for x in sql_query_by_ids('Theme_Links', 'theme_id', random_theme_id)}
                ## Choose a link id at random
                random_link_id = random.choice(list(all_theme_links.keys()))
                random_link = all_theme_links[random_link_id]
                if self.__link_is_updateable(random_link):
                    done = True
                else:
                    attempt += 1
                    ## Reload theme link because the cooldown was updated when it was checked
                    all_theme_links[random_link_id] = (sql_query_by_ids('Theme_Links', 'id', random_link_id))[0]
                    ## Clear out the random_link object in case we don't get an updatable link
                    random_link = {}
                    if attempt > retry:
                        done = True
            
            ## Check to be sure a random link was chosen
            if 'id' not in random_link:
                debug_message(f'Unable to get a memory link from theme {random_theme_id}', self.debug_messages_enabled)
                continue
            
            random_memory_id = random_link['memory_id']
            random_memory = (sql_query_by_ids('Memories', 'id', random_memory_id))[0]
            ## Keep track of all memories set to be rethemed
            random_memory_set = {}
            random_memory_set[random_memory_id] = random_memory

            ## Get a random bool and set the search direction
            coin_toss = bool(random.getrandbits(1))
            if coin_toss:
                search_direction = 'next_sibling_id'
            else:
                search_direction = 'past_sibling_id'
            
            ## If that direction ends early then reverse the direction
            if random_memory[search_direction] is None:
                if not coin_toss:
                    search_direction = 'next_sibling_id'
                else:
                    search_direction = 'past_sibling_id'    

            ## Get up to 2 or 5 memories in the randomly chosen search direction:
            for i in range(random.randint(2, 5)):
                sibling_memory_id = random_memory[search_direction]
                ## If there are no remaining sibling memories then exit
                if sibling_memory_id is None:
                    break
                ## Load sibling memory and add it to the set of memories to be rethemed
                random_memory_set[sibling_memory_id] = (sql_query_by_ids('Memories', 'id', sibling_memory_id))[0]

            if len(random_memory_set) <= 1:
                debug_message('Not enough memories found. Skipping retheme.', self.debug_messages_enabled)
                continue

            ## Sort the list of memories from oldest to most recent
            random_memory_set = dict(sorted(random_memory_set.items(), key=lambda x: x[1]["created_on"], reverse=True))
            random_memory_keys = list(random_memory_set.keys())

            ## Concatenate the content from each of the memories and retheme
            contents = []
            content_tokens = 0
            for m in random_memory_keys:
                contents.append(random_memory_set[m]['summary'])
            content = '\n'.join(contents)
            ## Extract the themes from the content
            retheme_results = self.extract_themes(content)
            rethemes_keys = list(retheme_results.keys())
            ## Get all of the theme items so we can update the record with new links if needed
            rethemes = sql_query_by_ids('Themes','id',rethemes_keys)

            ## Begin rethemeing process
            for memory_id in random_memory_keys:
                for retheme_id in rethemes_keys:
                    ## Query all existing existing links for this theme and get the associated memory ids
                    existing_memory_links = {x['memory_id']: x['id'] for x in sql_query_by_ids('Theme_Links', 'theme_id', retheme_id)}
                    ## Make sure the recurrence is at least 1
                    recurrence = retheme_results[retheme_id]['recurrence']
                    if recurrence <= 0:
                        recurrence = 1
                    ## Update the current memory's theme count
                    random_memory_set[memory_id]['total_themes'] += recurrence
                    if retheme_results[retheme_id]['new_theme'] or memory_id not in existing_memory_links:
                        ## If the theme is new or was never linked to this memory, create a new theme link record
                        timestamp = str(time())
                        new_theme_id = str(uuid4())
                        new_theme = self.create_theme_link_object(
                            id=new_theme_id,
                            depth=int(random_memory_set[memory_id]['depth']),
                            memory_id=memory_id,
                            theme_id=retheme_id,
                            recurrence=recurrence,
                            cooldown=2,
                            created_on=timestamp,
                            modified_on=timestamp
                        )
                        ## Insert new theme link record
                        sql_insert_row('Theme_Links','id',new_theme)
                    else:
                        ## Otherwise update the existing theme link record with the new weight
                        sql_update_row('Theme_Links','id',{'id':existing_memory_links[memory_id],'recurrence':recurrence,'cooldown':2})
                ## All theme links associated with this memory need to have their weights updated:
                weight_update_theme_links = sql_query_by_ids('Theme_Links', 'memory_id', memory_id)
                for wuth in weight_update_theme_links:
                    if wuth['recurrence'] <= 0:
                        wuth['recurrence'] = 1
                    wuth['weight'] = wuth['recurrence']/random_memory_set[memory_id]['total_themes']
                    sql_update_row('Theme_Links','id', wuth)
                ## Update memory record with the new total_themes count
                sql_update_row('Memories', 'id', {'id':memory_id, 'total_themes':random_memory_set[memory_id]['total_themes']})
        return

    ## If link is on cooldown decrement the cooldown counter, update the link, and return False; otherwise return True
    def __link_is_updateable(self, link):
        theme_id = link['theme_id']
        memory_id = link['memory_id']
        ## If the link cooldown count is zero then it can be updated
        if link['cooldown'] == 0:
            return True
        
        ## Check if the link cooldown is less than zero. If it is then set it to zero. For now it will act as though it was not updatable.
        if link['cooldown'] < 0:
            debug_message('WARNING: Cooldown on link: theme_id %s, memory_id: %s was less than zero. Setting it to zero.' % (theme_id, memory_id), self.debug_messages_enabled)
            new_cooldown = 0
        else:
            new_cooldown = int(link['cooldown']) - 1
        link['cooldown'] = new_cooldown
        ## Update theme link
        sql_update_row('Theme_Links','id',link)
        return False

    ## Thematic Searching
        ## Process of thematic searching will be to search for themes of the current conversation, focusing on the last user message, getting a number of top results, getting the memories referenced in those results, and comparing the returned memories to the semantic search results and the theme link strengths. The results of these comparisons will produce several memories which can then be extended to their most recent neighbor for context, affixed with the timestamp, and summarized in contast with the user's message. This recall will then be added as a section in the conversation prompt.
    

