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
from UtilityFunctions import *

## NOTE:Likely will move the OpenAi stuff later, not sure.
import openai

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
                            update_pinecone_vector(existing_theme_id, theme_string, theme_namespace)
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

    ## Create a Theme object and save it locally.
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
        ## TODO: Save the Theme object locally
        return theme_obj

    ## Create a link record to share between the Theme object and the Memory, then save it locally.
    def generate_theme_link(self, theme_id, memory_id, memory_depth):
        unique_id = str(uuid4())
        link = {
            'id':unique_id,
            'theme_id':theme_id,
            'memory_id':memory_id,
            'depth':int(memory_depth),
            'weight':0.0,
            'cooldown_count':0,
            'repeat_theme_count':0
        }
        ## TODO: Save the link record locally
        return link