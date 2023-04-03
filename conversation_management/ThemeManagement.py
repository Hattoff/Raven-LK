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
        self.__config = configparser.ConfigParser()
        self.__config.read('config.ini')
        self.__pinecone_indexing_enabled = self.__config.getboolean('pinecone', 'pinecone_indexing_enabled')

    ## Get a list of themes from a summary of a memory and prepare the theme embedding objects
    def extract_themes(self, content):
        print('Extracting themes...')
        extracted_themes = list()
        ## Prompt for themes
        themes_result = self.extract_content_themes(content)
        ## Cleanup themes
        themes, has_error = self.cleanup_theme_response(themes_result)
        if has_error:
            self.breakpoint('there was a thematic extraction error')
            return extracted_themes
        theme_namespace = self.__config['memory_management']['theme_namespace_template']
        theme_folderpath = self.__config['memory_management']['theme_stash_dir']
        theme_match_threshold = float(self.__config['memory_management']['theme_match_threshold'])
        print('themes extracted...')
        for theme in themes:
            theme = str(theme)
            vector = self.gpt3_embedding(theme)
            theme_matches = self.query_pinecone(vector, 1, namespace=theme_namespace)
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
                        existing_theme = self.load_json(theme_filepath)
                        if theme not in existing_theme['themes']:
                            existing_theme['themes'].append(theme)
                            theme_string = ','.join(existing_theme['themes'])
                            self.update_theme_vector(existing_theme_id, theme_string, theme_namespace)
                            existing_theme['theme_count'] = len(existing_theme['themes'])
                            self.save_json(theme_filepath, existing_theme)
                        ## Add to extracted theme list
                        if existing_theme_id not in extracted_themes:
                            extracted_themes.append(existing_theme_id)
                        continue
            print('Making new themes')
            ## The theme score falls under the threshold or there was no match, make a new theme
            unique_id = str(uuid4())

            
            ## Add the theme to pinecone before making it so that similar themes in the current list can be merged
            payload = [{'id': unique_id, 'values': vector}]
            self.save_payload_to_pinecone(payload, theme_namespace)

            ## Make theme and save it locally
            new_theme = self.generate_theme(unique_id)
            new_theme['themes'].append(theme)
            new_theme['theme_count'] = 1
            theme_filepath = '%s/%s.json' % (theme_folderpath, unique_id)
            self.save_json(theme_filepath, new_theme)

            ## Add to extracted theme list
            extracted_themes.append(unique_id)

        return extracted_themes

    ## Get themes
    def extract_content_themes(self, content):
        prompt_name = 'memory_theme'
        ## Load the prompt from a .json file
        prompt_obj = self.load_json('%s/%s.json' % (self.__config['memory_management']['memory_prompts_dir'], prompt_name))
        
        temperature = prompt_obj['summary']['temperature']
        response_tokens = prompt_obj['summary']['response_tokens']

        ## Generate memory element
        prompt_content = prompt_obj['summary']['system_message'] % (content)
        prompt = [self.compose_gpt_message(prompt_content,'user')]
        response, tokens = self.gpt_completion(prompt, temperature, response_tokens)
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
            self.breakpoint('\n\npausing...')
            return [], has_error

        dict_keys = list(themes_obj.keys())
        if len(dict_keys) > 1:
            print('ERROR: unknown response for theme extraction')
            print('Value from GPT:\n\n%s' % themes)
            print(themes_obj)
            self.breakpoint('\n\npausing...')
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
                self.breakpoint('\n\npausing...')
                return [], has_error
        return [], has_error

    ## Load the theme, regenerate the embedding string, get embedding, and update the pinecone record
    def update_theme_vector(self, theme_id, theme_string, namespace):
        if not self.__pinecone_indexing_enabled:
            return
        vector = self.gpt3_embedding(theme_string)
        update_response = self.__vector_db.update(id=theme_id,values=vector,namespace=namespace)

    ## Create a Theme object and save it locally.
    def generate_theme(self, theme_id):
        timestamp = time()
        timestring = self.timestamp_to_datetime(timestamp)
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

#####################################################
    ## Debug functions
    def breakpoint(self, message = '\n\nEnter to continue...'):
        input(message+'\n')
    
    def debug_message(self, message):
        if self.debug_messages_enabled:
            print(message)

######################################################
## Pinecone stuff
#### Query pinecone with vector. If search_all = True then name_space will be ignored.
    def query_pinecone(self, vector, return_n, namespace = "", search_all = False):
        if search_all:
            results = self.__vector_db.query(vector=vector, top_k=return_n)
        else:
            results = self.__vector_db.query(vector=vector, top_k=return_n, namespace=namespace)
        return results

    ## Seems like a useless function but I want this function to check for the config pinecone_indexing_enabled
    def save_payload_to_pinecone(self,payload, namespace):
        if not self.__pinecone_indexing_enabled:
            return
        self.__vector_db.upsert(payload, namespace=namespace)

    #### Save vector to pinecone
    def save_vector_to_pinecone(self, vector, unique_id, metadata, namespace=""):
        if not self.__pinecone_indexing_enabled:
            return
        self.debug_message('Saving vector to pinecone.')
        payload_content = {'id': unique_id, 'values': vector, 'metadata': metadata}
        payload = list()
        payload.append(payload_content)
        self.__vector_db.upsert(payload, namespace=namespace)


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