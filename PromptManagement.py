from UtilityFunctions import get_token_estimate, get_config

## Classes to store prompts and their metadata
class _Prompt:
    def __init__(self, temperature, response_tokens):
        self.config = get_config()
        self.__prompt_temperature = temperature
        self.__response_tokens = response_tokens
    
    def get_prompt(self):
        raise NotImplementedError()
    
    @property
    def temperature(self):
        return self.__prompt_temperature

    @property
    def response_tokens(self):
        return self.__response_tokens
    
    ## Each child class implements this by calling its get_prompt function and passing it to the get_token_estimate utility function
    @property
    def prompt_tokens(self):
        raise NotImplementedError()

    ## Return system instructions to guide GPT's response. This will normally be used to request special formatting of the response or interface with Python.
    @property
    def system_instructions(self):
        raise NotImplementedError()

    ## Return the file path where debug elements from this prompt can be saved
    @property
    def stash_path(self):
        raise NotImplementedError()

## Anticipate the needs of the USER based on the context of their conversation
class _Anticipation(_Prompt):
    def __init__(self, temperature, response_tokens):
        super().__init__(temperature, response_tokens)
        
    def get_prompt(self, log, notes = None):
        prompt_sections = ("" if (notes is None) else "conversation notes and ") + "conversation log"
        content = ("" if (notes is None) else f"CONVERSATION NOTES:\n{notes}\n") + f"CONVERSATION LOG:\n{log}"
        prompt = f"Given the following {prompt_sections}, infer the USER's actual information needs. Attempt to anticipate what the user truly needs even if the USER does not fully understand it yet themselves, or is asking the wrong questions. However, the USER may change topics, in which case their needs will have changed. Emphasize the needs of the last message by the USER.\n{content}"
        return prompt
    @property
    def prompt_tokens(self):
        return get_token_estimate(self.get_prompt(' ',' '))
    @property
    def stash_path(self):
        return self.config['conversation_management']['anticipation_stash_dir']

## Prompt RAVEN to address the USER's most recent message
class _Conversation(_Prompt):
    def __init__(self, temperature, response_tokens):
        super().__init__(temperature, response_tokens)
    def get_prompt(self, log, anticipation = None, notes = None):
        prompt_sections = ("" if (notes is None) else "conversation notes and ") + "conversation log"
        content = ("" if (anticipation is None) else f"ANTICIPATED USER NEEDS:\n{anticipation}\n") + ("" if (notes is None) else f"CONVERSATION NOTES:\n{notes}\n") + f"CONVERSATION LOG:\n{log}"
        prompt = f"I am a chatbot named RAVEN. My goals are to reduce suffering, increase prosperity, and increase understanding. I will review the {prompt_sections} below and then I will provide a detailed answer with emphasis on the last message by the user and my anticipation of their needs:\n{content}\nRAVEN:"
        return prompt
    @property
    def prompt_tokens(self):
        return get_token_estimate(self.get_prompt(' ',' ',' '))
    @property
    def stash_path(self):
        return self.config['conversation_management']['conversation_stash_dir']

## Summarize a message from either USER or RAVEN
class _EideticSummary(_Prompt):
    def __init__(self, temperature, response_tokens):
        super().__init__(temperature, response_tokens)            
    def get_prompt(self, speaker, content):
        prompt = f"I will review the message authored by {speaker} and summarize it so that all salient elements are represented in as little comprehensible text possible.\n{content}"
        return prompt
    @property
    def prompt_tokens(self):
        return get_token_estimate(self.get_prompt(' ',' '))
    @property
    def stash_path(self):
        return self.config['memory_management']['memory_prompts_dir']

## Summarize eidetic memories into an episodic memory
class _EideticToEpisodicSummary(_Prompt):
    def __init__(self, temperature, response_tokens):
        super().__init__(temperature, response_tokens)            
    def get_prompt(self, content):
        prompt = f"I will read the following conversation between USER and RAVEN below and then follow the directions in the INSTRUCTIONS section.\n{content}\nINSTRUCTIONS:I will summarize the conversation so that salient elements are represented in as little comprehensible text possible."
        return prompt
    @property
    def prompt_tokens(self):
        return get_token_estimate(self.get_prompt(' ',' ',' '))
    @property
    def stash_path(self):
        return self.config['memory_management']['memory_prompts_dir']

## Summarize a cache of other episodic memories
class _EpisodicSummary(_Prompt):
    def __init__(self, temperature, response_tokens):
        super().__init__(temperature, response_tokens)            
    def get_prompt(self, content):
        prompt = f"I will condense the following notes so that all salient elements are represented in as little comprehensible text possible.\n{content}"
        return prompt
    @property
    def stash_path(self):
        return self.config['memory_management']['memory_prompts_dir']

## Prompt to extract themes from a cache of memories
class _ThemeExtraction(_Prompt):
    def __init__(self, temperature, response_tokens):
        super().__init__(temperature, response_tokens)
    def get_prompt(self, content):
        prompt = f"Given the following chat log, identify the key themes of this information. Follow the INSTRUCTIONS at the end of the prompt.\n{content}\nINSTRUCTIONS:\nI will list all themes and format my response like this: {{\"themes\":[]}}"
        return prompt
    @property
    def prompt_tokens(self):
        return get_token_estimate(self.get_prompt(' '))
    @property
    def stash_path(self):
        return self.config['memory_management']['theme_stash_dir']
    
## Prompt to check if RAVEN needs more information
class _RecallExtraction(_Prompt):
    def __init__(self, temperature, response_tokens):
        super().__init__(temperature, response_tokens)
    def get_prompt(self, content):
        prompt = f"Review the conversation log between RAVEN and USER then follow the INSTRUCTIONS.\n{content}\nINSTRUCTIONS:\nBased on the information in the conversation log, with emphasis on the USER's last message, is there sufficient detailed information to address everything in USER's last message?"
        return prompt
    @property
    def prompt_tokens(self):
        return get_token_estimate(self.get_prompt(' ')) + get_token_estimate(self.system_instructions())
    @property
    def stash_path(self):
        return self.config['memory_management']['memory_recall_stash_dir']
    @property
    def system_instructions(self):
        return 'You are in interface to a Python program. All responses must conform to the following JSON schema:\n{\n\t\"type\": \"object\",\n\t\"properties\": {\n\t\t\"sufficient_information\": {\n\t\t\t\"type\": \"boolean\"\n\t\t},\n\t\t\"reasoning\": {\n\t\t\t\"type\": \"string\"\n\t\t},\n\t\t\"required_information\": {\n\t\t\t\"type\": \"string\"\n\t\t}\n\t},\n}'

## Prompt to extract themes with a focus on the last user message
class _RecallThemeExtraction(_Prompt):
    def __init__(self, temperature, response_tokens):
        super().__init__(temperature, response_tokens)
    def get_prompt(self, content):
        prompt = f"Given the following chat log, identify the key themes of this information. Follow the INSTRUCTIONS at the end of the prompt.\n{content}\nINSTRUCTIONS:\nWith emphasis on the USER's last message, list the themes of the user's request. Format your response like this: {{\"themes\":[]}}"
        return prompt
    @property
    def prompt_tokens(self):
        return get_token_estimate(self.get_prompt(' '))
    @property
    def stash_path(self):
        return self.config['memory_management']['theme_stash_dir']

## Prompt to check if recalled information is relevant to the conversation
class _RecallRelevancy(_Prompt):
    def __init__(self, temperature, response_tokens):
        super().__init__(temperature, response_tokens)
    def get_prompt(self, recent_message, potential_goals, relevant_content):
        content = f"MOST RECENT MESSAGE FROM USER:\n{recent_message}\nPOTENTIAL GOALS:\n{potential_goals}\nPOTENTIALLY RELEVANT INFORMATION:\n{relevant_content}"
        prompt = f"Review the most recent message from USER, the potential goals, and the potentially relevant information, then follow the INSTRUCTIONS at the end of the prompt.\n{content}\nINSTRUCTIONS:\nBased only on the potentially relevant information, is there any pertinent information related to addressing the USER's most recent message or the potential goals?"
        return prompt
    @property
    def prompt_tokens(self):
        return get_token_estimate(self.get_prompt(' '))
    @property
    def stash_path(self):
        return self.config['memory_management']['memory_recall_stash_dir']
    @property
    def system_instructions(self):
        return 'You are in interface to a Python program. All responses must conform to the following JSON schema:\n{\n\t\"type\": \"object\",\n\t\"properties\": {\n\t\t\"pertinent_information_present\": {\n\t\t\t\"type\": \"boolean\"\n\t\t},\n\t\t\"reasoning\": {\n\t\t\t\"type\": \"string\"\n\t\t},\n\t\t\"relevant_information_ids\": {\n\t\t\t\"type\": \"array\",\n\t\t  \"items\": {\n\t\t\t\"type\": \"string\"\n\t\t  }\n\t\t}\n\t}\n}'

## Initialize all prompt objects with their temperature and response token count
class PromptManager:
    def __init__(self):
        self.Anticipation = _Anticipation(0.0, 250)
        self.Conversation = _Conversation(0.7, 600)
        self.EideticSummary = _EideticSummary(0.0, 250)
        self.EideticToEpisodicSummary = _EideticToEpisodicSummary(0.0, 500)
        self.EpisodicSummary = _EpisodicSummary(0.0, 500)
        self.ThemeExtraction = _ThemeExtraction(0.0, 250)
        self.RecallExtraction = _RecallExtraction(0.7, 500)
        self.RecallThemeExtraction = _RecallThemeExtraction(0.0,250)
        self.RecallRelevancy = _RecallRelevancy(0.7, 500)
        
    ## Conversation prompts will combine several sections and a special prompt, this estimates the number of tokens needed before any additional content is added
    @property
    def conversation_token_buffer(self):
        token_buffer = self.Conversation.prompt_tokens + self.Anticipation.response_tokens + self.EideticToEpisodicSummary.response_tokens
        return token_buffer
