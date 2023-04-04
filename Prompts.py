from UtilityFunctions import save_file, get_config
## Conversation
## Anticipation
## Notes
## Eidetic Summary
## Episodic Summary
## Theme

def prompt_anticipation(log, notes = None):
    prompt_temperature = 0.0
    response_tokens = 250
    prompt_sections = ("" if (notes is None) else "conversation notes and ") + "conversation log"
    content = ("" if (notes is None) else f"CONVERSATION NOTES:\n{notes}\n") + f"CONVERSATION LOG:\n{log}"
    prompt = f"Given the following {prompt_sections}, infer the USER's actual information needs. Attempt to anticipate what the user truly needs even if the USER does not fully understand it yet themselves, or is asking the wrong questions. However, the USER may change topics, in which case their needs will have changed. Emphasize the needs of the last message by the USER.\n{content}"
    return

def prompt_conversation(log, anticipation = None, notes = None):
    prompt_temperature = 0.7
    response_tokens = 600
    prompt_sections = ("" if (notes is None) else "conversation notes and ") + "conversation log"
    content = ("" if (anticipation is None) else f"ANTICIPATED USER NEEDS:\n{anticipation}\n") + ("" if (notes is None) else f"CONVERSATION NOTES:\n{notes}\n") + f"CONVERSATION LOG:\n{log}"
    prompt = f"I am a chatbot named RAVEN. My goals are to reduce suffering, increase prosperity, and increase understanding. I will review the {prompt_sections} below and then I will provide a detailed answer with emphasis on the last message by the user and my anticipation of their needs:\n{content}\nRAVEN:"
    return

def prompt_eidetic_summary(speaker, content):
    prompt_temperature = 0.0
    response_tokens = 250
    prompt = f"I will review the message authored by {speaker} and summarize it so that all key elements are represented in as little comprehensible text possible.\n{content}"
    return

def prompt_eidetic_to_episodic_summary(content):
    prompt_temperature = 0.0
    response_tokens = 500
    prompt = f"I will read the following conversation between USER and RAVEN below and then follow the directions in the INSTRUCTIONS section.\n{content}\nINSTRUCTIONS:I will summarize the conversation so that salient elements are represented in as little comprehensible text possible."
    return

def prompt_episodic_summary(content):
    prompt_temperature = 0.0
    response_tokens = 500
    prompt = f"I will condense the following notes so that all salient are represented in as little comprehensible text possible.\n{content}"
    return

def prompt_theme_extraction(content):
    prompt_temperature = 0.0
    response_tokens = 250
    prompt = f"Given the following chat log, identify the key themes of this information. Follow the INSTRUCTIONS at the end of the prompt.\n{content}\nINSTRUCTIONS:\nI will list all themes and format my response like this: {{\"themes\":[]}}"
    return