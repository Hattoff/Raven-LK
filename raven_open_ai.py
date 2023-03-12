import configparser
import openai
config = configparser.ConfigParser()
config.read('config.ini')
openai.api_key = open_file(config['open_ai']['api_key'])

def gpt3_embedding(content):
    engine = config['open_ai']['input_engine']
    content = content.encode(encoding='ASCII',errors='ignore').decode()  # fix any UNICODE errors
    response = openai.Embedding.create(input=content,engine=engine)
    vector = response['data'][0]['embedding']  # this is a normal list
    return vector

def gpt_completion(prompt, engine=config['open_ai']['model'], temp=0.0, top_p=1.0, tokens=400, freq_pen=0.0, pres_pen=0.0, stop=['USER:', 'RAVEN:']):
    if 'gpt-3.5-turbo' in engine:
        return gpt3_5_completion(prompt, engine, temp, top_p, tokens, freq_pen, pres_pen, stop)
    else:
        return gpt3_completion(prompt, engine, temp, top_p, tokens, freq_pen, pres_pen, stop)

def gpt3_5_completion(prompt, engine, temp, top_p, tokens, freq_pen, pres_pen, stop):
    max_retry = 5
    retry = 0
    prompt = prompt.encode(encoding='ASCII',errors='ignore').decode()
    while True:
        try:
            response = openai.ChatCompletion.create(
                model=engine,
                messages=[
                    {'role': 'user', 'content': prompt}
                ],
                temperature=temp,
                max_tokens=tokens,
                top_p=top_p,
                frequency_penalty=freq_pen,
                presence_penalty=pres_pen,
                stop=stop)

            print_response_stats(response)
            text = response['choices'][0]['message']['content'].strip()
            text = re.sub('[\r\n]+', '\n', text)
            text = re.sub('[\t ]+', ' ', text)
            filename = '%s_gpt3.txt' % time()
            foldername = config['raven']['gpt_log_dir']
            if not os.path.exists(foldername):
                os.makedirs(foldername)
            save_file('%s/%s' % (foldername,filename), prompt + '\n\n==========\n\n' + text)
            # print(text)
            # print('\nAbove are the text results from the prompt...')
            return text
        except Exception as oops:
            retry += 1
            if retry >= max_retry:
                return "GPT3.5 error: %s" % oops
            print('Error communicating with OpenAI:', oops)
            sleep(2)

def gpt3_completion(prompt, engine, temp, top_p, tokens, freq_pen, pres_pen, stop):
    max_retry = 5
    retry = 0
    prompt = prompt.encode(encoding='ASCII',errors='ignore').decode()
    while True:
        try:
            response = openai.Completion.create(
                engine=engine,
                prompt=prompt,
                temperature=temp,
                max_tokens=tokens,
                top_p=top_p,
                frequency_penalty=freq_pen,
                presence_penalty=pres_pen,
                stop=stop)
            text = response['choices'][0]['text'].strip()
            text = re.sub('[\r\n]+', '\n', text)
            text = re.sub('[\t ]+', ' ', text)
            filename = '%s_gpt3.txt' % time()
            foldername = config['raven']['gpt_log_dir']
            if not os.path.exists(foldername):
                os.makedirs(foldername)
            save_file('%s/%s' % (foldername,filename), prompt + '\n\n==========\n\n' + text)
            return text
        except Exception as oops:
            retry += 1
            if retry >= max_retry:
                return "GPT3 error: %s" % oops
            print('Error communicating with OpenAI:', oops)
            sleep(2)

def print_response_stats(response):
    response_id = ('\nResponse %s' % str(response['id']))
    prompt_tokens = ('\nPrompt Tokens: %s' % (str(response['usage']['prompt_tokens'])))
    completion_tokens = ('\nCompletion Tokens: %s' % str(response['usage']['completion_tokens']))
    total_tokens = ('\nTotal Tokens: %s\n' % (str(response['usage']['total_tokens'])))
    print(response_id + prompt_tokens + completion_tokens + total_tokens)