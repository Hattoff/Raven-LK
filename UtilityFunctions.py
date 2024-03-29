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
import openai
import sqlite3
_raven_update_debug = None

#####################################################
                ## File Management ##

def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()

def save_file(filepath, content):
    with open(filepath, 'w', encoding='utf-8') as outfile:
        outfile.write(content)

def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return json.load(infile)

def save_json(filepath, payload):
    with open(filepath, 'w', encoding='utf-8') as outfile:
        json.dump(payload, outfile, ensure_ascii=False, sort_keys=True, indent=2)

def string_to_json(content):
    return json.loads(content)

#####################################################

config = configparser.ConfigParser()
config.read('config.ini')
enable_all_debug_message = False

openai.api_key = open_file(config['open_ai']['api_key'])
pinecone_indexing_enabled = config.getboolean('pinecone', 'pinecone_indexing_enabled')
pinecone.init(api_key=open_file(config['pinecone']['api_key']), environment=config['pinecone']['environment'])
vector_db = pinecone.Index(config['pinecone']['index'])

def get_config():
    return config

def reload_config():
    global config
    config = configparser.ConfigParser()
    config.read('config.ini')

def timestamp_to_datetime(unix_time):
    if type(unix_time) == str:
        unix_time = float(unix_time)
    return (datetime.datetime.fromtimestamp(unix_time).strftime("%A, %B %d, %Y at %I:%M:%S%p %Z")).strip()

def timestamp_to_detailed_datetime(unix_time):
    if type(unix_time) == str:
        unix_time = float(unix_time)
    
    # Get the datetime object from the Unix timestamp
    dt = datetime.datetime.fromtimestamp(unix_time)
    
    # Convert the datetime to various string formats
    date_str = dt.strftime("%A, %B %d, %Y")
    time_str = dt.strftime("%I:%M:%S %p %Z").lstrip("0")
    month_day_str = dt.strftime("%B %d")
    year_month_day_str = dt.strftime("%Y-%m-%d")
    
    result = f"date: {date_str} time: {time_str} month-day: {month_day_str} year-month-day: {year_month_day_str}"
    # Return a dictionary of the datetime strings
    return result

#####################################################
                ## TikToken ##
def get_token_estimate(content):
    content = content.encode(encoding='ASCII',errors='ignore').decode()
    # encoding = tiktoken.encoding_for_model(str(config['open_ai']['model']))
    ## TODO: TikToken doesn't know about the new model names, will need to hard code for now.
    encoding = tiktoken.encoding_for_model('gpt-3.5-turbo-0301')
    tokens = encoding.encode(content)
    token_count = len(tokens)
    return token_count

#####################################################
                ## OpenAI ##
def gpt3_embedding(content):
    engine = config['open_ai']['input_engine']
    content = content.encode(encoding='ASCII',errors='ignore').decode()
    response = openai.Embedding.create(input=content,engine=engine)
    vector = response['data'][0]['embedding']
    return vector

def gpt_completion(messages, temp=0.0, tokens=400, stop=['USER:', 'RAVEN:'], print_response = False):
    engine = config['open_ai']['model']
    top_p=1.0
    freq_pen=0.0
    pres_pen=0.0

    max_retry = 15
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
            total_tokens = int(response['usage']['total_tokens'])
            response_str = response['choices'][0]['message']['content'].strip()
            if print_response:
                print_response_stats(response)
            return response_str, total_tokens
        except Exception as oops:
            retry += 1
            debug_message(f"Didn't get a response from OpenAI, trying again in {2*retry} seconds...")
            if retry >= max_retry:
                return "GPT3.5 error: %s" % oops, -1
            print('Error communicating with OpenAI:', oops)
            sleep(2*retry)

def print_response_stats(response):
    response_id = ('\nResponse %s' % str(response['id']))
    prompt_tokens = ('\nPrompt Tokens: %s' % (str(response['usage']['prompt_tokens'])))
    completion_tokens = ('\nCompletion Tokens: %s' % str(response['usage']['completion_tokens']))
    total_tokens = ('\nTotal Tokens: %s\n' % (str(response['usage']['total_tokens'])))
    print(response_id + prompt_tokens + completion_tokens + total_tokens)

## The role can be either system or user. If the role is system then you are either giving the model instructions/personas or example prompts.
## Then name field is used for example prompts which guide the model on how to respond.
## If the name field has data, the model will not consider them part of the conversation; the role will be user by default.
def compose_gpt_message(content, role, name=''):
    content = content.encode(encoding='ASCII',errors='ignore').decode() ## Cheeky way to remove encoding errors
    if name == '':
        return {"role":role, "content": content}
    else:
        role = 'user'
        return {"role":role,"name":name,"content": content}

#####################################################
                ## Debug ##
def breakpoint( message = '\n\nEnter to continue...'):
    input(message+'\n')

def debug_message(message, display_debug = False):
    if display_debug or enable_all_debug_message:
        print(message)

#####################################################
                ## Pinecone ##

def query_pinecone(vector, return_n, namespace = "", search_all = False):
    if search_all:
        results = vector_db.query(vector=vector, top_k=return_n)
    else:
        results = vector_db.query(vector=vector, top_k=return_n, namespace=namespace)
    return results

def enable_pinecone_indexing():
    global pinecone_indexing_enabled
    pinecone_indexing_enabled = True

def disable_pinecone_indexing():
    global pinecone_indexing_enabled 
    pinecone_indexing_enabled = False

def save_payload_to_pinecone(payload, namespace):
    if not pinecone_indexing_enabled:
        return
    vector_db.upsert(payload, namespace=namespace)

def save_vector_to_pinecone(vector, unique_id, metadata, namespace=""):
    if not pinecone_indexing_enabled:
        return
    debug_message('Saving vector to pinecone.')
    payload_content = {'id': unique_id, 'values': vector, 'metadata': metadata}
    payload = list()
    payload.append(payload_content)
    vector_db.upsert(payload, namespace=namespace)

def update_pinecone_vector(id, vector, namespace):
    if not pinecone_indexing_enabled:
        return
    update_response = {}
    update_response = vector_db.update(id=id,values=vector,namespace=namespace)
    return update_response

#####################################################
                ## SQLite3 ##

def get_sqldb():
    sqldb = sqlite3.connect(config['database']['database_name'])
    ## Convert the entire returned row into a dictionary object.
    def dict_factory(cursor, row):
        d = {}
        for idx, col in enumerate(cursor.description):
            value = row[idx]
            ## If the field value was originally an object then ensure it is re-converted back into that object.
            if isinstance(value, str) and ((value.startswith("[") and value.endswith("]")) or (value.startswith("{") and value.endswith("}"))):
                try:
                    ## try to parse JSON-like string
                    value = json.loads(value)
                except json.JSONDecodeError:
                    ## ignore parsing error and leave value as-is
                    pass
            d[col[0]] = value
        return d
    sqldb.row_factory = dict_factory
    return sqldb

## Return a dictionary template of a sqlite3 table
def get_table_template(table_name):
    sqldb = get_sqldb()
    cursor = sqldb.cursor()
    cursor.execute(f"PRAGMA table_info({table_name})")
    metadata = cursor.fetchall()
    template = {}
    for column in metadata:
        column_name = column['name']
        column_type = column['type']
        if column_name == "id":
            template[column_name] = None
        elif "INTEGER" in column_type:
            template[column_name] = -1
        elif "REAL" in column_type:
            template[column_name] = -1.0
        elif "TEXT" in column_type:
            template[column_name] = None
        elif "BLOB" in column_type:
            template[column_name] = b""
        else:
            template[column_name] = None
    cursor.close()
    sqldb.close()
    return template

## Build a row object of a particular table template and populate it with none, some, or all elemnts
def create_row_object(table_name, **kwargs):
    row = get_table_template(table_name)
    for key, value in kwargs.items():
        if key in row:
            row[key] = value
    return row

## Search a SQL Database table using a list of ids. Leaving ids blank will fetch all records.
def sql_query_by_ids(table_name, key_name, ids=None):
    if type(ids) == str:
        ids = [ids]
    sqldb = get_sqldb()
    cursor = sqldb.cursor()
    rows = []
    if ids is not None:
        id_list = ','.join('?' for _ in ids)
        query = f"SELECT * FROM {table_name} WHERE {key_name} IN ({id_list})"
        cursor.execute(query, ids)
        ## Convert rows to dictionary objects
        rows = cursor.fetchall()
    else:
        query = f"SELECT * FROM {table_name}"
        cursor.execute(query)
        ## Convert rows to dictionary objects
        rows = cursor.fetchall()
    cursor.close()
    sqldb.close()
    return rows

## Take a dictionary representing a row in a table and update all values in that row
def sql_update_row(table_name, primary_key_name, row):
    actual_table_columns = get_table_template(table_name)
    verified_row = {}
    for column in row:
        if column not in actual_table_columns:
            debug_messages(f"The key {column} does not exist as a column in table {table_name}. Skipping the data on this key.")
        else:
            verified_row[column] = row[column]
    ## You need at least two verified keys (an id and a column to update)
    if len(list(verified_row.keys())) <= 1:
        debug_message('Not enough information to insert this record')
    elif primary_key_name not in verified_row:
        debug_message(f"Primary key name provided ({primary_key_name}) was not in the verified rows. Skipping update.")    
    ## Update the modified_on date if that column exists
    if 'modified_on' in verified_row:
        verified_row['modified_on'] = time()
    sqldb = get_sqldb()
    set_clauses = []
    for column_name, value in verified_row.items():
        ## Skip the primary key column
        if column_name != primary_key_name:
            ## If the value is an object then dump the jsonified version
            if not isinstance(value, str) and value is not None:
                try:
                    ## try to dump the JSON object
                    value = json.dumps(value)
                except json.JSONDecodeError:
                    pass
            ## Otherwise escape single quotes to avoid issues with the UPDATE statement
            elif value is not None:
                try:
                    escaped_val = value.replace("'", "''")
                except AttributeError:
                    escaped_val = str(value).replace("'", "''")
                value = escaped_val
            ## If the value is blank or None then null it out
            if value is not None and value != '':
                set_clauses.append(f"{column_name} = '{value}'")
            else:
                set_clauses.append(f"{column_name} = NULL")
    ## Construct the UPDATE statement
    update_sql = f"UPDATE {table_name} SET {', '.join(set_clauses)} WHERE {primary_key_name} = '{verified_row[primary_key_name]}'"
    ## Attempt to update the record
    update_success = False
    cursor = sqldb.cursor()
    try:
        cursor.execute(update_sql)
        sqldb.commit()
        update_success = True
    except Exception as e:
        # Log the error message and rollback the transaction
        debug_message(f"Update failed: {str(e)}")
        sqldb.rollback()
        update_success = False
    cursor.close()
    sqldb.close()
    return update_success

## Insert a blank new record into the database then update it with the row contents.
def sql_insert_row(table_name, key_name, row):
    sqldb = get_sqldb()
    ## Ensure the row passed 
    actual_table_columns = get_table_template(table_name)
    if actual_table_columns == {}:
        debug_message(f"Table {table_name} does not exist. Skipping insert.")
        return
    if key_name not in actual_table_columns:
        debug_message(f"Key {key_name} is not a column of table {table_name}. Skipping insert.")    
    ## Create a dummy record with just the id; everything else is Null. Force the first element in this list to be the primary key
    column_names = sorted(actual_table_columns.keys(), key=lambda x: x != key_name)
    values = [None for _ in range(len(column_names))]
    ## Set the primary key value
    values[0] = row[key_name]
    update_sql = f"INSERT OR REPLACE INTO {table_name} ({', '.join(column_names)}) VALUES ({', '.join(['?' for _ in row.values()])})"
    cursor = sqldb.cursor()
    insert_success = False
    try:
        cursor.execute(update_sql, values)
        sqldb.commit()
        insert_success = True
        ## Update the new blank record with the actual information
        sql_update_row(table_name, key_name, row)
    except Exception as e:
        debug_message(f"Insert failed: {str(e)}")
        sqldb.rollback()
        insert_success = False
    cursor.close()
    sqldb.close()
    return insert_success

## Remove a record from the database
def sql_delete_row(table_name, key_name, id):
    sqldb = get_sqldb()
    update_sql = f"DELETE FROM {table_name} where {key_name} = '{id}'"
    cursor = sqldb.cursor()
    update_success = False
    try:
        cursor.execute(update_sql)
        sqldb.commit()
        print("Delete successful")
        update_success = True
    except Exception as e:
        print(f"Delete failed: {str(e)}")
        sqldb.rollback()
        update_success = False
    cursor.close()
    sqldb.close()
    return update_success

## Execute simple queries
def sql_custom_query(query, params = None):
    sqldb = get_sqldb()
    cursor = sqldb.cursor()
    if params is not None:
        if type(params) not in (tuple, list):
            params = (params,)
        cursor.execute(query, params)
    else:
        cursor.execute(query)
    results = cursor.fetchall()
    cursor.close()
    sqldb.close()
    return results