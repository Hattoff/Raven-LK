import sqlite3
import configparser
config = configparser.ConfigParser()
config.read('config.ini')
conn = sqlite3.connect(config['database']['database_name'])

## Memories Table
conn.execute('''
CREATE TABLE IF NOT EXISTS Memories (
    id TEXT PRIMARY KEY NOT NULL,
    depth INTEGER,
    speaker TEXT,
    content TEXT,
    content_tokens INTEGER,
    summary TEXT,
    summary_tokens INTEGER,
    episodic_parent_id TEXT,
    episodic_children_ids TEXT,
    past_sibling_id TEXT,
    next_sibling_id TEXT,
    total_themes INTEGER,
    created_on REAL,
    modified_on REAL
)
''')

## Themes Table
conn.execute('''
CREATE TABLE IF NOT EXISTS Themes (
    id TEXT PRIMARY KEY NOT NULL,
    phrases TEXT,
    theme_history TEXT,
    created_on REAL,
    modified_on REAL
)
''')

## Theme Links Table
conn.execute('''
CREATE TABLE IF NOT EXISTS Theme_Links (
    id TEXT PRIMARY KEY NOT NULL,
    depth INTEGER,
    memory_id TEXT,
    theme_id TEXT,
    weight REAL,
    recurrence INTEGER,
    cooldown INTEGER,
    created_on REAL,
    modified_on REAL
)
''')

## Prompts Table
conn.execute('''
CREATE TABLE IF NOT EXISTS Prompts (
    id TEXT PRIMARY KEY NOT NULL,
    prompt TEXT,
    system_message TEXT,
    response TEXT,
    tokens INTEGER,
    temperature REAL,
    comments TEXT,
    created_on REAL
)
''')

## Memory States Table
conn.execute('''
CREATE TABLE IF NOT EXISTS Memory_States (
    id TEXT PRIMARY KEY NOT NULL,
    memory_cache_ids TEXT,
    created_on REAL,
    modified_on REAL
)
''')

## Memory Caches Table
conn.execute('''
CREATE TABLE IF NOT EXISTS Memory_Caches (
    id TEXT PRIMARY KEY NOT NULL,
    depth INTEGER,
    cache_token_limit INTEGER,
    max_tokens INTEGER,
    token_count INTEGER,
    memory_ids TEXT,
    first_memory_id TEXT,
    last_memory_id TEXT,
    past_cache_id TEXT,
    next_cache_id TEXT,
    created_on REAL,
    modified_on REAL
)
''')


## commit the changes to the database
conn.commit()

## close the connection
conn.close()

