import sqlite3

conn = sqlite3.connect('raven.sqlite')

## Memory Table
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
    created_on TEXT,
    modified_on TEXT
)
''')

## Themes Table
conn.execute('''
CREATE TABLE IF NOT EXISTS Themes (
    id TEXT PRIMARY KEY NOT NULL,
    phrases TEXT,
    theme_history TEXT,
    created_on TEXT,
    modified_on TEXT
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
    created_on TEXT,
    modified_on TEXT
)
''')

## Theme History Table
# conn.execute('''
# CREATE TABLE IF NOT EXISTS Theme_History (
#     id TEXT PRIMARY KEY NOT NULL,
#     theme_id TEXT,
#     phrase TEXT,
#     iteration INTEGER,
#     similarity REAL,
#     created_on TEXT,
#     modified_on TEXT
# )
# ''')

conn.commit()

