import sqlite3
import json
import os

# Connect to the database
conn = sqlite3.connect('raven.sqlite')
cursor = conn.cursor()

# Define the table name and column names
table_name = 'Memories'
column_names = ['id', 'depth', 'speaker', 'content', 'content_tokens', 'summary',
                'summary_tokens', 'episodic_parent_id', 'episodic_children_ids', 'past_sibling_id', 'next_sibling_id',
                'theme_link_ids', 'total_themes', 'create_date', 'modify_date']

# Loop through the JSON files in the folder
folder_path = 'memory_management/memory_stash/depth_0_stash'
for file_name in os.listdir(folder_path):
    if file_name.endswith('.json'):
        # Open the file and load the JSON data
        with open(os.path.join(folder_path, file_name), 'r') as file:
            data = json.load(file)

        # Extract the data into a tuple, following the column order
        values = (data['id'], data['depth'], data['speaker'], data['content'], data['content_tokens'],
                  data['summary'], data['summary_tokens'], data['episodic_parent_id'], '', data['past_sibling'],
                  data['next_sibling'], json.dumps(data['theme_links']), data['total_theme_count'],
                  data['timestring'], data['timestring'])

        # Create the SQL insert statement and execute it
        sql = f"INSERT INTO {table_name} ({','.join(column_names)}) VALUES ({','.join(['?']*len(column_names))})"
        cursor.execute(sql, values)

# Commit the changes and close the connection
conn.commit()
conn.close()
