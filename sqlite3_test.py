import sqlite3
import json
import configparser
from time import time,sleep
from uuid import uuid4
config = configparser.ConfigParser()
config.read('config.ini')

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

## Search a SQL Database table using a list of ids. Leaving ids blank will fetch all records.
def sql_query_by_ids(table_name, primary_key_name, ids=None):
    if type(ids) == str:
        ids = [ids]
    sqldb = get_sqldb()
    cursor = sqldb.cursor()
    if ids is not None:
        id_list = ','.join('?' for _ in ids)
        query = f"SELECT * FROM {table_name} WHERE {primary_key_name} IN ({id_list})"
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
    ## Update the modified_on date if that column exists
    timestamp = str(time()) 
    if 'modified_on' in row:
        row['modified_on'] = timestamp
    sqldb = get_sqldb()
    set_clauses = []
    for column_name, value in row.items():
        ## Skip the primary key column
        if column_name != primary_key_name:
            ## If the value is an object then dump the jsonified version
            if not isinstance(value, str) and value is not None:
                try:
                    # try to dump the JSON object
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
    update_sql = f"UPDATE {table_name} SET {', '.join(set_clauses)} WHERE {primary_key_name} = '{row[primary_key_name]}'"
    ## Attempt to update the record
    update_success = False
    cursor = sqldb.cursor()
    try:
        cursor.execute(update_sql)
        sqldb.commit()
        print("Update successful")
        update_success = True
    except Exception as e:
        # Log the error message and rollback the transaction
        print(f"Update failed: {str(e)}")
        sqldb.rollback()
        update_success = False
    cursor.close()
    sqldb.close()
    return update_success

## Insert a blank new record into the database then update it with the row contents.
def sql_insert_row(table_name, primary_key_name, row):
    sqldb = get_sqldb()
    ## Create a dummy record with just the id; everything else is Null. Force the first element in this list to be the primary key
    column_names = sorted(row.keys(), key=lambda x: x != primary_key_name)
    values = [None for _ in range(len(column_names))]
    ## Set the primary key value
    values[0] = row[primary_key_name]
    update_sql = f"INSERT OR REPLACE INTO {table_name} ({', '.join(column_names)}) VALUES ({', '.join(['?' for _ in row.values()])})"
    cursor = sqldb.cursor()
    insert_success = False
    try:
        cursor.execute(update_sql, values)
        sqldb.commit()
        print("Insert successful")
        insert_success = True
        ## Update the new blank record with the actual information
        sql_update_row(table_name, primary_key_name, row)
    except Exception as e:
        print(f"Insert failed: {str(e)}")
        sqldb.rollback()
        insert_success = False
    cursor.close()
    sqldb.close()
    return insert_success

## Remove a record from the database
def sql_delete_row(table_name, primary_key_name, id):
    sqldb = get_sqldb()
    def dict_factory(cursor, row):
        d = {}
        for idx, col in enumerate(cursor.description):
            d[col[0]] = row[idx]
        return d
    sqldb.row_factory = dict_factory
    update_sql = f"DELETE FROM {table_name} where {primary_key_name} = '{id}'"
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

## Build a row object with none, some, or all elemnts of a particular table template
def create_row_object(table_name, **kwargs):
    row = get_table_template(table_name)
    for key, value in kwargs.items():
        if key in row:
            row[key] = value
    return row

def create_memory_object(**kwargs):
    return create_row_object(table_name = 'Memories', **kwargs)