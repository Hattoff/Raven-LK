import sqlite3
import json

def dict_factory(cursor, row):
    d = {}
    for idx, col in enumerate(cursor.description):
        d[col[0]] = row[idx]
    return d

def sql_update_row(table_name, primary_key_name, row):
    set_clauses = []
    for column_name, value in row.items():
        # Skip the primary key column
        if column_name != primary_key_name:
            try:
                escaped_val = value.replace("'", "''")
            except AttributeError:
                escaped_val = str(value).replace("'", "''")
            set_clauses.append(f"{column_name} = '{escaped_val}'")

    # Construct the UPDATE statement
    update_sql = f"UPDATE {table_name} SET {', '.join(set_clauses)} WHERE {primary_key_name} = '{row[primary_key_name]}'"
    print(update_sql)
    conn = sqlite3.connect('raven.sqlite')
    cursor = conn.cursor()
    try:
        cursor.execute(update_sql)
        conn.commit()
        print("Update successful")
    except Exception as e:
        # Log the error message and rollback the transaction
        print(f"Update failed: {str(e)}")
        conn.rollback()
    conn.close()

def get_memories_by_id(ids):
    conn = sqlite3.connect('raven.sqlite')
    conn.row_factory = dict_factory
    cursor = conn.cursor()
    id_list = ','.join('?' for _ in ids)
    query = "select * from Memories where id in ({})".format(id_list)
    tmp = cursor.execute(query, ids)
    row = cursor.fetchone()
    conn.close()
    return row

def sql_query_by_ids(table_name, primary_key_name, ids):
    conn = sqlite3.connect('raven.sqlite')
    conn.row_factory = dict_factory
    cursor = conn.cursor()
    id_list = ','.join('?' for _ in ids)
    # query = f"SELECT * FROM {table_name} WHERE {primary_key_name} in ({})".format(id_list)
    # query = f"SELECT * FROM {table_name} WHERE {primary_key_name} IN ({','.join(['?']*len(id_list))})"
    query = f"SELECT * FROM {table_name} WHERE {primary_key_name} IN ({id_list})"
    cursor.execute(query, ids)
    ## Convert rows to dictionary objects
    rows = cursor.fetchall()
    cursor.close()
    return rows

def sql_upsert_row(table_name, primary_key_name, row):
    sqldb = sqlite3.connect('raven.sqlite')
    sqldb.row_factory = dict_factory
    column_names = sorted(row.keys(), key=lambda x: x != primary_key_name)
    values = [None for _ in range(len(column_names))]
    values[0] = row[primary_key_name]
    update_sql = f"INSERT OR REPLACE INTO {table_name} ({', '.join(column_names)}) VALUES ({', '.join(['?' for _ in row.values()])})"
    cursor = sqldb.cursor()
    upsert_success = False
    try:
        cursor.execute(update_sql, values)
        sqldb.commit()
        print("Upsert successful")
        upsert_success = True
        sql_update_row(table_name, primary_key_name, row)
    except Exception as e:
        print(f"Upsert failed: {str(e)}")
        sqldb.rollback()
        upsert_success = False
    return upsert_success

## Remove a record from the database
def sql_delete_row(table_name, primary_key_name, id):
    sqldb = sqlite3.connect('raven.sqlite')
    sqldb.row_factory = dict_factory
    ## Create a dummy record with just the id; everything else is Null.
    update_sql = f"DELETE FROM {table_name} where {primary_key_name} = '{id}'"
    print(update_sql)
    cursor = sqldb.cursor()
    upsert_success = False
    try:
        cursor.execute(update_sql)
        sqldb.commit()
        print("Upsert successful")
        upsert_success = True
    except Exception as e:
        print(f"Upsert failed: {str(e)}")
        sqldb.rollback()
        upsert_success = False
    cursor.close()
    return upsert_success