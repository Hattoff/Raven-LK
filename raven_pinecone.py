import configparser
import pinecone
config = configparser.ConfigParser()
config.read('config.ini')

pinecone.init(api_key=open_file(config['pinecone']['api_key']), environment=config['pinecone']['environment'])
vdb = pinecone.Index(config['pinecone']['index'])

#### Query pinecone with vector. If search_all = True then name_space will be ignored.
def query_pinecone(vector, return_n, name_space = "", search_all = False):
    if search_all:
        results = vdb.query(vector=vector, top_k=return_n)
    else:
        results = vdb.query(vector=vector, top_k=return_n, namespace=name_space)

#### Save vector to pinecone
def save_vector(vector, unique_id, metadata, name_space=""):
    payload_content = {'id': unique_id, 'values': vector, 'metadata': metadata}
    payload = list()
    payload.append(payload_content)
    vdb.upsert(payload, namespace=name_space)
    return unique_id

def update_vector(vector, unique_id, metadata, name_space=""):
    ## TODO: figure out how to update existing vectors.
    return unique_id