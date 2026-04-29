import requests
import os 
import json
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import joblib

def create_embedding(text_list):
#https://github.com/ollama/ollama/blob/main/docs/api.md#generate-embeddings
    # print(text_list)
    r= requests.post("http://localhost:11434/api/embed",
                     json= {"model":"bge-m3",
                            "input": text_list,
                            # "stream" : False
                              })
    
    response = r.json()["embeddings"]
    # print(response)
    return response



jsons = os.listdir("newjsons")    #Listing all jsons
my_dicts = []
chunk_id = 0

for json_file in jsons:
    with open(f"newjsons/{json_file}") as f:
        content = json.load(f)
    print(f"Creating Embeddings for {json_file}")
    
    embeddings = create_embedding([c['text'] for c in content['chunks']])
    # print(embeddings)

    for i, chunk in enumerate(content['chunks']):
        chunk['chunk_id'] = chunk_id
        chunk['embedding'] = embeddings[i]
        chunk_id += 1
        my_dicts.append(chunk)
    #     if(i==3):# Read 5 chunks
    #         break
    
        

df = pd.DataFrame.from_records(my_dicts)
# print(df)

#SAVE THIS DATAFRAME USING JOBLIB
joblib.dump(df, 'embeddings.joblib')




