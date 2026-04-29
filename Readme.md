#How to use this RAG AI Teaching assistant on your own data 
## Step 1 - Collect your videos 
Move all your video files to videos folder
OR 
Directly download videos in MP3 format

## Step 2 - Convert to mp3
Convert all the video file to mp3 by running video to mp3

## Step 3 - Convert mp3 to json 
Convert all the mp3 file to json by running creat_json_chunks

## Step 4 - Convert the json files to vectors
Use the file read_chunks to convert the json files to a dataframe with embeddings and save it as a joblib pickle

## Step 5 - Prompt generation and feeding to llm
Read the joblib file and load it into the memory. Then create a relavant prompt as per the user query and feed it to the llm

#Better the llm you use, Accurate are the results 