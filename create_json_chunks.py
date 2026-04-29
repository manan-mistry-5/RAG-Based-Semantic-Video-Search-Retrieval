import whisper
import json
import os
model = whisper.load_model("large-v2")

audios = os.listdir("audios")

for audio in audios:
    # print(audio)
    number= audio.split("#")[1].split(".mp3")[0]
    title =audio.split(" Sigma")[0]
    # print(number, title)

    # result= model.transcribe(audio = f"audios/{audio}",              #(MAIN) LOOPS THROUGH ALL AUDIOS

    # result= model.transcribe(audio = "audios/sample_audio/sample.mp3",      #FOR SAMPLE AUDIO
    result= model.transcribe(audio = "audios/Your First HTML Website  Sigma Web Development Course - Tutorial #2.mp3",
                            language = "hi",
                            task = "translate",
                            word_timestamps=False)
                            
                           

    chunks=[]
    for segment in result["segments"]:
        chunks.append({"title": title,"number": number,"start": segment["start"], "end": segment["end"], "text": segment["text"]})

    chunks_with_metadata = {"chunks": chunks, "text": result["text"]}

    # with open(f"jsons/{audio}.json", "w") as f:            #( MAIN ) 
    #     json.dump(chunks_with_metadata,f)
    # with open("output.json", "w") as f:                    #FOR TESTING THE SAMPLE AUDIO
    #     json.dump(chunks_with_metadata,f)
    with open("Your First HTML Website  Sigma Web Development Course - Tutorial #2.mp3.json", "w") as f:
         json.dump(chunks_with_metadata,f)