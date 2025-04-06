
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
import wave
import os
import re
import pandas as pd
from dotenv import load_dotenv
load_dotenv()
import speech_recognition as sr
from gtts import gTTS
from langchain_groq import ChatGroq
import json
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
os.environ["GROQ_API_KEY"]="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"
prompt='''

**Prompt:**  
You are a technical interviewer evaluating a candidate for a **Python Developer** role with **2.5 years of experience**. Your goal is to assess their conceptual understanding and practical knowledge across key topics by asking **10 questions in total**.  

### **Guidelines:**  
- Ask **concise** questions that can be answered **in 2 lines**.  
- **Do not** ask for code.  
- Assess based on the following topics and difficulty levels:  
  - **SQL** (4/5)  
  - **Python** (4/5)  
  
  - **Linux** (3/5)  
- Assign marks **liberally** out of **10**, ensuring that if the candidate demonstrates understanding, they receive **good marks**.  

### **JSON Output Format:**  
- **First question:** `"prev_question_marks"` should be an **empty string** (`""`).  
- **Intermediate questions:** `"prev_question_marks"` should contain the marks for the previous answer.  
- **Last question:** `"Question"` should be an **empty string** (`""`).  

### **Example Outputs:**  


**Intermediate Question:**  
```json
{
  "prev_question_marks": 7,
  "Question": "Why do we use pandas?"
}
```

**Last Question:**  
```json
{
  "prev_question_marks": 8,
  "Question": ""
}
```

Proceed with the interview by following these rules strictly.'''

llm=ChatGroq(model_name="llama3-70b-8192",temperature=0.7)
store={}

response_store=pd.DataFrame(columns=["cust_name","Q.no", "question", "answer", "marks"])

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]
model_with_memory=RunnableWithMessageHistory(llm,get_session_history)
def text_to_mp3(text,audio_file_name):
    tts = gTTS(text=text, lang="en",slow=False)
    tts.save(audio_file_name)
    return audio_file_name
# text_to_mp3("Hello, welcome to the fake app","/media/audio.mp3")
def restructured_response(ai_response):
    match = re.search(r'\{.*\}', ai_response, re.DOTALL)
    if match:
        json_str = match.group(0)  # Extract matched JSON string
        try:
            json_data = json.loads(json_str) 
            print(json_data)
            # Parse JSON
            return json_data
        except json.JSONDecodeError:
            print("Invalid JSON format.")
            return None
    else:
        print("No JSON found in response.")
        return None

def fetch_question(result,serial):
    config = {"configurable": {"session_id": serial}}

    data=model_with_memory.invoke((result),config=config).content
    print("data is "+data)
    response_json = restructured_response(data)
    question=response_json.get("Question")
    return question

def audio_processor(INPUT_FILENAME,serial,counter):
    global response_store
    result=audio_to_text(INPUT_FILENAME)
    
    new_row={}
    if (counter==0):

        new_row = {"Q.no": 1, "cust_name": serial, "question": "Tell me about yourself!"}
    if(counter<5):
        if counter==0:
            next_question=fetch_question(prompt,serial)
        else:
            next_question=fetch_question(result,serial)
        file_name=f"Q_{counter+1}_{serial}.mp3"
        OUTPUT_FILENAME=text_to_mp3(next_question,f"media/{file_name}")
        new_row = {"Q.no": counter+2, "cust_name": serial, "question": next_question }
    response_store = pd.concat([response_store, pd.DataFrame([new_row])], ignore_index=True)
    response_store.loc[(["Q.no"] == counter+1) & (response_store["cust_name"] == serial), "answer"] = result
    print(response_store)
 
    print("modified path is "+file_name)
    #response_store = pd.concat([response_store, pd.DataFrame([new_row])], ignore_index=True)
    
   
    return next_question,file_name
    


def audio_to_text(INPUT_FILENAME):
    r = sr.Recognizer()
    hellow=sr.AudioFile(INPUT_FILENAME)
    with hellow as source:
        audio = r.record(source)
    try:
        s = r.recognize_google(audio)
        print(s)
        return s
    except Exception as e: 
        print("Exception: "+str(e)) 
        return "error"
    




