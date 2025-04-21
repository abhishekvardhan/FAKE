
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
from .models import InterviewResponse
from langchain_groq import ChatGroq
import json
from fakeapp.models import InterviewResponse,IntervieweeDetails, IntervieweeSkill
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
os.environ["GROQ_API_KEY"]="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"
def fetch_prompt():
    with open("prompt.txt", "r") as file:
        prompt = file.read()
    return prompt

llm=ChatGroq(model_name="llama3-70b-8192",temperature=0.7)
store={}

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

def get_skills_by_session_id(session_id):
    try:

        interviewee = IntervieweeDetails.objects.get(session_id=session_id)
        # getch max questions from interviewee

        max_questions = interviewee.question_count      
        skills = interviewee.skills.all()  
        skill_list = [skill.skill_name for skill in skills]

        return skill_list, max_questions
    except IntervieweeDetails.DoesNotExist:
        print(f"Interviewee with session_id {session_id} does not exist.")
        return []
def update_prompt_with_skills(prompt,session_id):
    skills, total_questions = get_skills_by_session_id(session_id)
    if skills:
        skill_str = ", ".join(skills)
        replacements = {"thetopics": skill_str,"thismany": str(total_questions+1)}
        updated_prompt = prompt
        for old, new in replacements.items():
            updated_prompt = updated_prompt.replace(old, new)

        return updated_prompt
    else:
        print("No skills found for the given session ID.")
        return prompt

def fetch_question(result,serial):
    config = {"configurable": {"session_id": serial}}

    data=model_with_memory.invoke((result),config=config).content
    print("data is "+data)
    response_json = restructured_response(data)
    question=response_json.get("Question")
    return question

def fetch_prev_question(serial,counter):
    interviewee = IntervieweeDetails.objects.get(session_id=serial)
    # fetch question for given question number
    question = InterviewResponse.objects.filter(session_id=serial, question_number=counter).values_list('question_text', flat=True).first()
    if question:
        return question
    else:
        print(f"No question found for session ID {serial} and question number {counter}.")
        return None

def evaluate_answer(question, answer):
    """Evaluates an interview answer using ChatGroq and returns a score (0-10)."""
    
    prompt = f"""
    You are an expert interviewer. Evaluate the given answer based on correctness, clarity, and completeness.
    Score the answer on a scale of 0-10. Provide only the score as an integer.
    
    Question: {question}
    Answer: {answer}
    
    Score:
    """

    response = llm.invoke(prompt)
    
    try:
        text=response.content.strip()
        print(f"response is {text}")
        numbers = re.findall(r'\d+', text)
        score = int(numbers[0])
        return min(max(score, 0), 10)  
    except ValueError:
        return 0

def audio_processor(INPUT_FILENAME,serial,counter,max_questions):
    result=audio_to_text(INPUT_FILENAME)
    
    new_row={}

    if(counter<=max_questions):
        if counter==0:
            prompt=fetch_prompt()
            print(" actual prompt is "+prompt)
            prompt=update_prompt_with_skills(prompt,serial)
            print("updated prompt is "+prompt)
            next_question=fetch_question(prompt,serial)
        else:
            print("counter is "+str(counter))
            next_question=fetch_question(result,serial)
            new_row = {"qno": counter+1, "cust_name": serial, "question": next_question }
        file_name=f"Q_{counter+1}_{serial}.mp3"
        OUTPUT_FILENAME=text_to_mp3(next_question,f"media/{file_name}")

    pquestion=fetch_prev_question(serial,counter)

    marks=evaluate_answer(pquestion,result)

    print("modified path is "+file_name)

    return next_question,file_name,result,marks


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
    
def get_results_from_db(session_id):
    """Get interview results from the database"""
    responses = InterviewResponse.objects.filter(session_id=session_id).order_by('question_number')
    
    results = []
    for response in responses:
        # Skip the last question if it doesn't have an answer yet
        if not response.answer_text:
            continue
            
        results.append({

            'Question': response.question_text,
            'Answer': response.answer_text,
            'Score': response.score if response.score is not None else 'Not scored'
        })
    return results


