from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
import wave
import os
import re

from datetime import datetime
import pandas as pd
from dotenv import load_dotenv
load_dotenv()
import speech_recognition as sr
from gtts import gTTS
from . import fapp_resume_processor
from langchain_groq import ChatGroq
from groq import Groq
import json
from fakeapp.models import InterviewResponse,IntervieweeDetails, IntervieweeSkill, skillbased_interview, resumebased_interview
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
os.environ["GROQ_API_KEY"]=" "

import logging
logger = logging.getLogger('simple_logger')
import pdfplumber
import pytesseract
import PyPDF2
import docx
import textract
def extract_resume_text(file_path):
    file_extension = file_path.split('.')[-1].lower()
    
    if file_extension == 'pdf':
        try:
            # Try pdfplumber first for better formatting
            
            with pdfplumber.open(file_path) as pdf:
                text = ""
                for page in pdf.pages:
                    text += page.extract_text() or ""
            
            # If text extraction failed or returned empty, try pytesseract
            if not text.strip():
                from pdf2image import convert_from_path
                
                images = convert_from_path(file_path)
                text = ""
                for image in images:
                    text += pytesseract.image_to_string(image)
                    
        except Exception as e:
            print(f"Error with primary PDF methods: {e}")
            # Fallback to PyPDF2
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() or ""
                    
    elif file_extension == 'docx':
        import docx
        doc = docx.Document(file_path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
            
    elif file_extension in ['txt', 'rtf']:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            text = file.read()
            
    else:
        # Try textract for other formats
      
        text = textract.process(file_path).decode('utf-8')
        
    return text.strip()
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
def text_to_mp3(text, audio_file_name):
    """Convert text to MP3 using gTTS"""
    try:
        tts = gTTS(text=text, lang="en", slow=False)
        # Ensure the directory exists
        os.makedirs(os.path.dirname(audio_file_name), exist_ok=True)
        tts.save(audio_file_name)
        return audio_file_name
    except Exception as e:
        print(f"Error in text_to_mp3: {str(e)}")
        return None

def restructured_response(ai_response):
    match = re.search(r'\{.*\}', ai_response, re.DOTALL)
    if match:
        json_str = match.group(0)  # Extract matched JSON string
       
        json_data = json.loads(json_str) 
        print("json data from restructured_response is ",json_data)
        print(json_data)
        # Parse JSON
        return json_data
        
    

def get_skills_by_session_id(session_id):
    try:

        interviewee = IntervieweeDetails.objects.get(session_id=session_id)
        # getch max questions from interviewee
        max_questions= skillbased_interview.objects.filter(interviewee=interviewee).values_list('question_count', flat=True).first()

        skill_list = IntervieweeSkill.objects.filter(interviewee=interviewee).values_list('skill_name', flat=True)
        print("skills are ", skill_list )

        # skills = interviewee.skills.all()  
        # skill_list = [skill.skill_name for skill in skills]

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
    print("data is ",data)
    try:
        response_json = restructured_response(data)
        response_json=json.loads(response_json)
    except Exception as e:
        print("Error in restructured_response:", e)
        response_json = fapp_resume_processor.auto_repaid(data)
    finally:
        print("response json is ",response_json)
        question=response_json["Question"]
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



def audio_skill_processor(INPUT_FILENAME,serial,counter,max_questions):
    result=audio_to_text(INPUT_FILENAME)
    new_row={}
    score=28393
    print("result is "+result)
    print("session id is "+serial)
    if(counter<=max_questions):
        if counter==0:
            prompt=fetch_prompt()
        
            prompt=update_prompt_with_skills(prompt,serial)
            print("the first counter")
            next_question=fetch_question(prompt,serial)
        else:
            print("counter is "+str(counter))
            next_question=fetch_question(result,serial)
            new_row = {"qno": counter+1, "cust_name": serial, "question": next_question }
        file_name=f"Q_{counter+1}_{serial}.mp3"
        OUTPUT_FILENAME=text_to_mp3(next_question,f"media/{file_name}")
    pquestion=fetch_prev_question(serial,counter)
    eval_json=fapp_resume_processor.evaluate_response(pquestion,result)
    score=eval_json["score"]
    feedback=eval_json["feedback"]
    print("modified path is "+file_name)
    return next_question,file_name,result,score,feedback

def audio_resume_processor(INPUT_FILENAME,serial,counter,max_questions,state):
    result=audio_to_text(INPUT_FILENAME)
    print("result is "+result)
    if counter!=0:    
        question=state["list_of_questions"][-1]["question"]
    else:
        question="Tell me about yourself"
    
    evaluate_response=fapp_resume_processor.evaluate_response(question,result)
    score=evaluate_response["score"]
    feedback=evaluate_response["feedback"]
    if counter!=0:
        state["list_of_questions"][-1]["score"]=score
        state["list_of_questions"][-1]["feedback"]=feedback
        state["list_of_questions"][-1]["answer"]=result
    
    print("score is "+str(score), "feedback is "+feedback)
    # print(question,result,score,feedback)
    if counter==0:
        question="Tell me about yourself"
        q_a_json=[{"role":"assistant","content":question, "question_number":counter+1,"expected_answer":"Write a brief introduction about yourself"},{"role":"user","content":result}]
        state["current_question_type"]="project"
        state["current_question_number"]=0
        state["conversation_history"]=q_a_json

    if state["current_question_type"]=="project":
        if state["current_question_number"] < 3:
            print("#############Inside project############################")
            state = fapp_resume_processor.generate_project_questions(state)
            question=state["list_of_questions"][-1]["question"]
            state["current_question_number"] = state["current_question_number"]+1
        elif state["current_question_number"] >= 3:
            state["current_question_type"] = "technical"
            state["current_question_number"] = 0
    if state["current_question_type"]=="technical":
        if state["current_question_number"] < 4:
            print("#############Inside technical############################")
            state = fapp_resume_processor.generate_technical_questions(state)
            question=state["list_of_questions"][-1]["question"]
            print ("latest question is ",question)
            state["current_question_number"] += 1
        elif state["current_question_number"] >= 4:
            state["current_question_type"] = "scenario"
            state["current_question_number"] = 0
    if state["current_question_type"]=="scenario":
        if state["current_question_number"] < 2:
            print("#############Inside scenario############################")
            state = fapp_resume_processor.generate_scenario_questions(state)
            question=state["list_of_questions"][-1]["question"]
            state["current_question_number"] += 1
        elif state["current_question_number"] >= 2:
            state["current_question_type"] = "behavioral"
            state["current_question_number"] = 0
    if state["current_question_type"]=="behavioral":
        if state["current_question_number"] < 2:
            print("#############Inside behavioral############################")
            state = fapp_resume_processor.generate_behavioral_questions(state)
            question=state["list_of_questions"][-1]["question"]
            state["current_question_number"] += 1
        elif state["current_question_number"] >= 2:
            state["current_question_type"] = "perform_assessment"
            question="thank you for your time"
    file_name=f"Q_{counter+1}_{serial}.mp3"
    OUTPUT_FILENAME=text_to_mp3(question,f"media/{file_name}")
    print(state["current_question_type"],state["current_question_number"])
    print("______________________________")
    print("list of questions is ",state["list_of_questions"])
    

    return question, file_name,result, score, feedback," ",state


def audio_to_text(INPUT_FILENAME):
    try:
        client = Groq()
        with open(INPUT_FILENAME, "rb") as file:
            translation = client.audio.translations.create(file=(INPUT_FILENAME, file.read()), model="whisper-large-v3")
            text = translation["text"]
            print("text is "+text)
            return text
    except Exception as e:
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



def generate_excel_report(questions, assessment, serial):
    """
    Generate Excel report from assessment data and questions
    """
    # Create directory if it doesn't exist
    reports_dir = os.path.join('media', 'reports')
    os.makedirs(reports_dir, exist_ok=True)
    
    # Create filename with timestamp and serial number
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{serial}_{timestamp}_assessment.xlsx"
    filepath = os.path.join(reports_dir, filename)
    
    # Create DataFrame from questions and answers
    data = []
    for q in questions:
      
        q_text = q.get('question', '')
        q_expected_answer = q.get('expected_answer', '')
        q_type = q.get('question_type', '')
        q_score = q.get('score', 0)
        q_feedback = q.get('feedback', '')
        q_answer = q.get('answer', '')
        
        
        
        data.append(
            {
                "question": q_text,
                "expected_answer": q_expected_answer,
                "question_type": q_type,
                "score": q_score,   
                "feedback": q_feedback,
                "answer": q_answer

            }
        )
    
    # Create Excel writer
    df = pd.DataFrame(data)
    
    with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Questions', index=False)
        
        # Add summary sheet
        summary_data = {
            'Category': ['Total Score', 'Technical', 'Behavior', 'Project Level', 'Scenario Based', 'Resume-JD Similarity'],
            'Score': [
                assessment.get('overall_score', 0),
                assessment.get('technical_score', 0),
                assessment.get('behavioral_score', 0),
                assessment.get('project_score', 0),
                assessment.get('scenario_score', 0),
                assessment.get('resume_fit', 0)
                
            ]
        }
        pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
    
    # Return the relative URL path to the Excel file
    return f"/media/reports/{filename}"


def format_dashboard_data(assessment, excel_path):
    """
    Format assessment data for the dashboard
    """
    # Default strengths, weaknesses, and improvements
    default_strengths = [
        "Excellent technical knowledge",
        "Strong problem-solving skills",
        "Good communication skills"
    ]
    
    default_weaknesses = [
        "Limited experience with advanced technologies",
        "Could improve on system design concepts",
        "More practice needed with algorithms"
    ]
    
    default_improvements = [
        "Focus on learning cloud technologies",
        "Practice more complex system designs",
        "Strengthen knowledge of design patterns"
    ]
    
    # Get candidate name or use default
    candidate_name = assessment.get('Name', 'Candidate')
    
    # Create a clean dictionary with no duplicate keys
    dashboard_data = {
        "name": candidate_name,
        "totalScore": int(assessment.get('overall_score', 0)),
        "technical": int(assessment.get('technical_score', 0)),
        "behavior": int(assessment.get('behavioral_score', 0)), 
        "projectLevel": int(assessment.get('project_score', 0)),
        "scenarioBased": int(assessment.get('scenario_score', 0)),
        "resumeJdSimilarity": int(assessment.get('resume_fit', 0)),
        "strengths": assessment.get('strengths', default_strengths),
        "weaknesses": assessment.get('weaknesses', default_weaknesses),
        "improvements": assessment.get('improvements', default_improvements),
        "excelLink": excel_path
    }
    
    return dashboard_data

