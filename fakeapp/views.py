import string
import os
from django.shortcuts import render
from gtts import gTTS
from django.conf import settings

from django.views.decorators.csrf import csrf_exempt 
from django.http import JsonResponse
import random
from . import fapp_processor
from fakeapp.models import InterviewResponse,IntervieweeDetails, IntervieweeSkill, skillbased_interview, resumebased_interview
from urllib.parse import unquote
# Ensure counter is defined
import logging
logger = logging.getLogger('simple_logger')

def get_skills(request):
    # Initialize session variables
    request.session.flush()
    request.session["interview_type"] = "skill_based"
    return render(request, 'temp1.html')
@csrf_exempt
def save_user_info(request):
    if request.method == "POST":
        try:
            name = request.POST.get("Name")
            session_id=str(''.join(random.choices(string.ascii_letters + string.digits, k=6)))
            Intervieweeobj = IntervieweeDetails.objects.create(name=name, session_id=session_id )
            print(request.POST.get("interview_type"))
            if request.POST.get("interview_type") == "skill_based":
                selected_skills = request.POST.getlist("skills")
                print("Name:",name)
                print("Selected skills:", selected_skills)
                no_of_questions=request.POST.get("question_count")

                print("No of questions:", no_of_questions)
                
                request.session["session_id"] = session_id
                MAX_CYCLES=int(no_of_questions)
                request.session["MAX_CYCLES"]=MAX_CYCLES
                skillbased_interview_obj = skillbased_interview.objects.create(interviewee=Intervieweeobj, question_count=MAX_CYCLES)
                for skill in selected_skills:
                    IntervieweeSkill.objects.create(interviewee=Intervieweeobj, skill_name=skill)
            else:
                # resumebased_interview_obj = resumebased_interview.objects.create(interviewee=Intervieweeobj, )
                request.session["interview_type"] = "resume_based"
                resume_file = request.FILES.get("resume")
                
                print("Resume file:", resume_file)
                jd= request.POST.get("job_description", "")
                
                
                #save the resume file to the media directory
                if resume_file:
                    # Ensure the directory exists
                    resume_dir = os.path.join(settings.MEDIA_ROOT, "resumes")
                    os.makedirs(resume_dir, exist_ok=True)
                    
                    resume_path = os.path.join(resume_dir, session_id+resume_file.name)
                    with open(resume_path, "wb") as f:
                        for chunk in resume_file.chunks():
                            f.write(chunk)
                    print("Resume saved at:", resume_path)
                    resume_data=fapp_processor.extract_resume_text(resume_path)
                    resumebased_interview_obj = resumebased_interview.objects.create(interviewee=Intervieweeobj, resume_json=resume_data, job_description=jd)
                    
            
            return JsonResponse({'redirect_url': '/test-audio/'})
        except Exception as e:
            print(f"Error in save_user_info: {str(e)}")
            return JsonResponse({"status": "error", "message": str(e)}, status=500)
            
    return JsonResponse({"status": "error", "message": "Invalid request method"}, status=400)

@csrf_exempt
def test_audio(request):
    try:
        print("test_audio endpoint accessed")
        
        # Handle GET request
        if request.method == "GET":
            print("GET request to test_audio")
            return render(request, 'select_test.html', {"transcript": ""})
        
        # Handle POST request
        if not request.FILES.get("audio"):
            print("No audio file uploaded.")
            if request.headers.get('Accept') == 'application/json':
                return JsonResponse({"transcript": "No audio file uploaded.", "match": False}, status=400)
            else:
                return render(request, 'select_test.html', {"transcript": "No audio file uploaded."})
        
        print("Audio file uploaded.")
        audio_file = request.FILES["audio"]
        audio_file_path = os.path.join(settings.MEDIA_ROOT, "test_audio.wav")
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(audio_file_path), exist_ok=True)
        
        with open(audio_file_path, "wb") as f:
            for chunk in audio_file.chunks():
                f.write(chunk)
        
        if os.path.isfile(audio_file_path):
            print("File exists!")
        else:
            print("File does not exist.")
            if request.headers.get('Accept') == 'application/json':
                return JsonResponse({"error": "Failed to save audio file"}, status=500)
            
        text = fapp_processor.audio_to_text(audio_file_path)
        print(f"Transcribed text: {text}")
        
        if text.lower() == "how is the weather today":
            status = True
        else:
            status = False
        
        # If this is an AJAX request, return JSON
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest' or request.headers.get('Accept') == 'application/json':
            return JsonResponse({
                "transcript": text,
                "match": status
            })
        
        # Otherwise return the HTML response
        return render(request, 'select_test.html', {"transcript": text, "match": status})
    
    except Exception as e:
        print(f"Error in test_audio: {str(e)}")
        if request.headers.get('Accept') == 'application/json':
            return JsonResponse({"error": str(e)}, status=500)
        else:
            return render(request, 'select_test.html', {"transcript": f"Error: {str(e)}", "match": False})

@csrf_exempt
def store_audio_devices(request):
    if request.method == "POST":
        audio_device = request.POST.get("audio_device")
        microphone = request.POST.get("microphone")
        speaker = request.POST.get("speaker")
        print("Audio device:", audio_device)
        print("Microphone:", microphone)
        print("Speaker:", speaker)
        microphone_label = request.POST.get("microphone_label")
        speaker_label = request.POST.get("speaker_label")
        print("Microphone label:", microphone_label)
        print("Speaker label:", speaker_label)
        # Store the audio settings in the session
        request.session["audio_device"] = audio_device
        request.session["microphone"] = microphone
        request.session["speaker"] = speaker
        
        return JsonResponse({"status": "success"})
    
    return JsonResponse({"status": "error", "message": "Invalid request method"}, status=400)

@csrf_exempt
def index(request):
    # if request.method == "POST":
        #pass the audio settings here
        # Generate the audio file using gTTS
        
    audio_file_path = os.path.join(settings.MEDIA_ROOT, "recorded_audi1.mp3")
    if os.path.isfile(audio_file_path):
        print(" File exists!")
    else:
        print(" File does not exist.")
    relative_media_url = os.path.join(settings.MEDIA_URL, "recorded_audi1.mp3")

    audio_file_url = request.build_absolute_uri(relative_media_url)
    
        
    print(audio_file_url)
    return render(request, 'index.html',{"audio_file_url":audio_file_url })


@csrf_exempt
def upload_audio(request):
    if request.method == "POST" and request.FILES.get("audio"):
        audio_file = request.FILES["audio"]
        session_id=request.session.get("session_id", "unknown_session")
        if request.session.get("counter") is None:
            request.session["counter"] = 0
        counter=request.session["counter"]
        MAX_CYCLES= request.session["MAX_CYCLES"]
        print(session_id)
        save_path = os.path.join(settings.MEDIA_ROOT, f"{session_id}_response_{counter}.wav")


        with open(save_path, "wb") as f:
            for chunk in audio_file.chunks():
                f.write(chunk)
        question,audio_file,prev_ans,marks=fapp_processor.audio_processor(save_path,session_id,counter,MAX_CYCLES)
        if counter > 0: 
            try:
                prev_response = InterviewResponse.objects.get(
                    session_id=session_id,
                    question_number=counter
                )
                prev_response.answer_text = prev_ans
                prev_response.score = marks
                prev_response.save()
            except InterviewResponse.DoesNotExist:
                # Handle the case if the record doesn't exist
                pass
        
        # Create a new entry for the current question
        InterviewResponse.objects.create(
            session_id=session_id,
            question_number=counter + 1,
            question_text=question
        )
        show_text = f"{counter+1}. {question}"
        # print("Audio file URL:", audio_file_url)
        if counter >= MAX_CYCLES :
            show_text="Thank you for your time!"
            audio_file="last.mp3"
        relative_media_url = os.path.join(settings.MEDIA_URL, audio_file)
        audio_file_url = request.build_absolute_uri(relative_media_url)
        # audio_file_url = unquote(audio_file_url)
        

        response_data = {
            "audio_url":audio_file_url,
            "text": show_text,
            "button_text": "Record" if counter < MAX_CYCLES  else "Finish",
            "is_last": counter >= MAX_CYCLES - 1, 
            "session_id": session_id,
        }
        
        request.session["counter"]=counter+1
        return JsonResponse(response_data)

    return JsonResponse({"error": "Invalid request"}, status=400)
    
@csrf_exempt
def result(request):

    serial=request.POST.get("serial")
    print("serial is "+serial)
    request.session.flush()
    df_results=fapp_processor.get_results_from_db(serial)
    request.session['result_data'] = df_results
    print("Result data brfore:", df_results)
    return JsonResponse({'redirect_url': '/show-result/'})

def show_result(request):
    result_data = request.session.get('result_data', [])
    print("Result data:", result_data)
    
    return render(request, 'results.html', {"table_data": result_data})