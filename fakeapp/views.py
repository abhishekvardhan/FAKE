
import os
from django.shortcuts import render
from gtts import gTTS
from django.conf import settings
from pygame import mixer
from django.views.decorators.csrf import csrf_exempt 
from django.http import JsonResponse
import random
from . import fapp_processor

from urllib.parse import unquote

def index(request):
    
    # audio_file_path = "recorded_audi1.mp3"
    # print(audio_file_path)
 
    audio_file_path = os.path.join(settings.MEDIA_ROOT, "recorded_audi1.mp3")
    if os.path.isfile(audio_file_path):
        print("✅ File exists!")
    else:
        print("❌ File does not exist.")
    relative_media_url = os.path.join(settings.MEDIA_URL, "recorded_audi1.mp3")

    audio_file_url = request.build_absolute_uri(relative_media_url)
    
    request.session["session_id"] = str(random.randint(1000, 9999))
  
    print(audio_file_url)
    return render(request, 'index.html',{"audio_file_url":audio_file_url })

counter = 0  # Ensure counter is defined
MAX_CYCLES = 5 
@csrf_exempt


def upload_audio(request):
    global counter
    if request.method == "POST" and request.FILES.get("audio"):
        audio_file = request.FILES["audio"]
        serial=request.session.get("session_id", "unknown_session")
        print(serial)
        save_path = os.path.join(settings.MEDIA_ROOT, f"{serial}_response_{counter}.wav")

        # Save uploaded file
        with open(save_path, "wb") as f:
            for chunk in audio_file.chunks():
                f.write(chunk)
        text,audio_file=fapp_processor.audio_processor(save_path,serial,counter)
        # Simulate processing and return new audio + text
        # print(audio_file)
        # decoded_audio_file = unquote(audio_file)
        # print(os.getcwd())
        # audio_file_path = os.path.join(os.getcwd(), audio_file)
        # normalized_audio_file = decoded_audio_file.replace("\\", os.sep).replace("/", os.sep)
        # audio_file_path = os.path.join(os.getcwd(), normalized_audio_file)
        # print(f"final file path: {audio_file_path}")
        print("Audio file path:", audio_file)
        # audio_file_path = os.path.join(settings.MEDIA_ROOT, audio_file)
        # print(audio_file_path)
        relative_media_url = os.path.join(settings.MEDIA_URL, audio_file)
        audio_file_url = request.build_absolute_uri(relative_media_url)
        

        print("Audio file URL:", audio_file_url)
        # audio_file_url = unquote(audio_file_url)
        response_data = {
            "audio_url":audio_file_url,
            "text": f"{counter+1}. {text}",
            "button_text": "Record" if counter < MAX_CYCLES - 1 else "Finish",
            "is_last": counter >= MAX_CYCLES - 1
        }
        
        counter += 1
        return JsonResponse(response_data)

    return JsonResponse({"error": "Invalid request"}, status=400)
