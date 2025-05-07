from django.urls import path

from . import views
from django.contrib import admin
from django.conf import settings
from django.conf.urls.static import static
urlpatterns = [
    path('admin/', admin.site.urls),
    path("", views.get_skills, name="get_skills"),
    path("save-user-info/", views.save_user_info, name="save_user_info"),
    path("store-audio-devices/", views.store_audio_devices, name="store_audio_devices"),
    # path('audio-settings/', views.audio_settings, name='audio_settings'),
    path("test-audio/", views.test_audio, name="test_audio"),
    path("interview/", views.index, name="index"),
    path("upload-audio/", views.upload_audio, name="uploadaudio"),
    path("result/", views.result, name="result"),
    path("show-result/", views.show_result, name="show_result"),
]

if settings.DEBUG:  # Only serve media in development mode
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)