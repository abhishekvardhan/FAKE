from django.urls import path
from .views import index
from . import views
from django.contrib import admin
from django.conf import settings
from django.conf.urls.static import static
urlpatterns = [
    path('admin/', admin.site.urls),
    path("", index, name="index"),
    path("upload-audio/", views.upload_audio, name="uploadaudio"),
]

if settings.DEBUG:  # Only serve media in development mode
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)