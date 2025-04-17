from django.contrib import admin
from .models import InterviewResponse, IntervieweeDetails, IntervieweeSkill

class InterviewResponseAdmin(admin.ModelAdmin):
    list_display = ('session_id', 'question_number', 'question_text', 'answer_text', 'score')
    list_filter = ('session_id',)
    search_fields = ('session_id', 'question_text', 'answer_text')

admin.site.register(InterviewResponse, InterviewResponseAdmin)
admin.site.register(IntervieweeDetails)
admin.site.register(IntervieweeSkill)