from django.db import models

class InterviewResponse(models.Model):
    # Session information
    session_id = models.CharField(max_length=10)
    created_at = models.DateTimeField(auto_now_add=True)
    
    # Question and answer information
    question_number = models.IntegerField()
    question_text = models.TextField(null=True, blank=True)
    
    
    answer_text = models.TextField(null=True, blank=True)
   
    
    # Score information
    score = models.FloatField(null=True, blank=True)
    
    class Meta:
        app_label = 'fakeapp'
        # This ensures we can identify unique question/session combinations
        unique_together = ['session_id', 'question_number']
        ordering = ['session_id', 'question_number']
    
    def __str__(self):
        return f"Session {self.session_id} - Q{self.question_number}"
class IntervieweeDetails(models.Model):
    # Basic information
    name = models.CharField(max_length=100)
    session_id = models.CharField(max_length=10, unique=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    # Skills (using a separate model for multiple skills)
    
    # Interview statistics
    question_count = models.IntegerField(default=0)
    average_score = models.FloatField(null=True, blank=True)
    
    class Meta:
        app_label = 'fakeapp'
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.name} - Session {self.session_id}"


class IntervieweeSkill(models.Model):
    interviewee = models.ForeignKey(IntervieweeDetails, on_delete=models.CASCADE, related_name='skills')
    skill_name = models.CharField(max_length=100)
    
    class Meta:
        app_label = 'fakeapp'
        unique_together = ['interviewee', 'skill_name']
    
    def __str__(self):
        return f"{self.interviewee.name} - {self.skill_name}"