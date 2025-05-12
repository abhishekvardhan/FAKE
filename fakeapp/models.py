from django.db import models

class InterviewResponse(models.Model):
    # Session information
    session_id = models.CharField(max_length=10)
    created_at = models.DateTimeField(auto_now_add=True)
    
    # Question and answer information
    question_number = models.IntegerField()
    question_text = models.TextField(null=True, blank=True)
    expected_answer = models.TextField(null=True, blank=True)
    feedback = models.TextField(null=True, blank=True)
    
    answer_text = models.TextField(null=True, blank=True)
    # Score information
    score = models.FloatField(null=True, blank=True)
    
    class Meta:
        app_label = 'fakeapp'
        # This ensures we can identify unique question/session combinations
        unique_together = ['session_id', 'question_number']
        ordering = ['session_id', 'question_number']
    
class IntervieweeDetails(models.Model):
    # Basic information
    name = models.CharField(max_length=100)
    session_id = models.CharField(max_length=10, unique=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        app_label = 'fakeapp'
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.name} - {self.session_id}"

class skillbased_interview(models.Model):
    interviewee = models.ForeignKey(IntervieweeDetails, on_delete=models.CASCADE, related_name='skills')
    question_count = models.IntegerField(default=0)
    average_score = models.FloatField(null=True, blank=True)
    
    def __str__(self):
        return f"Skill-based interview for {self.interviewee.name}"

class resumebased_interview(models.Model):
    interviewee = models.ForeignKey(IntervieweeDetails, on_delete=models.CASCADE, related_name='resumes')
    resume_json = models.TextField(null=True, blank=True)
    job_description = models.TextField(null=True, blank=True)
    job_description_json = models.TextField(null=True, blank=True)
    final_score = models.FloatField(null=True, blank=True)
    technical_score = models.FloatField(null=True, blank=True)
    behavioral_score = models.FloatField(null=True, blank=True)
    project_score = models.FloatField(null=True, blank=True)
    scenario_score = models.FloatField(null=True, blank=True)
    strengths = models.TextField(null=True, blank=True)
    weaknesses = models.TextField(null=True, blank=True)
    interview_date = models.DateTimeField(auto_now_add=True)
    improvement = models.TextField(null=True, blank=True)
class IntervieweeSkill(models.Model):
    # Changed from skillbased_interview to IntervieweeDetails
    interviewee = models.ForeignKey(IntervieweeDetails, on_delete=models.CASCADE, related_name='interviewee_skills')
    skill_name = models.CharField(max_length=100)
    
    class Meta:
        app_label = 'fakeapp'
        unique_together = ['interviewee', 'skill_name']
    
    def __str__(self):
        return f"{self.interviewee.name} - {self.skill_name}"