from typing import Dict, List, Annotated, TypedDict, Tuple, Optional
import os
import json
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

import langgraph as lg
from langgraph.graph import StateGraph, END
from IPython.display import Image, display
import key_value
from langchain_groq import ChatGroq
from langchain_core.caches import BaseCache
import pandas as pd
ChatGroq.model_rebuild()
os.environ["GROQ_API_KEY"] =key_value.creds_data["groq"]
# Define State Types
class ResumeData(TypedDict):
    skills: Dict[str, str]  # skill -> proficiency level
    projects: List[Dict]
    experience: List[Dict]
    education: List[Dict]

class JobData(TypedDict):
    top_skills: List[str]
    preferred_skills: List[str]
    roles_responsibilities: List[str]
    behavioral_requirements: List[str]
    domain: str

class MatchAnalysis(TypedDict):
    matching_skills: List[str]
    missing_skills: List[str]
    skill_match_percentage: float
    experience_match: str  # Description of experience alignment

class InterviewQuestion(TypedDict):
    question: str
    context: str  # What aspect of the job/resume this relates to
    expected_details: List[str]  # What details to look for in a good answer

class CandidateResponse(TypedDict):
    question_id: int
    question_type: str
    question: str
    response_text: str

class Assessment(TypedDict):
    technical_score: int  # 0-100
    behavioral_score: int  # 0-100
    overall_score: int     # 0-100
    strengths: List[str]
    weaknesses: List[str]
    improvements: List[str]

class InterviewState(TypedDict):
    resume: str
    job_description: str
    company_name: str
    resume_data: Optional[ResumeData]
    job_data: Optional[JobData]
    match_analysis: Optional[MatchAnalysis]
    technical_questions: List[InterviewQuestion]
    project_questions: List[InterviewQuestion]
    scenario_questions: List[InterviewQuestion]
    behavioral_questions: List[InterviewQuestion]
    current_question_index: int
    current_question_type: str
    responses: List[CandidateResponse]
    assessment: Optional[Assessment]
    conversation_history: List[Dict]
    current_step: str

# Initialize LLM
llm=ChatGroq(model_name="llama3-70b-8192")

# Function nodes for the graph
def initialize_state(state: InterviewState) -> InterviewState:
    """Initialize the interview state with the provided inputs."""

    return {
        **state,
        "resume_data": None,
        "job_data": None,
        "match_analysis": None,
        "technical_questions": [],
        "project_questions": [],
        "scenario_questions": [],
        "behavioral_questions": [],
        "current_question_index": 0,
        "current_question_type": "none",
        "responses": [],
        "assessment": None,
        "conversation_history": [],
        "current_step": "extract_resume_data"
    }

def extract_resume_data(state: InterviewState) -> InterviewState:
    """Extract structured data from resume."""
    resume = state["resume"]
    
    
    prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert in parsing resumes for job interviews. 
    Extract the following information from the given resume and format it as a valid JSON object.
    
    The JSON structure must follow this exact format:
    {{
        "skills": {{
            "<skill_name>": "<proficiency_level>",
            ... (additional skills)
        }},
        "projects": [
            {{
                "title": "<project_title>",
                "description": "<project_description>",
                "technologies": ["<tech1>", "<tech2>", ...],
                "outcomes": ["<outcome1>", "<outcome2>", ...]
            }},
            ... (additional projects)
        ],
        "experience": [
            {{
                "company": "<company_name>",
                "role": "<job_title>",
                "duration": "<time_period>",
                "responsibilities": ["<responsibility1>", "<responsibility2>", ...]
            }},
            ... (additional experiences)
        ],
        "education": {{
            "degree": "<degree_name>",
            "institution": "<school_name>",
            "graduationDate": "<graduation_year>"
        }}
    }}
    
    EXTREMELY IMPORTANT INSTRUCTIONS:
    1. Return ONLY THE RAW JSON with no prefixes, explanations, or markdown formatting
    2. Do not include phrases like "Here is the extracted information" or similar text
    3. Do not wrap the JSON in code blocks or backticks
    4. The output must start with {{ and end with }} with no other characters before or after
    5. Every quotation mark must be properly escaped within the JSON
    6. Verify the JSON is complete and valid before returning
    """),
    ("user", "{resume}")
    ])

    
    parser = JsonOutputParser(pydantic_object=ResumeData)
    chain = prompt | llm | parser
    
    try:
      
        resume_data = chain.invoke({"resume": resume})

        return {**state, "resume_data": resume_data, "current_step": "extract_job_data"}
    except Exception as e:
        # Handle parsing errors gracefully
        print(f"Error extracting resume data: {e}")
        default_data = {
            "skills": {},
            "projects": [],
            "experience": [],
            "education": []
        }
        return {**state, "resume_data": default_data, "current_step": "extract_job_data"}

def extract_job_data(state: InterviewState) -> InterviewState:
    """Extract structured data from job description."""
    job_description = state["job_description"]
    company_name = state["company_name"]
    
    # Create a simpler prompt that's less likely to have formatting issues
    prompt = f"""Analyze the job description and extract the following information in valid JSON format:

            1. top_skills: Split into technical_skills array and soft_skills array
            2. preferred_skills: Array of preferred skills
            3. roles_responsibilities: Array of key responsibilities
            4. behavioral_requirements: Array of behavioral traits needed
            5. domain: The industry domain or focus area

            Job Description:
            {job_description}

            Company Name: {company_name}

            Return ONLY valid JSON with this exact structure:
            {{
            "top_skills": {{
                "technical_skills": ["skill1", "skill2"],
                "soft_skills": ["skill1", "skill2"]
            }},
            "preferred_skills": ["skill1", "skill2"],
            "roles_responsibilities": ["resp1", "resp2"],
            "behavioral_requirements": ["trait1", "trait2"],
            "domain": "domain name"
            }}
            """
    
    try:
        print("Parsing job description...")
        # Use the LLM directly to get raw text response
        raw_response = llm.invoke(prompt)
        
        # Extract content from the response
        response_text = raw_response.content
        
        # Try to parse the JSON
        try:
            # Remove any text before the first '{' and after the last '}'
            clean_json_str = response_text[response_text.find('{'):response_text.rfind('}')+1]
            job_data = json.loads(clean_json_str)
        except json.JSONDecodeError as e:
            print(f"Initial JSON parsing failed: {e}")
            
            # Attempt to fix the JSON using the LLM
            fix_prompt = f"""The following text is supposed to be a JSON object, but it may have formatting errors.
                Please fix any JSON syntax errors and return ONLY the corrected valid JSON.

                Original text:
                {response_text}

                Return ONLY the fixed JSON object with no explanations"""
            try:
                fix_response = llm.invoke(fix_prompt)
                fixed_text = fix_response.content
                
                # Find JSON-like content in the response
                clean_json_str = fixed_text[fixed_text.find('{'):fixed_text.rfind('}')+1]
                job_data = json.loads(clean_json_str)
                print("Successfully fixed malformed JSON")
            except Exception as fix_e:
                print(f"Failed to fix JSON: {fix_e}")
                # Use default data
                job_data = {
                    "top_skills": {"technical_skills": ["Python", "Data processing"], "soft_skills": ["Communication", "Problem-solving"]},
                    "preferred_skills": ["Cloud platforms", "Database management"],
                    "roles_responsibilities": ["Code development", "System optimization"],
                    "behavioral_requirements": ["Team collaboration", "Learning mindset"],
                    "domain": "Technology"
                }
        
        # Make sure all required keys are present
        if "top_skills" not in job_data:
            job_data["top_skills"] = {"technical_skills": [], "soft_skills": []}
        elif not isinstance(job_data["top_skills"], dict):
            job_data["top_skills"] = {"technical_skills": job_data["top_skills"] if isinstance(job_data["top_skills"], list) else [], 
                                      "soft_skills": []}
        elif "technical_skills" not in job_data["top_skills"]:
            job_data["top_skills"]["technical_skills"] = []
        elif "soft_skills" not in job_data["top_skills"]:
            job_data["top_skills"]["soft_skills"] = []
        
        for key in ["preferred_skills", "roles_responsibilities", "behavioral_requirements"]:
            if key not in job_data:
                job_data[key] = []
            
        if "domain" not in job_data:
            job_data["domain"] = "Not specified"
            
        print("Job data extraction successful")
        return {**state, "job_data": job_data, "current_step": "perform_match_analysis"}
    except Exception as e:
        print(f"Error extracting job data: {e}")
        default_data = {
            "top_skills": {"technical_skills": ["Python", "Data processing"], "soft_skills": ["Communication", "Problem-solving"]},
            "preferred_skills": ["Cloud platforms", "Database management"],
            "roles_responsibilities": ["Code development", "System optimization"],
            "behavioral_requirements": ["Team collaboration", "Learning mindset"],
            "domain": "Technology"
        }
        return {**state, "job_data": default_data, "current_step": "perform_match_analysis"}

def perform_match_analysis(state: InterviewState) -> InterviewState:
    """Analyze the match between resume and job requirements."""
    resume_data = state.get("resume_data", {})
    job_data = state.get("job_data", {})
    
    # Ensure we have the necessary data structures
    if not resume_data:
        resume_data = {"skills": {}, "projects": [], "experience": [], "education": {}}
        
    if not job_data:
        job_data = {
            "top_skills": {"technical_skills": [], "soft_skills": []},
            "preferred_skills": [],
            "roles_responsibilities": [],
            "domain": "Technology"
        }
    
    # Create a simpler prompt that's less likely to have formatting issues
    prompt = f"""Compare the candidate's resume with the job requirements and analyze the match.

        Resume data:
        {json.dumps(resume_data, indent=2)}

        Job data:
        {json.dumps(job_data, indent=2)}

        Provide an analysis of the match in valid JSON format with these fields:
        1. matching_skills: Array of skills that appear in both resume and job
        2. missing_skills: Array of job-required skills not found in the resume
        3. skill_match_percentage: Number from 0-100 representing match percentage
        4. experience_match: Object with "rating" (excellent/good/fair/poor) and "explanation"

        Return ONLY valid JSON with exactly this structure:
        {{
        "matching_skills": ["skill1", "skill2"],
        "missing_skills": ["skill1", "skill2"],
        "skill_match_percentage": 75,
        "experience_match": {{
            "rating": "good",
            "explanation": "The candidate has relevant experience in X"
        }}
        }}
        """
    
    try:
        print("Performing match analysis...")
        # Get raw LLM response
        raw_response = llm.invoke(prompt)
        response_text = raw_response.content
        
        # Try to parse the JSON
        try:
            # Remove any text before the first '{' and after the last '}'
            clean_json_str = response_text[response_text.find('{'):response_text.rfind('}')+1]
            match_analysis = json.loads(clean_json_str)
        except json.JSONDecodeError as e:
            print(f"Initial JSON parsing failed: {e}")
            
            # Attempt to fix the JSON using the LLM
            fix_prompt = f"""The following text is supposed to be a JSON object, but it may have formatting errors.
Please fix any JSON syntax errors and return ONLY the corrected valid JSON.

Original text:
{response_text}

Return ONLY the fixed JSON object with no explanations:"""
            
            try:
                fix_response = llm.invoke(fix_prompt)
                fixed_text = fix_response.content
                
                # Find JSON-like content in the response
                clean_json_str = fixed_text[fixed_text.find('{'):fixed_text.rfind('}')+1]
                match_analysis = json.loads(clean_json_str)
                print("Successfully fixed malformed JSON")
            except Exception as fix_e:
                print(f"Failed to fix JSON: {fix_e}")
                # Use default data
                match_analysis = {
                    "matching_skills": ["Python", "Communication"],
                    "missing_skills": ["Data processing frameworks"],
                    "skill_match_percentage": 60.0,
                    "experience_match": {
                        "rating": "fair",
                        "explanation": "Candidate has some relevant experience but lacks specific domain expertise."
                    }
                }
        
        # Make sure all required keys are present
        if "matching_skills" not in match_analysis:
            match_analysis["matching_skills"] = []
            
        if "missing_skills" not in match_analysis:
            match_analysis["missing_skills"] = []
            
        if "skill_match_percentage" not in match_analysis:
            match_analysis["skill_match_percentage"] = 0
            
        if "experience_match" not in match_analysis:
            match_analysis["experience_match"] = {
                "rating": "fair",
                "explanation": "Unable to determine exact experience match."
            }
        elif isinstance(match_analysis["experience_match"], str):
            match_analysis["experience_match"] = {
                "rating": "fair",
                "explanation": match_analysis["experience_match"]
            }
        
        # Ensure the experience_match object has the required fields
        if isinstance(match_analysis["experience_match"], dict):
            if "rating" not in match_analysis["experience_match"]:
                match_analysis["experience_match"]["rating"] = "fair"
            if "explanation" not in match_analysis["experience_match"]:
                match_analysis["experience_match"]["explanation"] = "No detailed explanation provided."
        
        print("Match analysis completed")
        return {**state, "match_analysis": match_analysis, "current_step": "generate_technical_questions"}
    except Exception as e:
        print(f"Error performing match analysis: {e}")
        default_analysis = {
            "matching_skills": ["Python", "Communication"],
            "missing_skills": ["Data processing frameworks"],
            "skill_match_percentage": 60.0,
            "experience_match": {
                "rating": "fair",
                "explanation": "Candidate has some relevant experience but lacks specific domain expertise."
            }
        }
        return {**state, "match_analysis": default_analysis, "current_step": "generate_technical_questions"}

def generate_technical_questions(state: InterviewState) -> InterviewState:
    """Generate a single technical interview question based on matching skills and previous responses."""
    resume_data = state["resume_data"]
    job_data = state["job_data"]
    match_analysis = state["match_analysis"]
    current_question_index = state.get("current_question_index", 0)
    technical_questions = state.get("technical_questions", [])
    
    # Get previous responses for technical questions
    technical_responses = [
        resp for resp in state.get("responses", []) 
        if resp.get("question_type") == "technical"
    ]
    
    # If we already have 4 technical questions, don't generate more
    if len(technical_questions) >= 4:
        return state
    
    print(f"Generating technical question #{current_question_index + 1}...")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert technical interviewer.
        Based on the candidate's matching skills, job requirements, and previous responses,
        create ONE in-depth technical question that will assess the candidate's expertise.
        
        For the question:
        1. Make it specific to the required skills
        2. Ensure it tests both theoretical knowledge and practical application
        3. Include context on what skill/requirement this relates to
        4. Include expected details that would be present in a good answer
        
        If there are previous responses, adapt this question to build upon those responses.
        If the candidate showed strength in a specific area, you can ask more challenging
        questions in that area. If they showed weakness, ask about a different area.
        
        Output as a single JSON object with 'question', 'context', and 'expected_details' fields."""),
        ("user", """Resume skills: {resume_skills}
        Job top skills: {job_top_skills}
        Matching skills: {matching_skills}
        Missing skills: {missing_skills}
        Question number: {question_number}
        Previous responses: {previous_responses}""")
    ])
    
    chain = prompt | llm | JsonOutputParser()
    
    try:
        question = chain.invoke({
            "resume_skills": resume_data["skills"],
            "job_top_skills": job_data["top_skills"],
            "matching_skills": match_analysis["matching_skills"],
            "missing_skills": match_analysis["missing_skills"],
            "question_number": current_question_index + 1,
            "previous_responses": technical_responses
        })
        
        # Add the new question to our list
        technical_questions.append(question)
        
        return {
            **state, 
            "technical_questions": technical_questions
        }
    except Exception as e:
        print(f"Error generating technical question: {e}")
        default_question = {
            "question": f"Based on your experience, how would you implement a solution for processing large datasets efficiently?",
            "context": "Data processing capabilities",
            "expected_details": ["Algorithm selection", "Scalability considerations", "Performance optimization"]
        }
        
        technical_questions.append(default_question)
        
        return {
            **state, 
            "technical_questions": technical_questions
        }

def generate_project_questions(state: InterviewState) -> InterviewState:
    """Generate questions about the candidate's projects and work experience."""
    resume_data = state.get("resume_data", {})
    job_data = state.get("job_data", {})
    
    # Ensure we have the necessary data structures
    if not resume_data:
        resume_data = {"projects": [], "experience": []}
        
    if not job_data:
        job_data = {"roles_responsibilities": []}
    
    # Make sure all required keys exist
    projects = resume_data.get("projects", [])
    experience = resume_data.get("experience", [])
    responsibilities = job_data.get("roles_responsibilities", [])
    
    # Create default questions in case of failure
    default_questions = [
        {
            "question": "Please describe your most challenging project and how you approached it.",
            "context": "Project experience assessment",
            "expected_details": ["Problem definition", "Approach", "Outcome", "Lessons learned"]
        },
        {
            "question": "How did you handle a situation where project requirements changed significantly?",
            "context": "Adaptability assessment",
            "expected_details": ["Flexibility", "Communication", "Problem-solving"]
        },
        {
            "question": "Tell me about a time you collaborated with others on a technical project.",
            "context": "Team collaboration",
            "expected_details": ["Role in team", "Communication", "Conflict resolution"]
        }
    ]
    
    try:
        print("Generating project questions...")
        
        # Generate one question at a time for better reliability
        project_questions = []
        
        # Topics for questions to ensure variety
        topics = [
            "challenging project",
            "team collaboration",
            "technical problem solving"
        ]
        
        for i, topic in enumerate(topics):
            # Create a focused prompt for just one question
            prompt = f"""Create ONE interview question about the candidate's experience with {topic}.
The question should help assess their technical skills and experience.

Resume info:
Projects: {str(projects)[:500]}
Experience: {str(experience)[:500]}

Return ONLY a valid JSON object with exactly these fields:
{{
  "question": "The complete interview question text",
  "context": "Brief context about what this question assesses",
  "expected_details": ["Detail 1", "Detail 2", "Detail 3"]
}}
"""
            try:
                # Get response
                response = llm.invoke(prompt)
                response_text = response.content
                
                # Clean and parse the JSON
                try:
                    # Extract JSON object from response
                    clean_json_str = response_text[response_text.find('{'):response_text.rfind('}')+1]
                    question_obj = json.loads(clean_json_str)
                    
                    # Validate the question object
                    if "question" in question_obj:
                        if "context" not in question_obj:
                            question_obj["context"] = f"{topic.capitalize()} assessment"
                        if "expected_details" not in question_obj or not question_obj["expected_details"]:
                            question_obj["expected_details"] = ["Approach", "Technical skills", "Results"]
                            
                        project_questions.append(question_obj)
                    else:
                        project_questions.append(default_questions[i])
                except json.JSONDecodeError:
                    # If JSON parsing fails, use the default question for this topic
                    project_questions.append(default_questions[i])
            except Exception as e:
                print(f"Error generating question about {topic}: {e}")
                project_questions.append(default_questions[i])
        
        # Ensure we have exactly 3 questions
        while len(project_questions) < 3:
            project_questions.append(default_questions[len(project_questions) % len(default_questions)])
            
        print(f"Successfully generated {len(project_questions)} project questions")
        return {**state, "project_questions": project_questions, "current_step": "generate_scenario_questions"}
    except Exception as e:
        print(f"Error generating project questions: {e}")
        return {**state, "project_questions": default_questions, "current_step": "generate_scenario_questions"}

def generate_scenario_questions(state: InterviewState) -> InterviewState:
    """Generate scenario-based questions related to the job responsibilities."""
    job_data = state.get("job_data", {})
    match_analysis = state.get("match_analysis", {})
    
    # Ensure we have the necessary data structures
    if not job_data:
        job_data = {"roles_responsibilities": [], "domain": "Technology"}
        
    if not match_analysis:
        match_analysis = {"experience_match": "Not evaluated"}
    
    # Make sure all required keys exist
    responsibilities = job_data.get("roles_responsibilities", [])
    domain = job_data.get("domain", "Technology")
    experience_match = match_analysis.get("experience_match", "Not evaluated")
    
    # Create default questions in case of failure
    default_questions = [
        {
            "question": "Imagine you're facing a tight deadline with competing priorities. How would you approach this situation?",
            "context": "Time management and prioritization",
            "expected_details": ["Prioritization method", "Communication", "Delegation", "Stress management"]
        },
        {
            "question": "Describe how you would handle a situation where a critical system you're responsible for fails in production.",
            "context": "Technical problem-solving",
            "expected_details": ["Troubleshooting approach", "Communication", "Root cause analysis", "Prevention"]
        }
    ]
    
    try:
        print("Generating scenario questions...")
        
        # Create a simple prompt that's less likely to have formatting issues
        prompt = f"""Create 2 scenario-based interview questions that assess how the candidate would handle realistic situations in this job role.

Job responsibilities: {str(responsibilities)}
Job domain: {str(domain)}
Candidate's experience level: {str(experience_match)}

Format each question as a JSON object with these exact keys: "question", "context", and "expected_details".
Return an array of these questions in valid JSON format with no additional text or explanation.
"""

        # Get raw completion from LLM
        raw_response = llm.invoke(prompt)
        response_text = raw_response.content
        
        # Try to extract a JSON array first
        questions = []
        try:
            # Look for array pattern [...]
            start_idx = response_text.find('[')
            end_idx = response_text.rfind(']') + 1
            
            if start_idx >= 0 and end_idx > 0:
                json_str = response_text[start_idx:end_idx]
                questions = json.loads(json_str)
            else:
                # If no array found, try to parse individual JSON objects
                pattern = r'\{[^{}]*\}'
                import re
                json_objects = re.findall(pattern, response_text)
                
                for obj_str in json_objects:
                    try:
                        question = json.loads(obj_str)
                        if "question" in question:
                            questions.append(question)
                    except:
                        pass
        except Exception as e:
            print(f"Error parsing scenario questions JSON: {e}")
            # Fall back to default questions
            questions = default_questions
        
        # Validate and fix each question
        scenario_questions = []
        for q in questions:
            if isinstance(q, dict) and "question" in q:
                if "context" not in q:
                    q["context"] = "Scenario assessment"
                if "expected_details" not in q:
                    q["expected_details"] = ["Problem-solving", "Decision-making", "Communication"]
                scenario_questions.append(q)
        
        # If we couldn't parse any valid questions, use defaults
        if not scenario_questions:
            scenario_questions = default_questions
            
        # Ensure we have exactly 2 questions
        while len(scenario_questions) < 2:
            scenario_questions.append(default_questions[len(scenario_questions) % len(default_questions)])
        scenario_questions = scenario_questions[:2]  # Limit to 2 questions
            
        return {**state, "scenario_questions": scenario_questions, "current_step": "generate_behavioral_questions"}
    except Exception as e:
        print(f"Error generating scenario questions: {e}")
        return {**state, "scenario_questions": default_questions, "current_step": "generate_behavioral_questions"}

def generate_behavioral_questions(state: InterviewState) -> InterviewState:
    """Generate behavioral questions based on the company and job requirements."""
    job_data = state.get("job_data", {})
    company_name = state.get("company_name", "the company")
    
    # Ensure we have the necessary data structures
    if not job_data:
        job_data = {"behavioral_requirements": [], "domain": "Technology"}
    
    # Make sure all required keys exist
    behavioral_reqs = job_data.get("behavioral_requirements", [])
    domain = job_data.get("domain", "Technology")
    
    # Create default questions in case of failure
    default_questions = [
        {
            "question": "Tell me about a time when you had to adapt to a significant change at work.",
            "context": "Adaptability assessment",
            "expected_details": ["Situation", "Action", "Result", "Learning"]
        },
        {
            "question": "Describe a situation where you had to work with someone who was difficult to get along with.",
            "context": "Interpersonal skills",
            "expected_details": ["Conflict resolution", "Communication", "Empathy", "Outcome"]
        }
    ]
    
    try:
        print("Generating behavioral questions...")
        
        # Create a simple prompt that's less likely to have formatting issues
        prompt = f"""Create 2 behavioral interview questions that assess the candidate's soft skills and cultural fit.

Company name: {company_name}
Job behavioral requirements: {str(behavioral_reqs)}
Job domain: {str(domain)}

Format each question as a JSON object with these exact keys: "question", "context", and "expected_details".
Return an array of these questions in valid JSON format with no additional text or explanation.
"""

        # Get raw completion from LLM
        raw_response = llm.invoke(prompt)
        response_text = raw_response.content
        
        # Try to extract a JSON array first
        questions = []
        try:
            # Look for array pattern [...]
            start_idx = response_text.find('[')
            end_idx = response_text.rfind(']') + 1
            
            if start_idx >= 0 and end_idx > 0:
                json_str = response_text[start_idx:end_idx]
                questions = json.loads(json_str)
            else:
                # If no array found, try to parse individual JSON objects
                pattern = r'\{[^{}]*\}'
                import re
                json_objects = re.findall(pattern, response_text)
                
                for obj_str in json_objects:
                    try:
                        question = json.loads(obj_str)
                        if "question" in question:
                            questions.append(question)
                    except:
                        pass
        except Exception as e:
            print(f"Error parsing behavioral questions JSON: {e}")
            # Fall back to default questions
            questions = default_questions
        
        # Validate and fix each question
        behavioral_questions = []
        for q in questions:
            if isinstance(q, dict) and "question" in q:
                if "context" not in q:
                    q["context"] = "Behavioral assessment"
                if "expected_details" not in q:
                    q["expected_details"] = ["Situation", "Action", "Result", "Reflection"]
                behavioral_questions.append(q)
        
        # If we couldn't parse any valid questions, use defaults
        if not behavioral_questions:
            behavioral_questions = default_questions
            
        # Ensure we have exactly 2 questions
        while len(behavioral_questions) < 2:
            behavioral_questions.append(default_questions[len(behavioral_questions) % len(default_questions)])
        behavioral_questions = behavioral_questions[:2]  # Limit to 2 questions
            
        return {
            **state, 
            "behavioral_questions": behavioral_questions, 
            "current_step": "ask_question",
            "current_question_type": "project"  # Start with project questions
        }
    except Exception as e:
        print(f"Error generating behavioral questions: {e}")
        return {
            **state, 
            "behavioral_questions": default_questions, 
            "current_step": "ask_question",
            "current_question_type": "project"  # Start with project questions
        }

def process_response(state: InterviewState, response: str) -> InterviewState:
    """Process user response and update state"""
    if not state["conversation_history"]:
        return state
        
    last_question = state["conversation_history"][-1]["content"]
    
    candidate_response: CandidateResponse = {
        "question_id": len(state["responses"]),
        "question_type": state["current_question_type"],
        "question": last_question,
        "response_text": response
    }
    
    updated_state = {
        **state,
        "responses": state["responses"] + [candidate_response],
        "conversation_history": state["conversation_history"] + [
            {"role": "user", "content": response}
        ]
    }
    
    # For technical questions, increment the index after storing the response
    if state["current_question_type"] == "technical":
        updated_state["current_question_index"] = state["current_question_index"] + 1
        
    return updated_state

def ask_question(state: InterviewState) -> Dict:
    """Determine the next question to ask based on current question type and index."""
    question_type = state["current_question_type"]
    index = state["current_question_index"]
    
    # Check if the last message in conversation history is from the assistant
    # This means we already asked a question and are waiting for an answer
    if state.get("conversation_history") and state["conversation_history"][-1]["role"] == "assistant":
        # Don't ask a new question, return the state as is
        return state
    
    if question_type == "project":
        questions = state["project_questions"]
        if index < len(questions):
            question = questions[index]
            # Ask the question and increment the index for next time
            return {
                **state,
                "current_question_index": index + 1,
                "conversation_history": state["conversation_history"] + [
                    {"role": "assistant", "content": question["question"]}
                ]
            }
        else:
            # Move to technical questions
            print("\n--- Moving to Technical Questions ---\n")
            return {
                **state,
                "current_question_type": "technical",
                "current_question_index": 0
            }
    
    elif question_type == "technical":
        questions = state.get("technical_questions", [])
        if index < len(questions):
            question = questions[index]
            # Ask the question (don't increment index here - will be done after response)
            return {
                **state,
                "conversation_history": state["conversation_history"] + [
                    {"role": "assistant", "content": question["question"]}
                ]
            }
        elif index >= 4:  # We've asked all 4 technical questions
            # Move to scenario questions
            print("\n--- Moving to Scenario Questions ---\n")
            return {
                **state,
                "current_question_type": "scenario",
                "current_question_index": 0
            }
        else:
            # Need to generate more questions, but don't change state here
            return state
    
    elif question_type == "scenario":
        questions = state["scenario_questions"]
        if index < len(questions):
            question = questions[index]
            # Ask the question and increment the index for next time
            return {
                **state,
                "current_question_index": index + 1,
                "conversation_history": state["conversation_history"] + [
                    {"role": "assistant", "content": question["question"]}
                ]
            }
        else:
            # Move to behavioral questions
            print("\n--- Moving to Behavioral Questions ---\n")
            return {
                **state,
                "current_question_type": "behavioral",
                "current_question_index": 0
            }
    
    elif question_type == "behavioral":
        questions = state["behavioral_questions"]
        if index < len(questions):
            question = questions[index]
            # Ask the question and increment the index for next time
            return {
                **state,
                "current_question_index": index + 1,
                "conversation_history": state["conversation_history"] + [
                    {"role": "assistant", "content": question["question"]}
                ]
            }
        else:
            # We've asked all questions, move to assessment
            print("\n--- Moving to Final Assessment ---\n")
            return {
                **state,
                "current_step": "perform_assessment"
            }
    
    # Fallback - should never reach here
    return {**state, "current_step": "perform_assessment"}

def perform_assessment(state: InterviewState) -> Dict:
    """Perform the final assessment of the candidate."""
    resume_data = state["resume_data"]
    job_data = state["job_data"]
    match_analysis = state["match_analysis"]
    responses = state["responses"]
    
    # Calculate average scores from the evaluations
    total_technical_score = 0
    technical_count = 0
    total_behavioral_score = 0
    behavioral_count = 0
    all_scores = []
    
    for resp in responses:
        eval_data = resp.get("evaluation", {})
        if eval_data is None:
            eval_data = {}
        score = eval_data.get("score")
        if isinstance(score, (int, float)):
            all_scores.append(score)
            q_type = resp.get("question_type", "")
            if q_type == "technical":
                total_technical_score += score
                technical_count += 1
            elif q_type in ["behavioral", "scenario"]:
                total_behavioral_score += score
                behavioral_count += 1
    
    avg_technical_score = round(total_technical_score / technical_count) if technical_count > 0 else None
    avg_behavioral_score = round(total_behavioral_score / behavioral_count) if behavioral_count > 0 else None
    overall_score = round(sum(all_scores) / len(all_scores)) if all_scores else None
    
    print("performing assessment")
    print(f"Average Technical Score: {avg_technical_score}")
    print(f"Average Behavioral Score: {avg_behavioral_score}")
    print(f"Overall Score: {overall_score}")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert hiring assessor.
        Based on the candidate's resume, job requirements, interview responses, and response scores, provide a comprehensive assessment:
        
        1. technical_score: Use the provided average technical score or assess technical skills (0-100)
        2. behavioral_score: Use the provided average behavioral score or assess soft skills (0-100)
        3. overall_score: Use the provided overall average score or provide a weighted total (0-100)
        4. strengths: list[str] Key positive aspects identified (3-5 points)
        5. weaknesses: list[str] Areas for improvement (3-5 points)
        6. improvements: list[str] Improvement suggestions Specific actionable feedback (3-5 points)
        
        The candidate's technical score average is: {avg_technical_score}
        The candidate's behavioral score average is: {avg_behavioral_score}
        The overall average of all question scores is: {overall_score}
        
        Output as a detailed JSON assessment."""),
        ("user", """Resume data: {resume_data}
        Job data: {job_data}
        Match analysis: {match_analysis}
        Interview responses: {responses}""")
    ])
    
    parser = JsonOutputParser(pydantic_object=Assessment)
    chain = prompt | llm | parser
    
    try:
        assessment = chain.invoke({
            "resume_data": resume_data,
            "job_data": job_data,
            "match_analysis": match_analysis,
            "responses": responses,
            "avg_technical_score": avg_technical_score,
            "avg_behavioral_score": avg_behavioral_score,
            "overall_score": overall_score
        })
        
        # Use calculated scores if they're available and not in assessment
        if avg_technical_score is not None and "technical_score" not in assessment:
            assessment["technical_score"] = avg_technical_score
        if avg_behavioral_score is not None and "behavioral_score" not in assessment:
            assessment["behavioral_score"] = avg_behavioral_score
        if overall_score is not None and "overall_score" not in assessment:
            assessment["overall_score"] = overall_score
        
        # Ensure all required keys are present
        if "improvements" not in assessment:
            assessment["improvements"] = ["Candidate should consider expanding their skills in the missing areas."]
        if "strengths" not in assessment:
            assessment["strengths"] = ["Technical skills aligned with job requirements."]
        if "weaknesses" not in assessment:
            assessment["weaknesses"] = ["Limited information to assess all required areas."]
        
        return {
            **state, 
            "assessment": assessment,
            "current_step": END
        }
    except Exception as e:
        print(f"Error performing assessment: {e}")
        default_assessment = {
            "technical_score": avg_technical_score or 50,
            "behavioral_score": avg_behavioral_score or 50,
            "overall_score": overall_score or 50,
            "strengths": ["Unable to properly assess strengths"],
            "weaknesses": ["Unable to properly assess weaknesses"],
            "improvements": ["Consider a more thorough interview process"]
        }
        return {
            **state, 
            "assessment": default_assessment,
            "current_step": END
        }

# Build the graph
def build_graph():
    """Build and compile the interview process graph."""
    workflow = StateGraph(InterviewState)
    
    # Add nodes
    workflow.add_node("initialize_state", initialize_state)
    workflow.add_node("extract_resume_data", extract_resume_data)
    workflow.add_node("extract_job_data", extract_job_data)
    workflow.add_node("perform_match_analysis", perform_match_analysis)
    workflow.add_node("generate_technical_questions", generate_technical_questions)
    workflow.add_node("generate_project_questions", generate_project_questions)
    workflow.add_node("generate_scenario_questions", generate_scenario_questions)
    workflow.add_node("generate_behavioral_questions", generate_behavioral_questions)
    workflow.add_node("ask_question", ask_question)
    workflow.add_node("perform_assessment", perform_assessment)
    workflow.add_node("process_response", process_response)
    
    # Create a router node to determine next step
    workflow.add_node("route_next_step", route_next_step)
    
    # Set edges
    workflow.add_edge("initialize_state", "extract_resume_data")
    workflow.add_edge("extract_resume_data", "extract_job_data")
    workflow.add_edge("extract_job_data", "perform_match_analysis")
    workflow.add_edge("perform_match_analysis", "generate_project_questions")
    workflow.add_edge("generate_project_questions", "ask_question")
    
    # Edge from generate_technical_questions to ask_question
    workflow.add_edge("generate_technical_questions", "ask_question")
    workflow.add_edge("generate_scenario_questions", "ask_question")
    workflow.add_edge("generate_behavioral_questions", "ask_question")
    
    # From ask_question to route_next_step
    workflow.add_edge("ask_question", "route_next_step")
    
    # Conditional routing from router
    workflow.add_conditional_edges(
        "route_next_step",
        lambda state: decide_next_step(state),
        {
            "process_response": "process_response",
            "assessment": "perform_assessment",
            "wait": "route_next_step"  # Wait for user input
        }
    )
    
    # After processing response, route to next step based on question type and index
    workflow.add_conditional_edges(
        "process_response",
        lambda state: state["current_question_type"] == "technical" and state["current_question_index"] >= 4,
        {
            True: "generate_scenario_questions",  # Finished with technical questions
            False: "ask_question"  # Continue with questions
        }
    )
    
    # Set the entrypoint
    workflow.set_entry_point("initialize_state")
    workflow1 = workflow.compile()
    try:
        print("Graph built successfully.")
        img_data = workflow1.get_graph().draw_mermaid_png(max_retries=5, retry_delay=2.0)
        with open("output.png", "wb") as f:
            f.write(img_data)
    except Exception as e:
        print(f"Error saving image: {e}")
    
    return workflow1

def decide_next_step(state: InterviewState) -> str:
    """Determine the next step in the interview process."""
    # Check if we need to perform assessment
    if state["current_step"] == "perform_assessment":
        return "assessment"
    
    # Check if we've already asked a question and are waiting for response
    if state["conversation_history"] and state["conversation_history"][-1]["role"] == "assistant":
        return "wait"  # Stay in the same state, waiting for user input
    
    # If we have a user response, process it
    if state["conversation_history"] and state["conversation_history"][-1]["role"] == "user":
        return "process_response"
    
    # Default: continue asking questions
    return "wait"

def route_next_step(state: InterviewState) -> Dict:
    """Router node that doesn't modify state."""
    return state

# Example usage
def run_interview(resume: str, job_description: str, company_name: str):
    """Run an interview process with the given inputs."""
    graph = build_graph()
    
    # Initialize the state
    state = {
        "resume": resume,
        "job_description": job_description,
        "company_name": company_name
    }
    setup_steps = [
        "initialize_state",
        "extract_resume_data",
        "extract_job_data",
        "perform_match_analysis",
        "generate_project_questions",  # Technical questions will be generated one by one
        "generate_scenario_questions",
        "generate_behavioral_questions"
    ]
    
    for step in setup_steps:
        state = graph.nodes[step].invoke(state)
    
    # Set initial question type to project
    state["current_question_type"] = "project"
    state["current_question_index"] = 0
    
    # Run the graph with initial state
    final_state = graph.invoke(state)
    
    # Save results to Excel files
    save_results_to_excel(final_state)
    
    # Return the assessment
    return {
        "match_analysis": final_state["match_analysis"],
        "assessment": final_state["assessment"],
        "interview_questions": {
            "project": final_state["project_questions"],
            "technical": final_state["technical_questions"],
            "scenario": final_state["scenario_questions"],
            "behavioral": final_state["behavioral_questions"]
        }
    }

def save_results_to_excel(state: InterviewState):
    """Save interview results to Excel sheets."""
    # Create a directory for saving Excel files if it doesn't exist
    os.makedirs("interview_results", exist_ok=True)
    
    # Save match analysis
    match_df = pd.DataFrame({
        "Matching Skills": pd.Series(state["match_analysis"].get("matching_skills", [])),
        "Missing Skills": pd.Series(state["match_analysis"].get("missing_skills", []))
    })
    match_summary = pd.DataFrame({
        "Metric": ["Skill Match Percentage", "Experience Match"],
        "Value": [
            state["match_analysis"].get("skill_match_percentage", 0),
            state["match_analysis"].get("experience_match", "Not available")
        ]
    })
    
    with pd.ExcelWriter("interview_results/match_analysis.xlsx") as writer:
        match_df.to_excel(writer, sheet_name="Skills Comparison", index=False)
        match_summary.to_excel(writer, sheet_name="Summary", index=False)
    
    # Save interview questions and responses
    all_questions = []
    
    # Add project questions
    for i, q in enumerate(state.get("project_questions", [])):
        all_questions.append({
            "Question Type": "Project",
            "Question": q.get("question", "N/A"),
            "Context": q.get("context", "Not specified"),
            "Expected Details": ", ".join(q.get("expected_details", ["Not specified"]))
        })
    
    # Add technical questions
    for i, q in enumerate(state.get("technical_questions", [])):
        all_questions.append({
            "Question Type": "Technical",
            "Question": q.get("question", "N/A"),
            "Context": q.get("context", "Not specified"),
            "Expected Details": ", ".join(q.get("expected_details", ["Not specified"]))
        })
    
    # Add scenario questions
    for i, q in enumerate(state.get("scenario_questions", [])):
        all_questions.append({
            "Question Type": "Scenario",
            "Question": q.get("question", "N/A"),
            "Context": q.get("context", "Not specified"),
            "Expected Details": ", ".join(q.get("expected_details", ["Not specified"]))
        })
    
    # Add behavioral questions
    for i, q in enumerate(state.get("behavioral_questions", [])):
        all_questions.append({
            "Question Type": "Behavioral",
            "Question": q.get("question", "N/A"),
            "Context": q.get("context", "Not specified"),
            "Expected Details": ", ".join(q.get("expected_details", ["Not specified"]))
        })
    
    # Convert to DataFrame and save
    questions_df = pd.DataFrame(all_questions)
    questions_df.to_excel("interview_results/interview_questions.xlsx", index=False)
    
    # Save responses if any
    if state.get("responses", []):
        responses_data = []
        for i, r in enumerate(state["responses"]):
            eval_data = r.get("evaluation", {})
            if eval_data is None:
                eval_data = {}
            responses_data.append({
                "Question ID": r.get("question_id", i),
                "Question Type": r.get("question_type", "Unknown"),
                "Question": r.get("question", "N/A"),
                "Response": r.get("response_text", "No response"),
                "Score": eval_data.get("score", "N/A"),
                "Feedback": eval_data.get("feedback", "Not evaluated"),
                "Strengths": ", ".join(eval_data.get("strengths", [])),
                "Areas for Improvement": ", ".join(eval_data.get("areas_for_improvement", []))
            })
        
        responses_df = pd.DataFrame(responses_data)
        
        # Calculate average scores by question type
        score_summary = []
        question_types = ["project", "technical", "scenario", "behavioral"]
        
        for q_type in question_types:
            type_responses = [r for r in responses_data if r["Question Type"] == q_type]
            if type_responses:
                scores = [r["Score"] for r in type_responses if r["Score"] != "N/A"]
                if scores:
                    avg_score = sum(scores) / len(scores)
                    score_summary.append({
                        "Question Type": q_type.capitalize(),
                        "Number of Questions": len(type_responses),
                        "Average Score": round(avg_score, 2)
                    })
        
        score_summary_df = pd.DataFrame(score_summary)
        
        with pd.ExcelWriter("interview_results/candidate_responses.xlsx") as writer:
            responses_df.to_excel(writer, sheet_name="Responses", index=False)
            if score_summary:
                score_summary_df.to_excel(writer, sheet_name="Score Summary", index=False)
    
    # Save assessment if available
    if state.get("assessment"):
        assessment = state["assessment"]
        assessment_summary = pd.DataFrame({
            "Metric": ["Technical Score", "Behavioral Score", "Overall Score"],
            "Value": [
                assessment.get("technical_score", 0),
                assessment.get("behavioral_score", 0),
                assessment.get("overall_score", 0)
            ]
        })
        
        feedback_df = pd.DataFrame({
            "Strengths": pd.Series(assessment.get("strengths", ["Not evaluated"])),
            "Weaknesses": pd.Series(assessment.get("weaknesses", ["Not evaluated"])),
            "Improvements": pd.Series(assessment.get("improvements", ["Not provided"]))
        })
        
        with pd.ExcelWriter("interview_results/assessment.xlsx") as writer:
            assessment_summary.to_excel(writer, sheet_name="Scores", index=False)
            feedback_df.to_excel(writer, sheet_name="Feedback", index=False)
        
    print(f"Interview results saved to Excel files in the 'interview_results' directory")

# Example of how to use:
# Resume example
resume = '''
John Smith
Software Engineer

Skills:
- Python (Advanced)
- Java (Intermediate)
- SQL (Advanced)
- Machine Learning (Intermediate)
- Docker (Basic)

Work Experience:
Software Engineer, ABC Tech (2020-Present)
- Developed backend services using Python and Flask
- Optimized database queries reducing response time by 30%
- Led a team of 3 developers for a customer-facing project

Junior Developer, XYZ Corp (2018-2020)
- Maintained Java applications and fixed bugs
- Developed automated testing scripts

Projects:
E-commerce Recommendation Engine
- Built a product recommendation system using collaborative filtering
- Used Python, Pandas, and scikit-learn
- Achieved 25 increase in click-through rate

Inventory Management System
- Developed a full-stack inventory tracking system
- Used Java Spring Boot and React
- Implemented real-time data synchronization

Education:
BS Computer Science, University of Technology (2018)
'''

# Job description example
job_description = '''
Senior Software Engineer - Data Platform
Company: Tech Innovations Inc.

About the Role:
We are seeking a Senior Software Engineer to join our Data Platform team. The ideal candidate will have strong Python skills and experience with data processing frameworks.

Key Responsibilities:
- Design and build scalable data processing pipelines
- Optimize data storage and retrieval mechanisms
- Collaborate with data scientists to implement ML models
- Mentor junior engineers and participate in code reviews

Required Skills:
- Advanced Python programming (5+ years)
- Experience with SQL and NoSQL databases
- Knowledge of data processing frameworks (Spark, Kafka)
- Strong understanding of system design principles
- CI/CD and testing practices

Preferred Skills:
- Experience with cloud platforms (AWS preferred)
- Kubernetes and container orchestration
- Data visualization tools
- ML/AI implementation experience

We value team players with excellent communication skills and problem-solving abilities.
'''

company_name = "Tech Innovations Inc."

# Run the interview
result = run_interview(resume, job_description, company_name)

print(json.dumps(result, indent=2))

def robust_json_parser(text: str) -> dict:
    """A robust JSON parser that attempts multiple methods to extract valid JSON from text."""
    if not text:
        return {}
        
    # Try direct parsing first
    try:
        # Extract what looks like JSON (from first { to last })
        json_text = text[text.find('{'):text.rfind('}')+1]
        return json.loads(json_text)
    except json.JSONDecodeError as e:
        print(f"Direct JSON parsing failed: {e}")
        
        # Try advanced extraction and cleaning
        try:
            import re
            
            # Method 1: Fix common JSON errors
            fixed_text = text
            
            # Replace single quotes with double quotes (only for key-value pairs)
            fixed_text = re.sub(r"([{,]\s*)'([^']+)'(\s*:)", r'\1"\2"\3', fixed_text)
            
            # Fix line breaks and spaces within string values
            fixed_text = re.sub(r'"\s*\n\s*([^"]*)"', r'"\1"', fixed_text)
            
            # Fix missing quotes around string values
            fixed_text = re.sub(r':\s*([a-zA-Z][a-zA-Z0-9_]*)\s*([,}])', r': "\1"\2', fixed_text)
            
            # Fix missing commas between key-value pairs
            fixed_text = re.sub(r'}\s*"', '},\n"', fixed_text)
            fixed_text = re.sub(r'"\s*{', '",\n{', fixed_text)
            fixed_text = re.sub(r'"[,}\]][\r\n\s]*"', '",\n"', fixed_text)
            
            # Insert missing commas between object properties
            fixed_text = re.sub(r'"\s*}\s*"', '"},\n"', fixed_text)
            
            # Fix objects without commas
            fixed_text = re.sub(r'"(\s*)(\w+)(\s*)":(\s*)("[^"]*"|[0-9]+|true|false|null|{[^{}]*}|\[[^\[\]]*\])(\s*)(\w+)(\s*)":',
                              r'"\2": \5,\n"\7":', fixed_text)
            
            # Add missing commas in arrays
            fixed_text = re.sub(r']\s*"', '],\n"', fixed_text)
            fixed_text = re.sub(r'"\s*\[', '",\n[', fixed_text)
            
            # Fix trailing commas
            fixed_text = re.sub(r',(\s*[}\]])', r'\1', fixed_text)
            
            # Remove control characters
            fixed_text = re.sub(r'[\x00-\x1F\x7F]', '', fixed_text)
            
            # Fix possible unclosed strings
            fixed_text = re.sub(r':\s*"([^"\n]*)(?:\n|$)', r': "\1"', fixed_text)
            
            # Extract JSON
            json_text = fixed_text[fixed_text.find('{'):fixed_text.rfind('}')+1]
            
            try:
                return json.loads(json_text)
            except json.JSONDecodeError as e2:
                print(f"Fixed JSON parsing failed: {e2}")
                
                # Try a more aggressive approach: remove problematic parts completely
                try:
                    # Get the line and column from the error message
                    error_line_match = re.search(r'line (\d+) column (\d+)', str(e2))
                    if error_line_match:
                        error_line = int(error_line_match.group(1))
                        error_col = int(error_line_match.group(2))
                        
                        lines = json_text.split('\n')
                        if 1 <= error_line <= len(lines):
                            # Try to fix the specific line
                            problem_line = lines[error_line - 1]
                            if error_col < len(problem_line):
                                # Look for issues around the error column
                                if ',' in problem_line[error_col-5:error_col+5]:
                                    # Might be a trailing comma issue
                                    fixed_line = problem_line[:error_col-1] + problem_line[error_col:]
                                else:
                                    # Might be a missing comma, add one
                                    fixed_line = problem_line[:error_col] + ',' + problem_line[error_col:]
                                
                                lines[error_line - 1] = fixed_line
                                fixed_json = '\n'.join(lines)
                                try:
                                    return json.loads(fixed_json)
                                except:
                                    pass
                except Exception as e3:
                    print(f"Line-specific fix failed: {e3}")
                    
                # Method 2: Extract individual fields and build a new object
                result = {}
                
                # Extract key-value pairs using regex
                pairs = re.findall(r'"([^"]+)"\s*:\s*("[^"]*"|\'[^\']*\'|\[[^\]]*\]|{[^}]*}|[^,}\]]+)', json_text)
                
                for key, value in pairs:
                    # Clean and parse the value
                    value = value.strip()
                    if value.startswith('"') and value.endswith('"'):
                        # String value
                        result[key] = value[1:-1]
                    elif value.startswith('[') and value.endswith(']'):
                        # Array value
                        try:
                            result[key] = json.loads(value)
                        except:
                            # If parsing fails, extract string items
                            items = re.findall(r'"([^"]*)"', value)
                            result[key] = items
                    elif value.startswith('{') and value.endswith('}'):
                        # Object value
                        try:
                            result[key] = json.loads(value)
                        except:
                            # If parsing fails, create an empty object
                            result[key] = {}
                    elif value.lower() == 'true':
                        result[key] = True
                    elif value.lower() == 'false':
                        result[key] = False
                    elif value.lower() == 'null':
                        result[key] = None
                    else:
                        # Try to parse as number
                        try:
                            if '.' in value:
                                result[key] = float(value)
                            else:
                                result[key] = int(value)
                        except:
                            # Fall back to string
                            result[key] = value
                
                if result:
                    return result
        
            # If all other methods fail, use a simpler approach - create expected structure with defaults
            expected_fields = ["score", "feedback", "strengths", "areas_for_improvement"]
            found_fields = {}
            
            # Try to find values for expected fields
            for field in expected_fields:
                field_match = re.search(rf'"{field}"\s*:\s*([^,}}]+)', json_text)
                if field_match:
                    value_str = field_match.group(1).strip()
                    if field == "score":
                        # Try to extract a number
                        num_match = re.search(r'(\d+)', value_str)
                        if num_match:
                            found_fields[field] = int(num_match.group(1))
                        else:
                            found_fields[field] = 50
                    elif field in ["strengths", "areas_for_improvement"]:
                        # Look for array contents
                        array_match = re.search(r'\[(.*)\]', value_str)
                        if array_match:
                            items = re.findall(r'"([^"]+)"', array_match.group(1))
                            found_fields[field] = items
                        else:
                            found_fields[field] = []
                    else:
                        # Extract string value
                        if value_str.startswith('"') and value_str.endswith('"'):
                            found_fields[field] = value_str[1:-1]
                        else:
                            found_fields[field] = value_str
            
            # Create default values for missing fields
            result = {
                "score": found_fields.get("score", 50),
                "feedback": found_fields.get("feedback", "Unable to parse detailed feedback."),
                "strengths": found_fields.get("strengths", ["Response showed some understanding"]),
                "areas_for_improvement": found_fields.get("areas_for_improvement", ["Consider more specific examples"])
            }
            
            return result
                
        except Exception as advanced_e:
            print(f"Advanced JSON parsing failed: {advanced_e}")
            
            # As a last resort, try to use LLM to fix the JSON
            try:
                fix_prompt = f"""The following text is supposed to be a JSON object, but it has syntax errors.
Please fix any JSON syntax errors and return ONLY the corrected valid JSON.

Original text:
{text}

Return ONLY the fixed JSON with no explanations.
"""
                
                fix_response = llm.invoke(fix_prompt)
                fixed_text = fix_response.content
                
                # Extract JSON from fixed response
                json_text = fixed_text[fixed_text.find('{'):fixed_text.rfind('}')+1]
                return json.loads(json_text)
            except Exception as llm_e:
                print(f"LLM-based JSON repair failed: {llm_e}")
                # Give up and return empty dict with expected structure
                return {
                    "score": 50,
                    "feedback": "Unable to evaluate response due to technical issues.",
                    "strengths": ["Unable to determine specific strengths"],
                    "areas_for_improvement": ["Unable to determine specific areas for improvement"]
                }

def evaluate_response(question: dict, response: str) -> dict:
    """Evaluate a user's response to an interview question and provide scoring and feedback."""
    question_text = question.get("question", "")
    question_context = question.get("context", "")
    expected_details = question.get("expected_details", [])
    
    # Convert expected_details to string if it's a list
    if isinstance(expected_details, list):
        expected_details_str = ", ".join(expected_details)
    else:
        expected_details_str = str(expected_details)
    
    # Create a simpler prompt with clearer JSON formatting instructions
    prompt = f"""Evaluate this candidate's response to an interview question.

Question: {question_text}
Context: {question_context}
Expected details in a good answer: {expected_details_str}

Candidate's response: {response}

Provide your evaluation as a JSON object with these exact fields:
- score: number from 0-100
- feedback: brief constructive feedback
- strengths: array of strengths (2-3 points)
- areas_for_improvement: array of areas to improve (2-3 points)

Use this EXACT format (including quotes and braces):
{{
  "score": 75,
  "feedback": "Your specific feedback here",
  "strengths": ["Specific strength 1", "Specific strength 2"],
  "areas_for_improvement": ["Specific area 1", "Specific area 2"]
}}

Return ONLY this JSON object, no additional text.
"""
    
    try:
        # Get raw LLM response
        raw_response = llm.invoke(prompt)
        response_text = raw_response.content
        
        # Try direct JSON parsing first
        try:
            # Extract what looks like JSON (from first { to last })
            json_text = response_text[response_text.find('{'):response_text.rfind('}')+1]
            evaluation = json.loads(json_text)
        except json.JSONDecodeError:
            # Fall back to robust parser
            evaluation = robust_json_parser(response_text)
        
        # If parsing completely failed, use default evaluation
        if not evaluation:
            evaluation = {
                "score": 50,
                "feedback": "Unable to evaluate response fully.",
                "strengths": ["Unable to determine specific strengths"],
                "areas_for_improvement": ["Consider providing more specific examples"]
            }
        
        # Ensure all required fields are present
        if "score" not in evaluation:
            evaluation["score"] = 50
        if "feedback" not in evaluation:
            evaluation["feedback"] = "Response evaluation completed."
        if "strengths" not in evaluation:
            evaluation["strengths"] = ["Response demonstrated understanding of the question"]
        if "areas_for_improvement" not in evaluation:
            evaluation["areas_for_improvement"] = ["Consider providing more specific examples"]
            
        # Ensure score is a number between 0-100
        try:
            score = float(evaluation["score"])
            if score < 0:
                evaluation["score"] = 0
            elif score > 100:
                evaluation["score"] = 100
            else:
                evaluation["score"] = round(score)
        except (ValueError, TypeError):
            evaluation["score"] = 50
            
        return evaluation
    except Exception as e:
        print(f"Error evaluating response: {e}")
        return {
            "score": 50,
            "feedback": "Unable to evaluate response fully.",
            "strengths": ["Unable to determine specific strengths"],
            "areas_for_improvement": ["Consider providing more specific examples"]
        }

def process_user_input(state: InterviewState) -> InterviewState:
    """Process user input for interview questions and evaluate the response."""
    # Get the last question asked
    if not state["conversation_history"] or state["conversation_history"][-1]["role"] != "assistant":
        return state
    
    last_question = state["conversation_history"][-1]["content"]
    question_type = state["current_question_type"]
    question_index = state["current_question_index"] - 1  # Get the index of the question we just asked
    
    # Get the question object
    question_obj = None
    if question_type == "project" and question_index >= 0 and question_index < len(state["project_questions"]):
        question_obj = state["project_questions"][question_index]
    elif question_type == "technical" and question_index >= 0 and question_index < len(state.get("technical_questions", [])):
        question_obj = state["technical_questions"][question_index]
    elif question_type == "scenario" and question_index >= 0 and question_index < len(state["scenario_questions"]):
        question_obj = state["scenario_questions"][question_index]
    elif question_type == "behavioral" and question_index >= 0 and question_index < len(state["behavioral_questions"]):
        question_obj = state["behavioral_questions"][question_index]
    
    # Get user input for the question
    print("\n---------------------------------------")
    print(f"INTERVIEWER: {last_question}")
    print("---------------------------------------")
    user_response = input("YOUR ANSWER: ")
    print("---------------------------------------\n")
    
    # Evaluate the response if we have the question object
    evaluation = None
    if question_obj:
        print("Evaluating your response...")
        evaluation = evaluate_response(question_obj, user_response)
        
        # Display score and feedback
        print(f"\nScore: {evaluation.get('score', 'N/A')}/100")
        print(f"Feedback: {evaluation.get('feedback', 'No feedback available')}")
        print("\nStrengths:")
        for strength in evaluation.get('strengths', []):
            print(f"- {strength}")
        print("\nAreas for improvement:")
        for area in evaluation.get('areas_for_improvement', []):
            print(f"- {area}")
        print("\n---------------------------------------\n")
    
    # Create candidate response object
    candidate_response = {
        "question_id": len(state["responses"]),
        "question_type": state["current_question_type"],
        "question": last_question,
        "response_text": user_response,
        "evaluation": evaluation
    }
    
    # Update state with response
    updated_state = {
        **state,
        "responses": state["responses"] + [candidate_response],
        "conversation_history": state["conversation_history"] + [
            {"role": "user", "content": user_response}
        ]
    }
    
    # For technical questions, increment the index after storing the response
    if state["current_question_type"] == "technical":
        updated_state["current_question_index"] = state["current_question_index"] + 1
    
    return updated_state

def run_interview_interactive(resume: str, job_description: str, company_name: str):
    """Run an interactive interview process with user input."""
    # Initialize the state
    state = {
        "resume": resume,
        "job_description": job_description,
        "company_name": company_name
    }
    
    # Run initialization steps
    state = initialize_state(state)
    state = extract_resume_data(state)
    state = extract_job_data(state)
    state = perform_match_analysis(state)
    
    # Generate questions for different categories
    state = generate_project_questions(state)
    state = generate_scenario_questions(state)
    state = generate_behavioral_questions(state)
    
    print("\n===== INTERVIEW STARTING =====")
    print(f"Company: {company_name}")
    print("This interview will begin with project questions, followed by technical questions.")
    print("For technical questions, your responses will influence the next questions.\n")
    
    # Set initial question type to project
    state["current_question_type"] = "project"
    state["current_question_index"] = 0
    
    # Track when user needs to provide input
    waiting_for_input = False
    
    # Interview loop
    while True:
        # If we're not waiting for input, ask the next question
        if not waiting_for_input:
            previous_state = state.copy()  # Save state before asking question
            state = ask_question(state)
            
            # If the state didn't change, we might need to generate a technical question
            if (state["current_question_type"] == "technical" and 
                state["current_question_index"] >= len(state.get("technical_questions", [])) and
                state["current_question_index"] < 4):
                state = generate_technical_questions(state)
                # Try asking the question again now that we have a new one
                state = ask_question(state)
            
            # Check if we're now waiting for input (last message from assistant)
            if state.get("conversation_history") and state["conversation_history"][-1]["role"] == "assistant":
                waiting_for_input = True
            
            # If we're moving to assessment, break the loop
            if state.get("current_step") == "perform_assessment":
                break
        
        # If we're waiting for user input, get it
        if waiting_for_input:
            previous_state = state.copy()  # Save state before processing input
            state = process_user_input(state)
            waiting_for_input = False  # After input is processed, we're no longer waiting
    
    # Perform final assessment
    state = perform_assessment(state)
    
    # Save results
    save_results_to_excel(state)
    
    print("\n===== INTERVIEW COMPLETED =====")
    if state.get('assessment'):
        print(f"Technical Score: {state['assessment'].get('technical_score', 'N/A')}")
        print(f"Behavioral Score: {state['assessment'].get('behavioral_score', 'N/A')}")
        print(f"Overall Score: {state['assessment'].get('overall_score', 'N/A')}")
        print("\nStrengths:")
        for strength in state['assessment'].get('strengths', []):
            print(f"- {strength}")
        print("\nAreas for Improvement:")
        for improvement in state['assessment'].get('improvements', []):
            print(f"- {improvement}")
    else:
        print("No assessment could be generated.")
    
    return {
        "match_analysis": state["match_analysis"],
        "assessment": state.get("assessment", {}),
        "interview_questions": {
            "project": state.get("project_questions", []),
            "technical": state.get("technical_questions", []),
            "scenario": state.get("scenario_questions", []),
            "behavioral": state.get("behavioral_questions", [])
        }
    }

# Update main code to use the interactive interview
if __name__ == "__main__":
    # Example resume and job description
    resume = '''
    John Smith
    Software Engineer

    Skills:
    - Python (Advanced)
    - Java (Intermediate)
    - SQL (Advanced)
    - Machine Learning (Intermediate)
    - Docker (Basic)

    Work Experience:
    Software Engineer, ABC Tech (2020-Present)
    - Developed backend services using Python and Flask
    - Optimized database queries reducing response time by 30%
    - Led a team of 3 developers for a customer-facing project

    Junior Developer, XYZ Corp (2018-2020)
    - Maintained Java applications and fixed bugs
    - Developed automated testing scripts

    Projects:
    E-commerce Recommendation Engine
    - Built a product recommendation system using collaborative filtering
    - Used Python, Pandas, and scikit-learn
    - Achieved 25 increase in click-through rate

    Inventory Management System
    - Developed a full-stack inventory tracking system
    - Used Java Spring Boot and React
    - Implemented real-time data synchronization

    Education:
    BS Computer Science, University of Technology (2018)
    '''

    # Job description example
    job_description = '''
    Senior Software Engineer - Data Platform
    Company: Tech Innovations Inc.

    About the Role:
    We are seeking a Senior Software Engineer to join our Data Platform team. The ideal candidate will have strong Python skills and experience with data processing frameworks.

    Key Responsibilities:
    - Design and build scalable data processing pipelines
    - Optimize data storage and retrieval mechanisms
    - Collaborate with data scientists to implement ML models
    - Mentor junior engineers and participate in code reviews

    Required Skills:
    - Advanced Python programming (5+ years)
    - Experience with SQL and NoSQL databases
    - Knowledge of data processing frameworks (Spark, Kafka)
    - Strong understanding of system design principles
    - CI/CD and testing practices

    Preferred Skills:
    - Experience with cloud platforms (AWS preferred)
    - Kubernetes and container orchestration
    - Data visualization tools
    - ML/AI implementation experience

    We value team players with excellent communication skills and problem-solving abilities.
    '''

    company_name = "Tech Innovations Inc."

    # Run the interactive interview
    print("Starting interactive interview...")
    result = run_interview_interactive(resume, job_description, company_name)
    
    # You can still save the final result to JSON if needed
    with open("interview_results/final_result.json", "w") as f:
        json.dump(result, f, indent=2)
        
    print("Interview results saved to 'interview_results/final_result.json'")
