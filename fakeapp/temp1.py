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
    behavioral_score: int
    project_score: int
    scenario_score: int
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
            "skill_match_percentage": 20.0,
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
        3. Include expected answer that would represent a good response
        
        If there are previous responses, adapt this question to build upon those responses.
        If the candidate showed strength in a specific area, you can ask more challenging
        questions in that area. If they showed weakness, ask about a different area.
        
        Output as a JSON with 'question' and 'expected_answer' fields.
        
        IMPORTANT: Pay careful attention to JSON formatting. Ensure you include a comma between fields.
        
        Return ONLY a valid JSON object with exactly these fields:
        {{
        "question": "The complete interview question text",
        "expected_answer": "2 line answer to the question"
        }}
        
        Double check that your JSON has a comma between the "question" field and the "expected_answer" field.
        """),
        ("user", """Resume skills: {resume_skills}
        Job top skills: {job_top_skills}
        Matching skills: {matching_skills}
        Missing skills: {missing_skills}
        Question number: {question_number}
        Previous responses: {previous_responses}""")
    ])
    
    chain = prompt | llm | JsonOutputParser()
    
    try:
        response = chain.invoke({
            "resume_skills": resume_data["skills"],
            "job_top_skills": job_data["top_skills"],
            "matching_skills": match_analysis["matching_skills"],
            "missing_skills": match_analysis["missing_skills"],
            "question_number": current_question_index + 1,
            "previous_responses": technical_responses
        })
        
        # Ensure the response is well-formed JSON
        try:
            if isinstance(response, str):
                # Try to parse the string as JSON if it's a string
                question = json.loads(response)
            else:
                # Use the response directly if it's already parsed
                question = response
                
            # Validate that expected keys exist
            if "question" not in question:
                raise ValueError("Missing 'question' field in response")
                
            if "expected_answer" not in question:
                question["expected_answer"] = "Candidate should provide specific examples from their experience."
        except Exception as parse_error:
            print(f"Error parsing technical question response: {parse_error}")
            question = robust_json_parser(response if isinstance(response, str) else json.dumps(response))
            
        print(f"Technical question generated: {question}")
        # Add the new question to our list
        technical_questions.append(question)
        state["technical_questions"] = technical_questions
        return {
            **state, 
            "technical_questions": technical_questions
        }
    except Exception as e:
        print(f"Error generating technical question: {e}")
        default_question = {
            "question": f"Based on your experience, how would you implement a solution for processing large datasets efficiently?",
            "expected_answer": "The candidate should discuss specific algorithms, tools, or frameworks they would use."
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
    state["current_question_type"]="project"
    current_question_index = state.get("current_question_index", 0)
    project_questions = state.get("project_questions", [])
    project_responses = [resp for resp in state.get("responses", []) if resp.get("question_type") == "project"]
    # Ensure we have the necessary data structures
    if not resume_data:
        resume_data = {"projects": [], "experience": []}
        
    if not job_data:
        job_data = {"roles_responsibilities": []}
    # Create default questions in case of failure
    default_questions = [
        {
            "question": "Please describe your most challenging project and how you approached it.",
            "expected_answer": "Describe the project, challenges faced, and solutions implemented",
            
        },
        {
            "question": "How did you handle a situation where project requirements changed significantly?",
            "expected_answer": "explain few examples of how you adapted to changes",
     
        },
        {
            "question": "Tell me about a time you collaborated with others on a technical project.",
            "expected_answer": "Describe the project, your role, and how you worked with others", 
        }
    ]
    
    try:
        print("Generating project questions...")
        if len(project_questions) >= 3:
            return state
        # Generate one question at a time for better reliability
        print(f"Generating project question #{current_question_index + 1}...")
        # Topics for questions to ensure variety
        
        
        prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert technical interviewer.
        Analyze the key projects and work experiences listed in the provided resume. Generate one highly specific, non-generic interview questions that require concise, two-line answers. Focus on probing unique aspects of each project, such as:

        Technical or operational challenges faced (e.g., 'How did you resolve [specific bottleneck] in your [project X] while balancing [constraint Y]?')
        or 
        Decisions behind tools/methods (e.g., 'Why did you choose [specific technology] over alternatives for [use case Z]?')
        or
        Impact quantification (e.g., 'What metric did you prioritize to validate the success of [feature A], and why?')
        or
        Collaboration dynamics (e.g., 'How did you align stakeholders when [specific conflict] arose during [project B]?')
        Avoid broad questions like 'Tell me about a challenge.' Instead, anchor questions to resume specifics (e.g., technologies, timelines, outcomes) to elicit focused, insightful responses.
    
        1.Ask questions for a project or work experience that is more related to job description.
        2.If there are previous responses, adapt this question to build upon those responses.
        3.If the candidate showed strength in a specific area, you can ask more challenging questions in that area. If they showed weakness, ask about a different area.
        
        Output as a JSON object with 'question', 'expected_answer' fields.
        
        IMPORTANT: Pay careful attention to JSON formatting. Ensure you include a comma between fields.
        
        Return ONLY a valid JSON object with exactly these fields:
        {{
            "question": "The complete interview question text",
            "expected_answer": "2 line answer to the question"
        }}
        
        Double check that your JSON has a comma between the "question" field and the "expected_answer" field.
        """),
        ("user", """Resume skills: {resume}
        Job description : {job_description}
        Question number: {question_number}
        Previous responses: {project_responses}
       """),
        ])
        
        
            # Get response
        chain = prompt | llm | JsonOutputParser()
        response = chain.invoke({
        "resume": resume_data,
        "job_description": job_data,
        
        "question_number": current_question_index + 1,
        "project_responses": project_responses
        })
        
        # Ensure the response is well-formed JSON
        try:
            if isinstance(response, str):
                # Try to parse the string as JSON if it's a string
                question = json.loads(response)
            else:
                # Use the response directly if it's already parsed
                question = response
                
            # Validate that expected keys exist
            if "question" not in question:
                raise ValueError("Missing 'question' field in response")
                
            if "expected_answer" not in question:
                question["expected_answer"] = "Candidate should provide specific examples from their experience."
        except Exception as parse_error:
            print(f"Error parsing project question response: {parse_error}")
            question = robust_json_parser(response if isinstance(response, str) else json.dumps(response))
            
        print(f"Project question generated: {question}" )
        project_questions.append(question)
        return {
        **state, 
        "project_questions": project_questions
        }
    except Exception as e:
            print(f"Error generating project question: {e}")
            if len(project_questions) < len(default_questions):
                project_questions.append(default_questions[len(project_questions)])
            else:
                project_questions.append(default_questions[0])
            
            return {**state, "project_questions": project_questions}


def generate_behavioral_questions(state: InterviewState) -> InterviewState:
    """Generate questions about the candidate's projects and work experience."""
    job_data=state.get("job_data", {})
    if not job_data:
        job_data = {"behavioral_requirements": [], "domain": "Technology"}
    state["current_question_type"]="behavioral"
    company_name=state["company_name"]
    domain = job_data.get("domain", "Technology")
    behavioral_reqs = job_data.get("behavioral_requirements", [])
  
    current_question_index = state.get("current_question_index", 0)
    behavioral_questions = state.get("behavioral_questions", [])
    behavioral_responses = [resp for resp in state.get("responses", []) if resp.get("question_type") == "behavioral"]
    
    # Create default questions in case of failure
    default_questions = [
      {
            "question": "Tell me about a time when you had to adapt to a significant change at work.",
             
            "expected_answer": "Describe the change, your response, and the outcome"
        },
        {
            "question": "Describe a situation where you had to work with a difficult team member.",
            "expected_answer": "Explain the conflict, your approach to resolution, and the result"
        }
    ]
    
    try:
        print("Generating behavioral questions...")
        
        # Generate one question at a time for better reliability
        print(f"Generating behavioral question #{current_question_index + 1}...")
        
        system_template = """You are an expert HR interviewer.
Create 1 behavioral interview question that assesses the candidate's soft skills and cultural fit.

Company name: {company_name}
Job behavioral requirements: {behavioral_reqs}
Job domain: {domain}

If there are previous responses, adapt this question to build upon those responses.
If the candidate showed strength in a specific area, you can ask more challenging questions in that area. 
If they showed weakness, ask about a different area.

Output as a single JSON object with 'question' and 'expected_answer' fields.

IMPORTANT: Pay careful attention to JSON formatting. Ensure you include a comma between fields.

Return ONLY a valid JSON object with exactly these fields:
{{
"question": "The complete interview question text",
"expected_answer": "2 line answer to the question"
}}

Double check that your JSON has a comma between the "question" field and the "expected_answer" field.
"""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_template),
            ("user", "Create a behavioral question based on the provided information.")
        ])
        
        # Get response
        chain = prompt | llm | JsonOutputParser()
        response = chain.invoke({
            "company_name": company_name,
            "behavioral_reqs": behavioral_reqs,
            "domain": domain
        })
        
        # Ensure the response is well-formed JSON
        try:
            if isinstance(response, str):
                # Try to parse the string as JSON if it's a string
                question = json.loads(response)
            else:
                # Use the response directly if it's already parsed
                question = response
                
            # Validate that expected keys exist
            if "question" not in question:
                raise ValueError("Missing 'question' field in response")
                
            if "expected_answer" not in question:
                question["expected_answer"] = "Candidate should provide specific examples from their experience."
        except Exception as parse_error:
            print(f"Error parsing behavioral question response: {parse_error}")
            question = robust_json_parser(response if isinstance(response, str) else json.dumps(response))
        
        print(f"behavioral question generated: {question}" )
        behavioral_questions.append(question)
        return {
        **state, 
        "behavioral_questions":behavioral_questions
        }
    except Exception as e:
            print(f"Error generating behavioral question: {e}")
            # Safely get a default question
            if len(behavioral_questions) < len(default_questions):
                default_q = default_questions[len(behavioral_questions)]
            else:
                default_q = default_questions[0]
                
            behavioral_questions.append(default_q)
            
            return {**state, "behavioral_questions": behavioral_questions}

def generate_scenario_questions(state: InterviewState) -> InterviewState:
    """Generate questions about the candidate's projects and work experience."""
    match_analysis = state.get("match_analysis", {})
    job_data = state.get("job_data", {})
    if not job_data:
        job_data = {"scenario_requirements": [], "domain": "Technology"}
    state["current_question_type"]="scenario"
    if not match_analysis:
        match_analysis = {"experience_match": {"rating": "Not evaluated", "explanation": "No evaluation available"}}
    responsibilities = job_data.get("roles_responsibilities", [])
    domain = job_data.get("domain", "Technology")
    experience_match = match_analysis.get("experience_match", {"rating": "Not evaluated", "explanation": "No evaluation available"})
  
    current_question_index = state.get("current_question_index", 0)
    scenario_questions = state.get("scenario_questions", [])
    scenario_responses = [resp for resp in state.get("responses", []) if resp.get("question_type") == "behavioral"]
    # Ensure we have the necessary data structures
    # Create default questions in case of failure
    default_questions = [
      {
            "question": "Imagine you're facing a tight deadline with competing priorities. How would you approach this situation?",
            "expected_answer": "I would prioritize tasks based on urgency and impact, communicate with stakeholders, and focus on delivering the most critical work first."
           
        },
        {
            "question": "Describe how you would handle a situation where a critical system you're responsible for fails in production.",
            "expected_answer": "I would immediately assess the situation, communicate with the team, and implement a rollback or fix while keeping stakeholders informed."
           
        }
    ]
    
    try:
        print("Generating scenario questions...")
        
        # Generate one question at a time for better reliability
        print(f"Generating scenario question #{current_question_index + 1}...")
        
        system_template = """You are an expert manager interviewer.
Create 1 behavioral interview question that assesses the candidate's soft skills and cultural fit.

Job responsibilities: {responsibilities}
Job domain: {domain}
Candidate's experience level: {experience_match}

If there are previous responses, adapt this question to build upon those responses.
If the candidate showed strength in a specific area, you can ask more challenging questions in that area. 
If they showed weakness, ask about a different area.

Output as a single JSON object with 'question' and 'expected_answer' fields.

IMPORTANT: Pay careful attention to JSON formatting. Ensure you include a comma between fields.

Return ONLY a valid JSON object with exactly these fields:
{{
"question": "The complete interview question text",
"expected_answer": "2 line answer to the question"
}}

Double check that your JSON has a comma between the "question" field and the "expected_answer" field.
"""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_template),
            ("user", "Create a scenario question based on the provided information.")
        ])
        
        # Get response
        chain = prompt | llm | JsonOutputParser()
        response = chain.invoke({
            "responsibilities": str(responsibilities),
            "domain": str(domain),
            "experience_match": str(experience_match)
        })
        
        # Ensure the response is well-formed JSON
        try:
            if isinstance(response, str):
                # Try to parse the string as JSON if it's a string
                question = json.loads(response)
            else:
                # Use the response directly if it's already parsed
                question = response
                
            # Validate that expected keys exist
            if "question" not in question:
                raise ValueError("Missing 'question' field in response")
                
            if "expected_answer" not in question:
                question["expected_answer"] = "Candidate should provide specific examples from their experience."
        except Exception as parse_error:
            print(f"Error parsing scenario question response: {parse_error}")
            question = robust_json_parser(response if isinstance(response, str) else json.dumps(response))
        
        print(f"scenario question generated: {question}" )
        scenario_questions.append(question)
        return {
            **state, 
            "scenario_questions": scenario_questions
        }
    except Exception as e:
        print(f"Error generating scenario question: {e}")
        # Safely get the default question
        if len(scenario_questions) < len(default_questions):
            default_q = default_questions[len(scenario_questions)]
        else:
            default_q = default_questions[0]
        
        scenario_questions.append(default_q)
        return {**state, "scenario_questions": scenario_questions}

def auto_repaid(response_text):
    fix_prompt=f"""The following text is supposed to be a JSON object, but it may have formatting errors.
            Please fix any JSON syntax errors and return ONLY the corrected valid JSON.

            Original text:
            {response_text}

            Return ONLY the fixed JSON object with no explanations:"""
    try:
        fix_response = llm.invoke(fix_prompt)
        fixed_text = fix_response.content
        modified_text = fixed_text[fixed_text.find('{'):fixed_text.rfind('}')+1]
        print("Successfully fixed malformed JSON")
        print("the response text is", modified_text)
        data= json.loads(modified_text)
        return data
    except Exception as fix_e:
        print(f"Failed to fix JSON: {fix_e}")
        # Use default data
        return response_text

def ask_question(state: InterviewState) -> Dict:
    """Determine the next question to ask based on current question type and index."""
    print("entered the stage ask question")

    question_type = state["current_question_type"]
    index = state["current_question_index"]
    print("the question type is", question_type)
    
    # Check if the last message in conversation history is from the assistant
    # This means we already asked a question and are waiting for an answer
    if state.get("conversation_history") and state["conversation_history"][-1]["role"] == "assistant":
        # Don't ask a new question, return the state as is
        return state
    
    
    if question_type == "project":
        questions = state.get("project_questions", [])

        if index < len(questions):
            question = questions[index]
            
            question_text = question.get("question", "Could you tell me about your project experience?")
            
            # Ask the question and increment the index for next time
            return {
                **state,
                "current_question_index": index + 1,
                "conversation_history": state["conversation_history"] + [
                    {"role": "assistant", "content": question_text,"expected_answer": question.get("expected_answer", "")}
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
            question_text = question.get("question", "Could you describe your technical experience?")
            # Ask the question (don't increment index here - will be done after response)
            return {
                **state,
                "conversation_history": state["conversation_history"] + [
                    {"role": "assistant", "content": question_text,"expected_answer": question.get("expected_answer", "")}
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
    elif question_type == "scenario":
            questions = state.get("scenario_questions", [])

            
            if index < len(questions):
                question = questions[index]
                print("the index for scenario is", index)
                print("the scenario question is", question)
                question_text = question.get("question", "How would you handle a challenging scenario?")
                # Ask the question and increment the index for next time
                return {
                    **state,
                    "current_question_index": index + 1,
                    "conversation_history": state["conversation_history"] + [
                        {"role": "assistant", "content": question_text, "expected_answer": question.get("expected_answer", "")}
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
        questions = state.get("behavioral_questions", [])
        if index < len(questions):
            question = questions[index]
            question_text = question.get("question", "Could you tell me about a challenging situation you faced?")
            # Ask the question and increment the index for next time
            return {
                **state,
                "current_question_index": index + 1,
                "conversation_history": state["conversation_history"] + [
                    {"role": "assistant", "content": question_text,"expected_answer": question.get("expected_answer", "")}
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
    total_scenario_score = 0
    scenario_count = 0
    total_project_score = 0
    project_count = 0
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
            elif q_type in ["behavioral"]:
                total_behavioral_score += score
                behavioral_count += 1
            elif q_type in ["scenario"]:
                total_scenario_score += score
                scenario_count += 1
            elif q_type in ["project"]:
                total_project_score += score
                project_count += 1
    
    avg_technical_score = round(total_technical_score / technical_count) if technical_count > 0 else None
    avg_project_score = round(total_project_score / project_count) if project_count > 0 else None
    avg_scenario_score = round(total_scenario_score / scenario_count) if scenario_count > 0 else None
    avg_behavioral_score = round(total_behavioral_score / behavioral_count) if behavioral_count > 0 else None
    overall_score = round(sum(all_scores) / len(all_scores)) if all_scores else None
    
    print("performing assessment")
    
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert hiring assessor.
        Based on the candidate's resume, job requirements, interview responses, and response scores, provide a comprehensive assessment:
        

        1. strengths: list[str] Key positive aspects identified based on interview response (3-5 points)
        2. weaknesses: list[str] Areas for improvement based on interview response (3-5 points)
        2. improvements: list[str] Improvement suggestions Specific actionable feedback based on interview response (3-5 points)
        

        
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
         
        })
        print(f"the respones for the interview are {responses}")
     
        # Use calculated scores if they're available and not in assessment
        if avg_technical_score is not None :
            assessment["technical_score"] = avg_technical_score
        if avg_behavioral_score is not None :
            assessment["behavioral_score"] = avg_behavioral_score
        if avg_project_score is not None :
            assessment["project_score"] = avg_project_score
        if avg_scenario_score is not None :
            assessment["scenario_score"] = avg_scenario_score
        if overall_score is not None :
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
            "scenario_score": avg_scenario_score or 50,
            "project_score": avg_project_score or 50,
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

# # Run the interview
# result = run_interview(resume, job_description, company_name)

# print(json.dumps(result, indent=2))


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

        

        Candidate's response: {response}

        Provide your evaluation as a JSON object with these exact fields:
        - score: number from 0-100
        - feedback: brief constructive feedback
 

        Use this EXACT format (including quotes and braces):
        
        IMPORTANT: Pay careful attention to JSON formatting. Ensure you include a comma between fields.
        
        Return ONLY a valid JSON object with exactly these fields:
        {{
        "score": 75,
        "feedback": "Your specific feedback here"
        }}
        
        Double check that your JSON has a comma between the "score" field and the "feedback" field.

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
            try:
                # Try to fix the JSON using auto_repaid function
                evaluation = auto_repaid(response_text)
            except Exception as e:
                print(response_text)
                print(f"Error fixing JSON: {e}")
                evaluation = robust_json_parser(response_text)
        
        # If parsing completely failed, use default evaluation
        if not evaluation:
            evaluation = {
                "score": 0,
                "feedback": "Unable to evaluate response fully."
            }
        
        # Ensure all required fields are present
        if "score" not in evaluation:
            evaluation["score"] = 10
        if "feedback" not in evaluation:
            print(f"evaluation")
            evaluation["feedback"] = "error in feedback"
       
            
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
            "score": 10,
            "feedback": "Unable to evaluate response fully.",

        }

def process_user_input(state: InterviewState) -> InterviewState:
    """Process user input for interview questions and evaluate the response."""
    # Get the last question asked
    if not state["conversation_history"] or state["conversation_history"][-1]["role"] != "assistant":
        return state
    
    last_question = state["conversation_history"][-1]["content"]
    expected_answer = state["conversation_history"][-1].get("expected_answer", "")
    question_type = state["current_question_type"]
    question_index = state["current_question_index"] - 1  # Get the index of the question we just asked
    
    # Get the question object
    question_obj = None
    if question_type == "project" and question_index >= 0 and question_index < len(state.get("project_questions", [])):
        question_obj = state["project_questions"][question_index]
    elif question_type == "technical" and question_index >= 0 and question_index < len(state.get("technical_questions", [])):
        question_obj = state["technical_questions"][question_index]
    elif question_type == "scenario" and question_index >= 0 and question_index < len(state.get("scenario_questions", [])):
        question_obj = state["scenario_questions"][question_index]
    elif question_type == "behavioral" and question_index >= 0 and question_index < len(state.get("behavioral_questions", [])):
        question_obj = state["behavioral_questions"][question_index]
    
    # Get user input for the questionz
    print("\n---------------------------------------")
    print(f"INTERVIEWER: {last_question}")
    print("---------------------------------------")
    user_response = "I dont know"
    print(user_response)
    # user_response = input("YOUR ANSWER: ")
    
    print("---------------------------------------\n")
    
    # Evaluate the response if we have the question object
    evaluation = None
    if question_obj:
        print("Evaluating your response...")
        evaluation = evaluate_response(question_obj, user_response)
        
        # Display score and feedback
        print(f"\nScore: {evaluation.get('score', 'N/A')}/100")
        print(f"Feedback: {evaluation.get('feedback', 'No feedback available')}")
        print("\n---------------------------------------\n")
    
    # Create candidate response object
    candidate_response = {
        "question_id": len(state["responses"]),
        "question_type": state["current_question_type"],
        "question": last_question,
        "expected_answer": expected_answer,
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
    print("the current question index is", updated_state["current_question_index"])
    # For technical questions, increment the index after storing the response
    if state["current_question_type"] == "technical":
        updated_state["current_question_index"] = state["current_question_index"] + 1
    print("the current question index is", updated_state["current_question_index"])
    
    return updated_state

def run_interview_interactive(resume: str, job_description: str, company_name: str):
    """Run an interactive interview process with user input."""
    # Initialize the state
    state = {
        "resume": resume,
        "job_description": job_description,
        "company_name": company_name,
        "responses": [],
        "conversation_history": []
    }
    
    # Run initialization steps
    state = initialize_state(state)
    state = extract_resume_data(state)
    state = extract_job_data(state)
    state = perform_match_analysis(state)
    
    # Generate questions for different categories
    state["current_question_type"] = "project"
    # state = generate_project_questions(state)

    
    print("\n===== INTERVIEW STARTING =====")
    print(f"Company: {company_name}")
    print("This interview will begin with project questions, followed by technical questions.")
    print("For technical questions, your responses will influence the next questions.\n")
    
    # Set initial question type to project
    
    state["current_question_index"] = 0
    
    # Track when user needs to provide input
    waiting_for_input = False
    
    # Interview loop
    while True:
        # print("Current state:", state)
        # If we're not waiting for input, ask the next question
        print("Current question type is", state["current_question_type"])
        print("Current question index is", state["current_question_index"])
        
        # Check if we need to move to assessment phase
        if state.get("current_step") == "perform_assessment":
            break
            
        if not waiting_for_input:
            previous_state = state.copy()  # Save state before asking question
            try:
                if (state["current_question_type"] == "project" and 
                    state["current_question_index"] >= len(state.get("project_questions", [])) and
                    state["current_question_index"] < 3):
                    state = generate_project_questions(state)
                    # Try asking the question again now that we have a new one
                    state = ask_question(state)
                elif (state["current_question_type"] == "project" and state["current_question_index"] >= 3):
                    state["current_question_type"] = "technical"
                    state["current_question_index"] = 0          
                # If the state didn't change, we might need to generate a technical question
                if (state["current_question_type"] == "technical" and 
                    state["current_question_index"] >= len(state.get("technical_questions", [])) and
                    state["current_question_index"] < 4):
                    state = generate_technical_questions(state)
                    # Try asking the question again now that we have a new one
                    state = ask_question(state)
                elif (state["current_question_type"] == "technical" and state["current_question_index"] >= 4):
                    print("finished technical questions")
                    print("Moving to scenario questions")
                    state["current_question_type"] = "scenario"
                    state["current_question_index"] = 0

                if (state["current_question_type"] == "scenario" and 
                    state["current_question_index"] >= len(state.get("scenario_questions", [])) and
                    state["current_question_index"] < 2):
                    state = generate_scenario_questions(state)
                    # Try asking the question again now that we have a new one
                    state = ask_question(state)
                elif (state["current_question_type"] == "scenario" and state["current_question_index"] >= 2):
                    state["current_question_type"] = "behavioral"
                    state["current_question_index"] = 0

                if (state["current_question_type"] == "behavioral" and 
                    state["current_question_index"] >= len(state.get("behavioral_questions", [])) and
                    state["current_question_index"] < 2):
                    state = generate_behavioral_questions(state)
                    # Try asking the question again now that we have a new one
                    state = ask_question(state)
                elif (state["current_question_type"] == "behavioral" and state["current_question_index"] >= 2):
                    state["current_step"] = "perform_assessment"
                    break
            except Exception as e:
                print(f"Error in interview flow: {e}")
                # If there was an error, try to move to the next question type
                if state["current_question_type"] == "project":
                    state["current_question_type"] = "technical"
                    state["current_question_index"] = 0
                elif state["current_question_type"] == "technical":
                    state["current_question_type"] = "scenario"
                    state["current_question_index"] = 0
                elif state["current_question_type"] == "scenario":
                    state["current_question_type"] = "behavioral"
                    state["current_question_index"] = 0
                elif state["current_question_type"] == "behavioral":
                    state["current_step"] = "perform_assessment"
                    break
            
            # Check if we're now waiting for input (last message from assistant)
            if state.get("conversation_history") and state["conversation_history"][-1]["role"] == "assistant":
                waiting_for_input = True
        
        # If we're waiting for user input, get it
        if waiting_for_input:
            previous_state = state.copy()  # Save state before processing input
            state = process_user_input(state)
            waiting_for_input = False  # After input is processed, we're no longer waiting
    
    # Perform final assessment
    try:
        state = perform_assessment(state)
        #save json to file
        with open("interview_results/state.json", "w") as f:
            json.dump(state, f, indent=2)
     
    except Exception as e:
        print(f"Error performing assessment: {e}")
        # Create a default assessment
        state["assessment"] = {
            "technical_score": 50,
            "behavioral_score": 50,
            "overall_score": 50,
            "strengths": ["Unable to properly assess strengths due to technical issues."],
            "weaknesses": ["Unable to properly assess weaknesses due to technical issues."],
            "improvements": ["Complete a more thorough interview process."]
        }
    
    # Save results
    # save_results_to_excel(state)
    
    print("\n===== INTERVIEW COMPLETED =====")
    if state.get('assessment'):
        print(f"Technical Score: {state['assessment'].get('technical_score', 'N/A')}")
        print(f"Behavioral Score: {state['assessment'].get('behavioral_score', 'N/A')}")
        print(f"Project Score: {state['assessment'].get('project_score', 'N/A')}")
        print(f"Scenario Score: {state['assessment'].get('scenario_score', 'N/A')}")

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
        "match_analysis": state.get("match_analysis", {}),
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

        
    print("Interview results saved to 'interview_results/final_result.json'")
