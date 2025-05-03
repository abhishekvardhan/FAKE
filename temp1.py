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
    
    prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert in analyzing job descriptions for hiring.
    Extract the following information from the given job description:
    
    The output must follow this exact JSON structure:
    {{
        "top_skills": {{
            "technical_skills": ["<technical_skill_1>", "<technical_skill_2>", ...],
            "soft_skills": ["<soft_skill_1>", "<soft_skill_2>", ...]
        }},
        "preferred_skills": ["<preferred_skill_1>", "<preferred_skill_2>", ...],
        "roles_responsibilities": ["<responsibility_1>", "<responsibility_2>", ...],
        "behavioral_requirements": ["<requirement_1>", "<requirement_2>", ...],
        "domain": "<domain_or_industry_focus>"
    }}
    
    Consider the company name: {company_name}
    
    Important:
    1. Ensure all JSON keys and formatting exactly match the structure above
    2. The "top_skills" object must contain "technical_skills" and "soft_skills" arrays
    3. All list items should be complete, concise phrases extracted from the job description
    4. Return only valid JSON without any additional text or markdown formatting
    5. Verify the JSON is complete and properly structured before returning
    """),
    ("user", "{job_description}")
    ])
    
    parser = JsonOutputParser(pydantic_object=JobData)
    chain = prompt | llm | parser
    
    try:
        print("parsing job description")
        job_data = chain.invoke({"job_description": job_description, "company_name": company_name})
        
        return {**state, "job_data": job_data, "current_step": "perform_match_analysis"}
    except Exception as e:
        # Handle parsing errors gracefully
        print(f"Error extracting job data: {e}")
        default_data = {
            "top_skills": [],
            "preferred_skills": [],
            "roles_responsibilities": [],
            "behavioral_requirements": [],
            "domain": ""
        }
        return {**state, "job_data": default_data, "current_step": "perform_match_analysis"}

def perform_match_analysis(state: InterviewState) -> InterviewState:
    """Analyze the match between resume and job requirements."""
    resume_data = state["resume_data"]
    job_data = state["job_data"]
    
    prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert in candidate-job matching analysis.
    Compare the candidate's resume data with the job requirements and provide a detailed assessment.
    
    Your analysis must be returned in this exact JSON structure:
    {{
        "matching_skills": ["<matching_skill_1>", "<matching_skill_2>", ...],
        "missing_skills": ["<missing_skill_1>", "<missing_skill_2>", ...],
        "skill_match_percentage": <number_between_0_and_100>,
        "experience_match": {{
            "rating": "<excellent/good/fair/poor>",
            "explanation": "<brief_explanation_of_rating>"
        }}
    }}
    
    Important:
    1. "matching_skills" should list all skills that appear in both the resume and job requirements
    2. "missing_skills" should list skills required by the job but not found in the resume
    3. "skill_match_percentage" should be a number between 0-100 representing the percentage of required skills the candidate possesses
    4. "experience_match" should assess how well the candidate's work experience aligns with the job requirements
    5. Return only valid JSON without any additional text or markdown formatting
    """),
    ("user", "Resume data: {resume_data}\nJob data: {job_data}")
])
    
    parser = JsonOutputParser(pydantic_object=MatchAnalysis)
    chain = prompt | llm | parser
    
    try:
        match_analysis = chain.invoke({"resume_data": resume_data, "job_data": job_data})
        print("match analysis")
        return {**state, "match_analysis": match_analysis, "current_step": "generate_technical_questions"}
    except Exception as e:
        print(f"Error performing match analysis: {e}")
        default_analysis = {
            "matching_skills": [],
            "missing_skills": [],
            "skill_match_percentage": 0.0,
            "experience_match": "Unable to determine"
        }
        return {**state, "match_analysis": default_analysis, "current_step": "generate_technical_questions"}

def generate_technical_questions(state: InterviewState) -> InterviewState:
    """Generate technical interview questions based on the matching skills and job requirements."""
    resume_data = state["resume_data"]
    job_data = state["job_data"]
    match_analysis = state["match_analysis"]
    print("job data is")
    print(job_data)
    print(resume_data)
    print(match_analysis)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert technical interviewer.
        Based on the candidate's matching skills and the job's top skill requirements, 
        create 4 in-depth technical questions that will assess the candidate's expertise.
        
        For each question:
        1. Make it specific to the required skills
        2. Ensure it tests both theoretical knowledge and practical application
        3. Include context on what skill/requirement this relates to
        4. Include expected details that would be present in a good answer
        
        Output as a JSON array of question objects."""),
        ("user", """Resume skills: {resume_skills}
        Job top skills: {job_top_skills}
        Matching skills: {matching_skills}
        Missing skills: {missing_skills}""")
    ])
    
    chain = prompt | llm | JsonOutputParser()
    
    try:
        questions = chain.invoke({
            "resume_skills": resume_data["skills"],
            "job_top_skills": job_data["top_skills"],
            # match_analysis["matching_skills"].append("python")
            "matching_skills": match_analysis["matching_skills"],
            "missing_skills": match_analysis["missing_skills"]
        })
        
        # Ensure we have exactly 4 questions
        technical_questions = questions[:4]
        while len(technical_questions) < 4:
            technical_questions.append({
                "question": f"Default technical question #{len(technical_questions)+1}",
                "context": "General technical assessment",
                "expected_details": ["Technical knowledge", "Problem-solving approach"]
            })
            
        return {**state, "technical_questions": technical_questions, "current_step": "generate_project_questions"}
    except Exception as e:
        print(f"Error generating technical questions: {e}")
        default_questions = [
            {
                "question": "Please describe your experience with the primary technologies mentioned in your resume.",
                "context": "General technical assessment",
                "expected_details": ["Technical depth", "Practical experience", "Problem-solving approach"]
            } for _ in range(4)
        ]
        return {**state, "technical_questions": default_questions, "current_step": "generate_project_questions"}

def generate_project_questions(state: InterviewState) -> InterviewState:
    """Generate questions about the candidate's projects and work experience."""
    resume_data = state["resume_data"]
    job_data = state["job_data"]
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert interviewer focusing on work experience and projects.
        Based on the candidate's resume and the job requirements, create questions that:
        
        1. Deep dive into their most relevant projects
        2. Explore their problem-solving approach
        3. Assess their role and contributions in team settings
        4. Evaluate how their experience relates to the job requirements
        
        Create questions that will allow follow-up based on their responses.
        Output as a JSON array of question objects."""),
        ("user", """Resume projects: {resume_projects}
        Resume experience: {resume_experience}
        Job responsibilities: {job_responsibilities}""")
    ])
    
    chain = prompt | llm | JsonOutputParser()
    
    try:

        questions = chain.invoke({
            "resume_projects": resume_data["projects"],
            "resume_experience": resume_data["experience"],
            "job_responsibilities": job_data["roles_responsibilities"]
            
        })
        
        # Ensure we have at least 3 questions
        project_questions = questions[:3] if len(questions) > 3 else questions
        while len(project_questions) < 3:
            project_questions.append({
                "question": f"Default project question #{len(project_questions)+1}",
                "context": "Project experience assessment",
                "expected_details": ["Project contributions", "Challenges faced", "Solutions implemented"]
            })
            
        return {**state, "project_questions": project_questions, "current_step": "generate_scenario_questions"}
    except Exception as e:
        print(f"Error generating project questions: {e}")
        default_questions = [
            {
                "question": "Please describe your most challenging project and how you approached it.",
                "context": "Project experience assessment",
                "expected_details": ["Problem definition", "Approach", "Outcome", "Lessons learned"]
            } for _ in range(3)
        ]
        return {**state, "project_questions": default_questions, "current_step": "generate_scenario_questions"}

def generate_scenario_questions(state: InterviewState) -> InterviewState:
    """Generate scenario-based questions related to the job responsibilities."""
    job_data = state["job_data"]
    match_analysis = state["match_analysis"]
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert interviewer specializing in scenario-based questions.
        Based on the job's roles and responsibilities, create 2 scenario questions that:
        
        1. Present realistic situations the candidate might face in this role
        2. Test their problem-solving approach and decision-making
        3. Assess their ability to apply their skills in context
        4. Allow for follow-up questions based on their response
        
        Output as a JSON array of question objects."""),
        ("user", """Job responsibilities: {job_responsibilities}
        Job domain: {job_domain}
        Skill match: {skill_match}""")
    ])
    
    chain = prompt | llm | JsonOutputParser()
    
    try:
        questions = chain.invoke({
            "job_responsibilities": job_data["roles_responsibilities"],
            "job_domain": job_data["domain"],
            "skill_match": match_analysis["experience_match"]
        })
        print(f"scenario questions {questions}")
        
        # Ensure we have exactly 2 questions
        scenario_questions = questions[:2] if len(questions) >= 2 else questions
        while len(scenario_questions) < 2:
            scenario_questions.append({
                "question": f"Default scenario question #{len(scenario_questions)+1}",
                "context": "Role-specific scenario",
                "expected_details": ["Approach", "Problem-solving", "Communication", "Technical application"]
            })
            
        return {**state, "scenario_questions": scenario_questions, "current_step": "generate_behavioral_questions"}
    except Exception as e:
        print(f"Error generating scenario questions: {e}")
        default_questions = [
            {
                "question": "Imagine you're facing a tight deadline with competing priorities. How would you approach this situation?",
                "context": "Time management and prioritization",
                "expected_details": ["Prioritization method", "Communication", "Delegation", "Stress management"]
            } for _ in range(2)
        ]
        return {**state, "scenario_questions": default_questions, "current_step": "generate_behavioral_questions"}

def generate_behavioral_questions(state: InterviewState) -> InterviewState:
    """Generate behavioral questions based on the company and job requirements."""
    job_data = state["job_data"]
    company_name = state["company_name"]
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert in behavioral interviewing.
        Based on the company culture and job requirements, create 2 behavioral questions that:
        
        1. Assess cultural fit with {company_name}
        2. Evaluate soft skills required for the role
        3. Test for behaviors specifically mentioned in the job description
        4. Allow insight into the candidate's work style and values
        
        Output as a JSON array of question objects."""),
        ("user", """Company name: {company_name}
        Job behavioral requirements: {behavioral_requirements}
        Job domain: {job_domain}""")
    ])
    
    chain = prompt | llm | JsonOutputParser()
    
    try:
        questions = chain.invoke({
            "company_name": company_name,
            "behavioral_requirements": job_data["behavioral_requirements"],
            "job_domain": job_data["domain"]
        })
        
        # Ensure we have exactly 2 questions
        behavioral_questions = questions[:2] if len(questions) >= 2 else questions
        while len(behavioral_questions) < 2:
            behavioral_questions.append({
                "question": f"Default behavioral question #{len(behavioral_questions)+1}",
                "context": "Behavioral assessment",
                "expected_details": ["Past behavior", "Self-awareness", "Growth mindset", "Communication style"]
            })
            
        return {
            **state, 
            "behavioral_questions": behavioral_questions, 
            "current_step": "ask_question",
            "current_question_type": "project"  # Start with project questions
        }
    except Exception as e:
        print(f"Error generating behavioral questions: {e}")
        default_questions = [
            {
                "question": "Tell me about a time when you had to adapt to a significant change at work.",
                "context": "Adaptability assessment",
                "expected_details": ["Situation", "Action", "Result", "Learning"]
            } for _ in range(2)
        ]
        return {
            **state, 
            "behavioral_questions": default_questions, 
            "current_step": "ask_question",
            "current_question_type": "project"  # Start with project questions
        }

def ask_question(state: InterviewState) -> Dict:
    """Determine the next question to ask and simulate a response."""
    question_type = state["current_question_type"]
    index = state["current_question_index"]
    
    # Generate a simulated response for the previous question if not the first question
    if len(state["responses"]) > 0:
        # Get the last asked question
        last_question = state["conversation_history"][-1]["content"]
        simulated_response = "This is a simulated response to: " + last_question[:20] + "..."
        
        # Store the response
        candidate_response: CandidateResponse = {
            "question_id": len(state["responses"]),
            "question_type": state["current_question_type"],
            "question": last_question,
            "response_text": simulated_response
        }
        
        # Update state with the response
        state = {
            **state,
            "responses": state["responses"] + [candidate_response],
            "conversation_history": state["conversation_history"] + [
                {"role": "user", "content": simulated_response}
            ]
        }
    
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
            return {
                **state,
                "current_question_type": "technical",
                "current_question_index": 0
            }
    
    elif question_type == "technical":
        questions = state["technical_questions"]
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
            # Move to scenario questions
            return {
                **state,
                "current_question_type": "scenario",
                "current_question_index": 0
            }
    
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
    print("performing assessment")
    print(responses)
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert hiring assessor.
        Based on the candidate's resume, job requirements, and interview responses, provide a comprehensive assessment:
        
        1. technical_score: (0-100) How well the candidate's technical skills match the job requirements
        2. behavioral_score: (0-100) How well the candidate's soft skills and behaviors align with the role
        3. overall_score: (0-100) A weighted total assessment
        4. strengths: list[str] Key positive aspects identified (3-5 points)
        5. weaknesses: list[str] Areas for improvement (3-5 points)
        6. improvements: list[str] Improvement suggestions Specific actionable feedback (3-5 points)
        
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
            "responses": responses
        })
        
        return {
            **state, 
            "assessment": assessment,
            "current_step": END
        }
    except Exception as e:
        print(f"Error performing assessment: {e}")
        default_assessment = {
            "technical_score": 50,
            "behavioral_score": 50,
            "overall_score": 50,
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
    
    # Set conditional edges
    workflow.add_edge("initialize_state", "extract_resume_data")
    workflow.add_edge("extract_resume_data", "extract_job_data")
    workflow.add_edge("extract_job_data", "perform_match_analysis")
    workflow.add_edge("perform_match_analysis", "generate_technical_questions")
    workflow.add_edge("generate_technical_questions", "generate_project_questions")
    workflow.add_edge("generate_project_questions", "generate_scenario_questions")
    workflow.add_edge("generate_scenario_questions", "generate_behavioral_questions")
    workflow.add_edge("generate_behavioral_questions", "ask_question")
    
    # Dynamic routing based on current_step
    workflow.add_conditional_edges(
        "ask_question",
        lambda state: state.get("current_step", "ask_question"),
        {
            "ask_question": "ask_question",
            "perform_assessment": "perform_assessment"
        }
    )
    
    # Set the entrypoint
    workflow.set_entry_point("initialize_state")
    workflow1 = workflow.compile()
    try:
        print("Graph built successfully.")
        # img_data = display(Image(workflow1.get_graph().draw_mermaid_png()))
        img_data =workflow1.get_graph().draw_mermaid_png(max_retries=5, retry_delay=2.0)
        with open("output.png", "wb") as f:
            f.write(img_data)
        # print("Graph image saved as output.png")
        # display(Image(filename="output.png"))
    except Exception as e:
   
        print(f"Error saving image: {e}")
    
    return workflow1

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
        "Matching Skills": pd.Series(state["match_analysis"]["matching_skills"]),
        "Missing Skills": pd.Series(state["match_analysis"]["missing_skills"])
    })
    match_summary = pd.DataFrame({
        "Metric": ["Skill Match Percentage", "Experience Match"],
        "Value": [state["match_analysis"]["skill_match_percentage"], state["match_analysis"]["experience_match"]]
    })
    
    with pd.ExcelWriter("interview_results/match_analysis.xlsx") as writer:
        match_df.to_excel(writer, sheet_name="Skills Comparison", index=False)
        match_summary.to_excel(writer, sheet_name="Summary", index=False)
    
    # Save interview questions and responses
    all_questions = []
    
    # Add project questions
    for i, q in enumerate(state["project_questions"]):
        all_questions.append({
            "Question Type": "Project",
            "Question": q["question"],
            "Context": q.get("context", "Not specified"),
            "Expected Details": ", ".join(q.get("expected_details", ["Not specified"]))
        })
    
    # Add technical questions
    for i, q in enumerate(state["technical_questions"]):
        all_questions.append({
            "Question Type": "Technical",
            "Question": q["question"],
            "Context": q.get("context", "Not specified"),
            "Expected Details": ", ".join(q.get("expected_details", ["Not specified"]))
        })
    
    # Add scenario questions
    for i, q in enumerate(state["scenario_questions"]):
        all_questions.append({
            "Question Type": "Scenario",
            "Question": q["question"],
            "Context": q.get("context", "Not specified"),
            "Expected Details": ", ".join(q.get("expected_details", ["Not specified"]))
        })
    
    # Add behavioral questions
    for i, q in enumerate(state["behavioral_questions"]):
        all_questions.append({
            "Question Type": "Behavioral",
            "Question": q["question"],
            "Context": q.get("context", "Not specified"),
            "Expected Details": ", ".join(q.get("expected_details", ["Not specified"]))
        })
    
    # Convert to DataFrame and save
    questions_df = pd.DataFrame(all_questions)
    questions_df.to_excel("interview_results/interview_questions.xlsx", index=False)
    
    # Save responses if any
    if state["responses"]:
        responses_df = pd.DataFrame([
            {
                "Question ID": r["question_id"],
                "Question Type": r["question_type"],
                "Question": r["question"],
                "Response": r["response_text"]
            } for r in state["responses"]
        ])
        responses_df.to_excel("interview_results/candidate_responses.xlsx", index=False)
    
    # Save assessment
    assessment = state["assessment"]
    assessment_summary = pd.DataFrame({
        "Metric": ["Technical Score", "Behavioral Score", "Overall Score"],
        "Value": [assessment["technical_score"], assessment["behavioral_score"], assessment["overall_score"]]
    })
    
    feedback_df = pd.DataFrame({
        "Strengths": pd.Series(assessment["strengths"]),
        "Weaknesses": pd.Series(assessment["weaknesses"]),
        "Improvements": pd.Series(assessment["improvements"])
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
