
  # 🧠 F.A.K.E (Friendly Ai for Knowledge Evaluation)

An intelligent, end-to-end interview platform that conducts personalized interviews using AI. Built with **LangGraph**, **LLaMA-3 via GROQ API**, and **Whisper-v3**, the app can assess candidates either through a **resume-based multi-round interview** or a **skill-driven question set**, all integrated into a clean dashboard deployed with **Django**, **PostgreSQL**, and **Docker** on **AWS EC2**.

---


## 🌐 Live Demo
[https://fake-app.site](https://fake-app.site)

## 🔐 Login & Interview Selection

Upon login, the user is presented with two flows:

### 1️⃣ Skill-Based Interview

* **Inputs**:

  * User-defined skills
  * Number of questions

* **Flow**:

  * System generates only technical questions based on uploaded skills.
  * Uses LLaMA-3 via GROQ for question generation.
  * Candidate responds via voice (transcribed with Whisper-v3).
  * Each response is scored in real time.
  * Results visualized in the final dashboard.

---

  ![image](https://github.com/user-attachments/assets/ca729399-6ee3-4269-8088-6adb6ccd069b)

---

### 2️⃣ Resume-Based Interview

* **Inputs**:

  * Resume (.pdf or .txt)
  * Job Description

* **Interview Rounds** (Total: 11 Questions):

  * 🔧 4 Technical
  * 🧪 3 Project-based
  * 🧠 2 Scenario-based
  * 💬 2 Behavioral

* **LangGraph Flow Overview** (see diagram above):

  * `initialize_state`: Initializes graph state
  * `extract_resume_data` & `extract_job_data`: Extracts relevant info
  * `perform_match_analysis`: Compares skills & roles
  * `generate_*_questions`: Generates type-specific questions
  * `ask_question`: Interacts with user using Whisper-v3 for voice input
  * `process_response`: Evaluates with context
  * `route_next_step`: Determines next type of question or triggers assessment
  * `perform_assessment`: Scores across dimensions

* **Assessment Dimensions**:

  * Technical Strength
  * Communication Skills
  * Relevance to Job Role
  * Problem-Solving Skills

---

![image](https://github.com/user-attachments/assets/cc35b26f-12e3-4890-951d-e0147b18eb14)

![image](https://github.com/user-attachments/assets/84941080-15a8-4d9a-b621-98408a8cd03b)

![fakegraph](https://github.com/user-attachments/assets/e3d63dad-e9d5-4918-989a-cc33d4a7b5bf)

---

## 🧰 Tech Stack

| Component         | Technology           |
| ----------------- | -------------------- |
| LLM Integration   | LLaMA-3 via GROQ API |
| State Machine     | LangGraph            |
| Audio Interface   | Whisper-v3           |
| Backend Framework | Django (Python)      |
| Database          | PostgreSQL           |
| Deployment        | AWS EC2              |

---

## 📊 Final Dashboard

A comprehensive dashboard is generated post-interview:

* Round-wise Performance
* Skill Match Visualization
* Recommendations & Final Score

   
📬 Contact
For questions or support, reach out to: [abhivsp10@gmail.com]
---

