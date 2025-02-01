# Personalized AI Tutoring Platform

## Overview
The **Personalized AI Tutoring Platform** is designed to provide students with AI-powered learning tools that help them overcome educational barriers. By offering intelligent tutoring, personalized study plans, adaptive quizzes, and contextualized summarization, the platform enhances the learning experience for students and educators alike.
Here's a cleaned-up and well-formatted version of your instructions:  

---

## Get Started  

To locally host this project, follow these steps:  

1. **Download and Extract**  
   - Download the ZIP file and unzip it.  

2. **Set Up a Virtual Environment**  
   - Use your Python version to create a new virtual environment:  
     ```sh
     python -m venv env
     ```  
   - Activate the virtual environment:  
     - **Windows:**  
       ```sh
       env\Scripts\activate
       ```  
     - **Mac/Linux:**  
       ```sh
       source env/bin/activate
       ```  

3. **Configure Environment Variables**  
   - Create a `.env` file in the project directory and add the following:  
     ```
     OPENAI_API_KEY={YOUR_API_KEY_HERE}
     GUARDRAILS_CLI_TOKEN={API_KEY_FOR_GUARDRAILS}
     ```  

4. **Install Dependencies**  
   - Install required packages:  
     ```sh
     pip install -r requirements.txt
     ```  

5. **Run the Setup Script**  
   - Start the setup process:  
     ```sh
     python server_setup.py
     ```  
   - Wait for all Guardrails to finish installing & until you see "Server is running on ____"  

Type in localhost:10000 or 127.0.0.1:10000 to view the website!

You're now ready to use the AI tutoring platform! 🚀 

---


## Problem Statement
Many students lack access to adequate study resources, leading to learning gaps, decreased motivation, and limited opportunities. This issue is particularly severe for students in underserved communities, exacerbating educational inequalities.

## Solution
Our AI tutoring platform addresses these challenges by offering:
- **Intelligent Chatbot**: AI-driven explanations across various subjects, ensuring accuracy and minimizing bias.
- **Personalized Study Plans**: Tailored learning paths based on student performance.
- **Adaptive Quizzing**: Dynamic quizzes to test knowledge and reinforce learning.
- **Automated Text Summarization**: Allows users to upload textbooks/PDFs and receive contextualized summaries with references.

## Key Innovations
- **Contextualized Summarization**: Displays the original source material alongside AI-generated summaries.
- **Adaptive Learning Paths**: AI refines study plans dynamically based on student interaction.
- **Bias-Mitigated Responses**: The chatbot ensures fairness and neutrality in explanations.
- **Integrated Learning Ecosystem**: Combines chatbot, study plans, quizzes, and summarization for a seamless learning experience.

## Target Impact
- **Students**: Improved learning outcomes, confidence, and academic preparation.
- **Teachers**: Insights into student performance and reduced workload for repetitive tasks.
- **Educational Institutions**: Enhanced student performance and reduced achievement gaps.

---

## Technical Documentation
### System Architecture
Our platform is built as a **web-based AI tutoring system** with the following components:

### **User Interface (UI)**
Developed using **Flask**, the UI includes:
- Input fields for questions and PDF uploads.
- Display area for AI-generated answers.
- Navigation between chatbot, study plans, quizzes, and summarizer.

### **Backend Processing**
- Built with **Python & Flask** for handling user requests.
- Interacts with the AI model for processing queries and generating responses.

### **AI Model**
- **OpenAI's GPT-4** for natural language understanding and response generation.
- Uses OpenAI API to create study plans, quizzes, and summaries.

### **Data Processing**
- Uses **PyPDF2** to extract text from uploaded PDFs.
- Implements **text cleaning and validation** to ensure quality content.


### **AI Model & Libraries**
- **Core AI Model**: OpenAI’s GPT API
- **Libraries**:
  - `flask` - Web application framework
  - `openai` - Interacting with GPT-4 API
  - `pypdf2` - PDF text extraction
  - `guardrails-ai` - Ensuring unbiased and safe responses
  - `ChatOpenAI` - Used for Q&A chains
  - `guard` - Additional safety checks
- **Dependencies**: Listed in `requirements.txt`

### **Data Handling**
- **User-provided PDFs** serve as data sources.
- **Validation**: Ensures only educational textbooks are processed.
- **Privacy**:
  - **Data minimization**: Only essential inputs are stored temporarily.
  - **Auto-deletion**: User data (PDFs, inputs) is wiped after session expiry.

---

## Functionality
### **User Workflow**
1. **Select a feature:** Chatbot, Study Plan, Quiz, or Summarizer.
2. **Provide input:** Enter a question, topic, or upload a textbook.
3. **AI processes request:** Extracts data, generates responses, and displays results.

### **Features**
- **Chatbot**: Answering user queries based on provided textbooks.
- **Study Plan Generator**: Personalized learning paths with targeted questions.
- **Quiz Generator**: AI-crafted quizzes for self-assessment.
- **Summarizer**: Contextual summaries from uploaded textbooks.

### **Use Cases**
- **Students**: Homework help, concept clarification, and test preparation.
- **Teachers**: Supplemental teaching resources and performance insights.

---

## Ethics & Responsibility
### **Bias Mitigation**
- **Guardrails-AI tools** detect and correct biased language.
- **Transparency measures** explain AI decision-making processes.

### **Fairness & Privacy**
- **User testing** ensures diverse accessibility.
- **Data privacy compliance** with minimized data retention policies.

### **Transparency Measures**
- **Explainable AI (XAI)** clarifies system decisions.
- **Comprehensive documentation** on platform workings and limitations.

---

## Conclusion
The **Personalized AI Tutoring Platform** democratizes access to high-quality education by offering intelligent, adaptive, and bias-mitigated learning tools. By leveraging AI responsibly, we empower students and educators, ensuring **fair, transparent, and effective learning experiences** for all.

---

## License
This project is licensed under the MIT License.

## Contact
For contributions, questions, or collaborations, reach out via GitHub.

---

