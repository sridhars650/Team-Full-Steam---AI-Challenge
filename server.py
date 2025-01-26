# HELLO sridhar
##Imports for Baseline QA Pipeline
import subprocess
from langchain_community.document_loaders import PyPDFLoader # for loading the pdf
from langchain_openai import OpenAIEmbeddings # for creating embeddings
from langchain_community.vectorstores import Chroma # for the vectorization part
from langchain.chains import RetrievalQA #For the retrieval QA chain part # apparently deprecated
from langchain_openai import ChatOpenAI #for getting an LLM for QA chain
#from langchain_core.output_parsers import StrOutputParser #Not used currently, leaving, as can be used for parsing output from LLM
#from langchain_core.runnables import RunnablePassthrough #Not used currently, leaving, as can be used for getting LLM output
from langchain.prompts import ChatPromptTemplate #for setting up prompts
from langchain.text_splitter import TokenTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.docstore.document import Document
from langchain.chat_models import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from prompts import PROMPT_QUESTIONS, REFINE_PROMPT_QUESTIONS
from langchain.chains import RetrievalQA
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma


#Guardrails stuff
from guardrails import Guard, OnFailAction
from guardrails.hub import BanList, BiasCheck, NSFWText, ProfanityFree, LogicCheck, MentionsDrugs, PolitenessCheck, ToxicLanguage#, ToxicLanguage# Updated import
#import guardrails.hub #which needs fuzzysearch, which is already downloaded
#from guardrails.datatypes import String
from langchain.chains import LLMChain
from langchain_community.llms import OpenAI

#Setup openai key
import os
from dotenv import load_dotenv
load_dotenv() 


from guardrails import Guard


guard = Guard().use_many(
    BiasCheck(
        threshold=0.5,
        on_fail="fix"
    ),

    NSFWText(
        threshold=0.8,
        validation_method="sentence"
    ),

    ProfanityFree(
        on_fail = "fix"
    ),

    LogicCheck(
        model="gpt-3.5-turbo",
        on_fail="fix"
    ),

    MentionsDrugs(
        on_fail = "fix"
    ),

    PolitenessCheck(
        llm_callable="gpt-3.5-turbo",
        on_fail = "fix"
    ),

    ToxicLanguage(
        threshold=0.5,
        validation_method="sentence",
        on_fail="fix"
    )

)



#Setup Base QA system pipeline
class BaseQAPipeline:
    def __init__(self):
        self.doc = "tutor_textbook.pdf"
        self.loader = PyPDFLoader(self.doc)

        # Load the document and store it in the 'data' variable
        self.data = self.loader.load_and_split()

        self.embeddings = OpenAIEmbeddings()
        self.vectordb = Chroma.from_documents(self.data, embedding=self.embeddings,
                                 persist_directory=".")

        # Initialize a language model with ChatOpenAI
        self.llm = ChatOpenAI(model_name= 'gpt-3.5-turbo', temperature=0.6)

        #Setup a prompt template
        template = """\
        You are an assistant for question-answering tasks.

        Use the following pieces of retrieved context to answer the question.

        If either the PDF or the question are not related to each other or not 
        related to any educational standard, state the following: This content is 
        not related to any educational purposes. 

        For example, if topics are not the same, like a java textbook is given, 
        however, the user asks about a physics question, state the following: This
        content is not related to the inputted textbook, please select another textbook
        and try again.

        If you don't know the answer, just say that you don't know.

        Use three sentences maximum and keep the answer concise. 

        Question: {question}

        Context: {context}

        Answer:

        """

        prompt = ChatPromptTemplate.from_template(template)

        chain_type_kwargs = {"prompt": prompt}



        # 1. Vectorstore-based retriever
        self.vectorstore_retriever = self.vectordb.as_retriever()

        # Initialize a RetrievalQA chain with the language model and vector database retriever
        self.qa_chain = RetrievalQA.from_chain_type(self.llm, retriever= self.vectorstore_retriever, chain_type_kwargs=chain_type_kwargs)
        self.chat_history = []  # Initialize chat history

    def update_chat_history(self, question, answer):
        self.chat_history.append({"question": question, "answer": answer})

    def build_combined_context(self):
        """Combine chat history and document context."""
        # Combine all previous chat history
        chat_context = "\n".join([f"Q: {entry['question']}\nA: {entry['answer']}" for entry in self.chat_history])
        
        # Fetch relevant context from the vector store based on the current question
        if self.chat_history:
            current_question = self.chat_history[-1]['question']
            context_from_db = self.vectorstore_retriever.get_relevant_documents(current_question)
        else:
            context_from_db = self.vectorstore_retriever.get_relevant_documents("")

        # Convert the list of context documents into a string
        context_str = "\n".join([doc.page_content for doc in context_from_db])

        # Combine both chat history and the document context
        combined_context = f"Chat history:\n{chat_context}\n\nContext from the document:\n{context_str}"
        
        return combined_context


    def invoke(self, input_dict):
        question = input_dict.get("question")
        
        if (self.guardrails(question) == False):
          print("The user entered in a bad question (this is only a message to debug)\n")
          return {'query': question, 'context': 'No context.', 'result': 'Sorry, please ask another question '}
        combined_context = self.build_combined_context()

        result = self.qa_chain.invoke({
            "query": question,
            "context": combined_context
        })

        self.update_chat_history(question, result['result'])

        if (self.guardrails(result['result']) == False):
            print("the LLM has generated a bad resposne (this is a message to debug)")
            return {'query': question, 'context': 'No context.', 'result': 'Sorry, please ask another question '}
        return result

    def guardrails(self, input):
      #if guardrails return true send back whatever the input is,
      #else send back an error message
      try:
        guard.validate(input)
        return True
      except Exception as e:
        print(e)
        return False
      
#Setup GenerateStudyPlan pipeline
class GenerateStudyPlan:

    def invoke(self, input_dict):
        # Load data from PDF
        file_path = "tutor_textbook.pdf"

        loader = PyPDFLoader(file_path)
        data = loader.load()

        # Combine text from Document into one string for question generation
        text_question_gen = ''
        for page in data:
            text_question_gen += page.page_content

        # Initialize Text Splitter for question generation
        text_splitter_question_gen = TokenTextSplitter(model_name="gpt-3.5-turbo", chunk_size=10000, chunk_overlap=200)

        # Split text into chunks for question generation
        text_chunks_question_gen = text_splitter_question_gen.split_text(text_question_gen)

        # Convert chunks into Documents for question generation
        docs_question_gen = [Document(page_content=t) for t in text_chunks_question_gen]

        # Initialize Text Splitter for answer generation
        text_splitter_answer_gen = TokenTextSplitter(model_name="gpt-3.5-turbo", chunk_size=1000, chunk_overlap=100)

        # Split documents into chunks for answer generation
        docs_answer_gen = text_splitter_answer_gen.split_documents(docs_question_gen)

        # Initialize Large Language Model for question generation
        llm_question_gen = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"),temperature=0.4, model="gpt-3.5-turbo")

        # Initialize question generation chain
        question_gen_chain = load_summarize_chain(llm = llm_question_gen, chain_type = "refine", verbose = True, question_prompt=PROMPT_QUESTIONS, refine_prompt=REFINE_PROMPT_QUESTIONS)

        # Run question generation chain
        questions = question_gen_chain.run(docs_question_gen)

        # Initialize Large Language Model for answer generation
        llm_answer_gen = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"),temperature=0.1, model="gpt-3.5-turbo")

        # Create vector database for answer generation
        embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

        # Initialize vector store for answer generation
        vector_store = Chroma.from_documents(docs_answer_gen, embeddings)

        # Initialize retrieval chain for answer generation
        answer_gen_chain = RetrievalQA.from_chain_type(llm=llm_answer_gen, chain_type="stuff", retriever=vector_store.as_retriever(k=2))

        # Split generated questions into a list of questions
        question_list = questions.split("\\n")

        # Answer each question and save to a file
        for question in question_list:
            # print("Question: ", question)
            answer = answer_gen_chain.run(question)
            # print("Answer: ", answer)
            # print("--------------------------------------------------\\n\\n")
            # Save answer to file
            with open("answers.txt", "a") as f:
                f.write("Question: " + question + "\\n")
                f.write("Answer: " + answer + "\\n")
                f.write("--------------------------------------------------\\n\\n")
    

#Setup Summarizer pipeline
class Summarizer:
    def __init__(self):
        self.doc = "tutor_textbook.pdf"
        self.loader = PyPDFLoader(self.doc)

        # Load the document and store it in the 'data' variable
        self.data = self.loader.load_and_split()

        self.embeddings = OpenAIEmbeddings()
        self.vectordb = Chroma.from_documents(self.data, embedding=self.embeddings,
                                 persist_directory=".")

        # Initialize a language model with ChatOpenAI
        self.llm = ChatOpenAI(model_name= 'gpt-3.5-turbo', temperature=0.6)

        #Setup a prompt template
        template = """\
            You are an assistant for summarizing the textbook.

        If the user has given a topic or topics, make the summary more focused on those topics.

        Give a nice and detailed summary. 

        Make sure to include all main points given. There can be no points missed 
        out. The output limit is 5 sentences minimum to 15 sentences maximum. 
        Use the textbook given ONLY, and give every point that is important and
        summarize those. 

        If the user states a specific chapter, ONLY GIVE THEM THE SUMMARY OF THAT
        SPECIFIC CHAPTER. 

        ALWAYS DOUBLE CHECK YOUR RESPONSE AND BE ACCURATE. DON'T TALK ABOUT ANOTHER
        TOPIC, ONLY GIVE INFORMATION YOU OBTAINED FROM THAT SPECIFIC TOPIC.

        If no query is given from the user, SUMMARIZE THE WHOLE TEXTBOOK GIVEN.

        Instead of giving a general summary of a number of pages, write bullet 
        points in terms of notes that the user can take into their notebook. Make
        sure to address the main points of the summary and don't generalize 
        specific topics. Just give bullet point summaries.

        USE MARKDOWN FORMATTING FOR BULLET POINTS. 

        Question: {question}

        Context: {context}

        Answer:

        """

        prompt = ChatPromptTemplate.from_template(template)

        chain_type_kwargs = {"prompt": prompt}


        # 1. Vectorstore-based retriever
        self.vectorstore_retriever = self.vectordb.as_retriever()

        # Initialize a RetrievalQA chain with the language model and vector database retriever
        self.qa_chain = RetrievalQA.from_chain_type(self.llm, retriever= self.vectorstore_retriever, chain_type_kwargs=chain_type_kwargs)
        self.chat_history = []  # Initialize chat history

    def update_chat_history(self, question, answer):
        self.chat_history.append({"question": question, "answer": answer})

    def build_combined_context(self):
        """Combine chat history and document context."""
        # Combine all previous chat history
        chat_context = "\n".join([f"Q: {entry['question']}\nA: {entry['answer']}" for entry in self.chat_history])
        
        # Fetch relevant context from the vector store based on the current question
        if self.chat_history:
            current_question = self.chat_history[-1]['question']
            context_from_db = self.vectorstore_retriever.get_relevant_documents(current_question)
        else:
            context_from_db = self.vectorstore_retriever.get_relevant_documents("")

        # Convert the list of context documents into a string
        context_str = "\n".join([doc.page_content for doc in context_from_db])

        # Combine both chat history and the document context
        combined_context = f"Chat history:\n{chat_context}\n\nContext from the document:\n{context_str}"
        
        return combined_context


    def invoke(self, input_dict):
        question = input_dict.get("question")
        
        if (self.guardrails(question) == False):
          print("It has failed (this is only a message to debug)\n")
          return {'query': question, 'context': 'No context.', 'result': 'Sorry, please ask another question '}
        
        combined_context = self.build_combined_context()

        result = self.qa_chain.invoke({
            "query": question,
            "context": combined_context
        })

        self.update_chat_history(question, result['result'])

        if (self.guardrails(result['result']) == False):
            print("the LLM has generated a bad resposne (this is a message to debug)")
            return {'query': question, 'context': 'No context.', 'result': 'Sorry, please ask another question '}

        return result

    def guardrails(self, input):
      #if guardrails return true send back whatever the input is,
      #else send back an error message
      try:
        guard.validate(input)
        return True
      except Exception as e:
        print(e)
        return False

#Setup Quiz pipeline
class QuizAI:
    def __init__(self):
        self.doc = "tutor_textbook.pdf"
        self.loader = PyPDFLoader(self.doc)

        # Load the document and store it in the 'data' variable
        self.data = self.loader.load_and_split()

        self.embeddings = OpenAIEmbeddings()
        self.vectordb = Chroma.from_documents(self.data, embedding=self.embeddings,
                                 persist_directory=".")

        # Initialize a language model with ChatOpenAI
        self.llm = ChatOpenAI(model_name= 'gpt-4o', temperature=0.6)

        #Setup a prompt template
        template = """\
            You are an assistant for quizzing topics on the textbook.

         Read through the texbook and generate quiz questions. You have to create 
        10 different quiz questions. They may be on the same topic or different 
        topics. If the user has given a query, focus the topic on that query. If 
        no query has been provided, use the whole textbook and pick topics at random.

        Your response format will ALWAYS be this:
        *** 
        Query: (THE USER'S QUERY GOES HERE)
        Quiz Question: (YOUR GENERATED QUIZ QUESTION GOES HERE)
        Quiz Correct Answer: (YOUR GENERATED QUIZ ANSWER GOES HERE)
        Quiz Incorrect Answer: (YOUR GENERATED QUIZ ANSWER GOES HERE)
        Quiz Another Incorrect Answer: (YOUR GENERATED QUIZ ANSWER GOES HERE)
        Quiz Final Tricky Incorrect Answer: (YOUR GENERATED QUIZ ANSWER GOES HERE)
        Quiz Answer Explanation: (YOUR GENERATED EXPLANATION GOES HERE)
        ***

        The parenthesis inside the format are where you plug in the parameters if
        given. Say a user gives a query of "Chapter 1". You plug in the parameter
        with *** Query: Chapter 1 ***. If no query is given, put in the query field,
        NO QUERY. 

        Query: {question}

        Context: {context}

        """

        prompt = ChatPromptTemplate.from_template(template)

        chain_type_kwargs = {"prompt": prompt}


        # 1. Vectorstore-based retriever
        self.vectorstore_retriever = self.vectordb.as_retriever()

        # Initialize a RetrievalQA chain with the language model and vector database retriever
        self.qa_chain = RetrievalQA.from_chain_type(self.llm, retriever= self.vectorstore_retriever, chain_type_kwargs=chain_type_kwargs)
        self.chat_history = []  # Initialize chat history

    def update_chat_history(self, question, answer):
        self.chat_history.append({"question": question, "answer": answer})

    def build_combined_context(self):
        """Combine chat history and document context."""
        # Combine all previous chat history
        chat_context = "\n".join([f"Q: {entry['question']}\nA: {entry['answer']}" for entry in self.chat_history])
        
        # Fetch relevant context from the vector store based on the current question
        if self.chat_history:
            current_question = self.chat_history[-1]['question']
            context_from_db = self.vectorstore_retriever.get_relevant_documents(current_question)
        else:
            context_from_db = self.vectorstore_retriever.get_relevant_documents("")

        # Convert the list of context documents into a string
        context_str = "\n".join([doc.page_content for doc in context_from_db])

        # Combine both chat history and the document context
        combined_context = f"Chat history:\n{chat_context}\n\nContext from the document:\n{context_str}"
        
        return combined_context


    def invoke(self, input_dict):
        question = input_dict.get("question")
        
        # COMMENTED OUT RIGHT NOW AS IT GIVES FALSE POSITIVES, NEEDS MORE TESTING
        if (self.guardrails(question) == False):
          print("It has failed (this is only a message to debug)\n")
          return {'query': question, 'context': 'No context.', 'result': 'Sorry, please ask another question '}
        
        combined_context = self.build_combined_context()

        result = self.qa_chain.invoke({
            "query": question,
            "context": combined_context
        })

        self.update_chat_history(question, result['result'])

        if (self.guardrails(result['result']) == False):
            print("the LLM has generated a bad resposne (this is a message to debug)")
            return {'query': question, 'context': 'No context.', 'result': 'Sorry, please ask another question '}

        return result

    def guardrails(self, input):
      #if guardrails return true send back whatever the input is,
      #else send back an error message
      try:
        guard.validate(input)
        return True
      except Exception as e:
        print(e)
        return False

from flask import Flask, render_template, request, redirect, url_for
import markdown
from bs4 import BeautifulSoup
import json

filepath = "./tutor_textbook.pdf"
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx', 'png', 'jpg', 'jpeg', 'gif'}
QUIZGENERATED = False
app = Flask(__name__)

@app.route("/",methods=["GET"])
def index():
   return render_template("index.html")

@app.route("/tutor-ai", methods=["GET", "POST"])
def tutor_ai():
    global url_data, prompt_data  # Access global variables

    if request.method == "POST":
        url_data = request.form.get("url")
        print("URL: ", url_data)
        if 'file' not in request.files:
            print('No file uploaded!')
        else:
          file = request.files['file']
          file.save(filepath)
          print("File saved:", filepath)
        if (url_data != ""):
            subprocess.check_call("curl", url_data, ">", "tutor_textbook.pdf")
        print("File: ",file)
        prompt_data = request.form.get("prompt")
        base_qa_pipeline = BaseQAPipeline()
        result = base_qa_pipeline.invoke({'question' : prompt_data})
        print(result)
        return render_template("tutor-ai.html", result=result)

    return render_template("tutor-ai.html")

@app.route("/summarizer", methods=["GET", "POST"])
def summarizer():
    global url_data, prompt_data  # Access global variables

    if request.method == "POST":
        url_data = request.form.get("url")
        print("URL: ", url_data)
        if 'file' not in request.files:
            print('No file uploaded!')
        else:
          file = request.files['file']
          file.save(filepath)
          print("File saved:", filepath)
        if (url_data != ""):
            subprocess.check_call("curl", url_data, ">", "tutor_textbook.pdf")
        print("File: ",file)
        prompt_data = request.form.get("prompt")
        base_qa_pipeline = Summarizer()
        result = base_qa_pipeline.invoke({'question' : prompt_data})
        print(result)
        return render_template("summarizer.html", result=result)
    return render_template("summarizer.html")

@app.route('/how-it-works', methods=['GET'])
def how_it_works():
    return render_template('how-it-works.html')

@app.route('/study-plan', methods=['GET', "POST"])
def generate_plan():
    if request.method == "POST":
        if 'file' not in request.files:
            print('No file uploaded!')
        else:
          file = request.files['file']
          file.save(filepath)
          print("File saved:", filepath)
        print("File: ",file)
        prompt_data = request.form.get("prompt")
        generate_plan = GenerateStudyPlan()
        result = generate_plan.invoke({'question' : prompt_data})
        result['result'] = markdown.markdown(result['result'])
        print(result)
        return render_template("study-plan.html", result=result)

    return render_template("study-plan.html")

@app.route('/quiz', methods=['GET', "POST"])
def quiz_maker():
    global QUIZGENERATED
    if request.method == "POST":
        if 'file' not in request.files:
            print('No file uploaded!')
        else:
          file = request.files['file']
          file.save(filepath)
          print("File saved:", filepath)
        print("File: ",file)
        prompt_data = request.form.get("prompt")
        quiz = QuizAI()
        print(prompt_data)
        result = quiz.invoke({'question' : prompt_data})
        QUIZGENERATED = True
        print(result)

        # Parse the result HTML to extract the question, answer, and explanation
        soup = BeautifulSoup(result['result'], 'html.parser')
        
        # Extracting relevant parts
        result_text = result['result']
        quizzes = result_text.split("***")
        parsed_quizzes = []
        quiz_filepath = "static/quiz-data.json"

        for quiz in quizzes:
            lines = [line.strip() for line in quiz.split("\n") if line.strip()]
            if len(lines) < 7:
                continue  # Skip invalid or incomplete entries
            try:
                # Extracting quiz data
                question = lines[1].split(": ", 1)[1]
                correct_answer = lines[2].split(": ", 1)[1]
                incorrect_answers = [
                    lines[3].split(": ", 1)[1],
                    lines[4].split(": ", 1)[1],
                    lines[5].split(": ", 1)[1],
                ]
                explanation = lines[6].split(": ", 1)[1]

                # Add parsed quiz to the list
                parsed_quizzes.append({
                    "question": question,
                    "correct_answer": correct_answer,
                    "options": [correct_answer] + incorrect_answers,
                    "explanation": explanation
                })
            except Exception as e:
                print(f"Error parsing quiz: {quiz}\nError: {e}")


            # Write to a JSON file
            with open(quiz_filepath, "w") as quiz_file:
                json.dump(parsed_quizzes, quiz_file, indent=4)

            print("Quiz data successfully parsed and saved to 'quiz-data.json'!")

        # Render the template with the file path passed as a parameter
        return render_template('generated-quiz.html', quiz_file=quiz_filepath)

    return render_template("quiz.html")

@app.route('/generated-quiz', methods=['GET'])
def generated_quiz():
    global QUIZGENERATED
    QUIZGENERATED = True
    if QUIZGENERATED:
        # The AI result data you provided
        ai_result = {
            'query': 'Quiz me on Chapter 1.',
            'context': 'Chat history:\n\n\nContext from the document:\nThis page intentionally left blank\nThis page intentionally left blank\nThis page intentionally left blank\nThis page intentionally left blank',
            'result': '''<hr />\n<p>Query: Quiz me on Chapter 1</p>
                         <p>Quiz Question: What is the main topic covered in Chapter 1 of the textbook?</p>
                         <p>Quiz Answer: The main topic covered in Chapter 1 is introduction to the subject.</p>
                         <p>Quiz Answer Explanation: Chapter 1 typically serves as an introduction to the subject matter of the textbook, providing an overview of what will be discussed in the following chapters.</p>
                         <hr />'''
        }

        # Parse the result HTML to extract the question, answer, and explanation
        soup = BeautifulSoup(ai_result['result'], 'html.parser')

        # Extract the relevant parts
        try:
            # Ensure we have enough <p> elements
            paragraphs = soup.find_all('p')
            if len(paragraphs) >= 4:
                question = paragraphs[1].text.split(':')[1].strip()
                answer = paragraphs[2].text.split(':')[1].strip()
                explanation = paragraphs[3].text.split(':')[1].strip()
            else:
                raise ValueError("Insufficient <p> tags in the result HTML.")
        except Exception as e:
            # Handle the case where there are not enough <p> tags or the content is malformed
            return f"Error parsing the quiz result: {str(e)}", 500

        # Pass the parsed data to the template
        return render_template('generated-quiz.html', question=question, answer=answer, explanation=explanation)

    return "404 Not Found", 404

if __name__ == "__main__":
    #from waitress import serve
    #serve(app, host="0.0.0.0", port=8081)
    # above code is for SERVER
    #below code right now is to debug
    print("Server is running...")
    app.run(port=8081,debug=True)
    print("Stopping server...") 