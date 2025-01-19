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
        on_fail="exception"
    ),

    NSFWText(
        threshold=0.8,
        validation_method="sentence"
    ),

    ProfanityFree(
        on_fail = "exception"
    ),

    LogicCheck(
        model="gpt-3.5-turbo",
        on_fail="exception"
    ),

    MentionsDrugs(
        on_fail = "exception"
    ),

    PolitenessCheck(
        llm_callable="gpt-3.5-turbo",
        on_fail = "exception"
    ),

    ToxicLanguage(
        threshold=0.5,
        validation_method="sentence",
        on_fail="exception"
    )

)

try:
  guard.validate("I hate you!")
  print("hello")
except Exception as e:
  print(e)

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
        
        # COMMENTED OUT RIGHT NOW AS IT GIVES FALSE POSITIVES, NEEDS MORE TESTING
        # if (self.guardrails(question) == False):
        #   print("It has failed (this is only a message to debug)\n")
        #   return {'query': 'What is an EDR?', 'context': 'No context.', 'result': 'Sorry, please ask another question '}
        combined_context = self.build_combined_context()

        result = self.qa_chain.invoke({
            "query": question,
            "context": combined_context
        })

        self.update_chat_history(question, result['result'])
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
            You are an assistant for generating study plans on a singular subject.

        Use the following pieces of retrieved context to answer the question.

        If the user has given a topic to study or topics that they need focus on,
        make the plan more focused on those topics.

        Give a nice and detailed study plan. If the user doesnt specify a type of 
        study plan (schedule wise), you can come up with your own, hourly or daily 
        plan, or you can ask the user to give the same instructions but with the 
        study plan of their choice. 

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
        
        # COMMENTED OUT RIGHT NOW AS IT GIVES FALSE POSITIVES, NEEDS MORE TESTING
        # if (self.guardrails(question) == False):
        #   print("It has failed (this is only a message to debug)\n")
        #   return {'query': 'What is an EDR?', 'context': 'No context.', 'result': 'Sorry, please ask another question '}
        combined_context = self.build_combined_context()

        result = self.qa_chain.invoke({
            "query": question,
            "context": combined_context
        })

        self.update_chat_history(question, result['result'])
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
        
        # COMMENTED OUT RIGHT NOW AS IT GIVES FALSE POSITIVES, NEEDS MORE TESTING
        # if (self.guardrails(question) == False):
        #   print("It has failed (this is only a message to debug)\n")
        #   return {'query': 'What is an EDR?', 'context': 'No context.', 'result': 'Sorry, please ask another question '}
        combined_context = self.build_combined_context()

        result = self.qa_chain.invoke({
            "query": question,
            "context": combined_context
        })

        self.update_chat_history(question, result['result'])
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

filepath = "./tutor_textbook.pdf"
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx', 'png', 'jpg', 'jpeg', 'gif'}
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


if __name__ == "__main__":
    #from waitress import serve
    #serve(app, host="0.0.0.0", port=8081)
    # above code is for SERVER
    #below code right now is to debug
    print("Server is running...")
    app.run(port=8081,debug=True)
    print("Stopping server...") 