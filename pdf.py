import streamlit as st
import pdfplumber
import os
import google.generativeai as genai
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatGoogleGenerativeAI
from langchain.chains import LLMChain
from dotenv import load_dotenv
import re
from io import BytesIO  # For handling in-memory files

# Load environment variables
load_dotenv()

# Google Gemini API Setup
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

# Initialize the ChatGoogleGenerativeAI model
model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)

# Function to extract text from PDF using pdfplumber
def extract_pdf_text(pdf_file):
    try:
        # Open the uploaded PDF file using pdfplumber (in memory)
        with pdfplumber.open(pdf_file) as pdf:
            text = ""
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            return text
    except Exception as e:
        return f"Error extracting PDF text: {str(e)}"

# Function to extract questions from the text using regex
def extract_questions(text):
    # This regex captures lines that end with a question mark
    question_pattern = r"([^\n]+?\?)"
    questions = re.findall(question_pattern, text)
    return questions

# Function to generate related questions using the ChatGoogleGenerativeAI model
def generate_related_questions_with_gemini(question, num_questions, context):
    try:
        # Define the enhanced prompt template
        prompt_template = (
            "Context:\n{context}\n\n"
            "Question:\n{question}\n\n"
            "Based on the above question and context, generate {num_questions} semantically related questions. "
            "These should be new questions that explore similar topics or perspectives, "
            "but should not be the same as the original question. Reword, reframe, or generate new questions that are "
            "related but not identical to the original question."
        )
        
        # Create the PromptTemplate
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question", "num_questions"])
        
        # Create the LLMChain with the model and the prompt
        chain = LLMChain(llm=model, prompt=prompt)

        # Run the chain with the input data
        response = chain.run({"context": context, "question": question, "num_questions": num_questions})
        
        # Clean and split the response into individual questions
        related_questions = re.split(r'\n|\d+\.', response)
        related_questions = [q.strip() for q in related_questions if q.strip().endswith('?')]

        return related_questions[:num_questions]  # Ensure exact count
    except Exception as e:
        return [f"Error generating questions: {str(e)}"]

# Streamlit app
def app():
    st.title("Generate Semantically Related Questions from PDF (Using Google Gemini)")

    # File uploader for the PDF
    uploaded_file = st.file_uploader("Upload a PDF containing questions", type=["pdf"])

    # If a file is uploaded
    if uploaded_file is not None:
        # Pass the uploaded file as a BytesIO object to pdfplumber
        pdf_text = extract_pdf_text(uploaded_file)

        # Handle case where no text is extracted
        if not pdf_text or "Error" in pdf_text:
            st.write(f"Error: {pdf_text}")
            return

        # Extract questions from the PDF text
        questions = extract_questions(pdf_text)
        if not questions:
            st.write("No questions found in the document.")
            return

        # Input for number of related questions to be generated
        num_questions = st.number_input(
            "Enter the number of related questions to generate", 
            min_value=1, 
            max_value=10, 
            value=3
        )

        # Submit button to trigger the generation of related questions
        if st.button("Generate Related Questions"):
            all_related_questions = []
            for idx, question in enumerate(questions, 1):
                st.write(f"Processing Question {idx}: {question}")
                related_questions = generate_related_questions_with_gemini(question, num_questions, pdf_text)
                all_related_questions.extend(related_questions)

            # Remove duplicates and exact matches
            unique_related_questions = []
            seen = set()
            for q in all_related_questions:
                q_lower = q.lower()
                if q_lower not in seen and not any(q_lower == original_q.lower() for original_q in questions):
                    unique_related_questions.append(q)
                    seen.add(q_lower)
                if len(unique_related_questions) >= num_questions:
                    break

            # Display related questions
            st.subheader("Semantically Related Questions:")
            if unique_related_questions:
                for idx, related_question in enumerate(unique_related_questions, 1):
                    st.write(f"{idx}. {related_question}")
            else:
                st.write("No related questions found.")

if __name__ == "__main__":
    app()
