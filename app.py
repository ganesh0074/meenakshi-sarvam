from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from sentence_transformers import SentenceTransformer
import faiss
import fitz  # PyMuPDF for PDF text extraction
import numpy as np
import os
import torch

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')


from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteriaList, StoppingCriteria

model = AutoModelForCausalLM.from_pretrained("HPAI-BSC/Llama3-Aloe-8B-Alpha", return_dict=True)
tokenizer = AutoTokenizer.from_pretrained("HPAI-BSC/Llama3-Aloe-8B-Alpha")

# Store embeddings and documents
index = None
documents = []

class StopAtSentenceEnd(StoppingCriteria):
    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores):
        decoded_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        return decoded_text.endswith('.') or decoded_text.endswith('?') or decoded_text.endswith('!')

def answer_question(question, context):

    max_context_length = 64
    truncated_context = context[:max_context_length]

    # Prepare inputs for LLAMA
    inputs = tokenizer(question + " " + truncated_context, return_tensors="pt")

    # Set max_new_tokens to control output generation and pad_token_id to EOS
    max_new_tokens = 128
    stopping_criteria = StoppingCriteriaList([StopAtSentenceEnd(tokenizer)])

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,  # Controls how many tokens to generate
            pad_token_id=tokenizer.eos_token_id,  # Padding with EOS token
            stopping_criteria=stopping_criteria  # Stops at sentence end
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


# Agent function to determine action based on user query
def agent_response(query):
    if query.lower() in ["hello", "hi", "hey"]:
        return "Hello! How can I assist you today?"
    elif "calculate" in query.lower():
        return "Please provide the calculation you need (e.g., 2 + 2)."
    else:
        return None  # Indicates to check VectorDB



def calculate(expression):
    try:
        result = eval(expression)
        return f"The result of {expression} is {result}."
    except Exception as e:
        return "Error in calculation."


@app.route('/upload_pdf/', methods=['POST'])
def upload_pdf():
    global index, documents

    if 'pdf_file' not in request.files:
        return render_template('index.html', error="No file part")

    file = request.files['pdf_file']

    if file.filename == '' or not allowed_file(file.filename):
        return render_template('index.html', error="No valid PDF file selected")

    # Save the uploaded PDF to a temporary location
    filename = secure_filename(file.filename)
    pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(pdf_path)

    # Extract text from the PDF
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    documents = text.split('\n')  # Split the text into sentences or paragraphs

    # Create embeddings for the documents
    embeddings = embedding_model.encode(documents, show_progress_bar=True)

    # Create a FAISS index
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings).astype('float32'))


    os.remove(pdf_path)

    return render_template('index.html', message="PDF uploaded and processed successfully.")


@app.route('/query/', methods=['GET'])
def query():
    global index, documents

    if index is None or not documents:
        return render_template('index.html', error="No documents available. Please upload a PDF first.")

    user_query = request.args.get('text')
    if not user_query:
        return render_template('index.html', error="No query provided.")


    agent_output = agent_response(user_query)
    if agent_output:
        return render_template('index.html', message=agent_output)


    query_embedding = embedding_model.encode([user_query])


    distances, indices = index.search(np.array(query_embedding).astype('float32'), k=1)

    # Generate response based on closest document
    if distances[0][0] < 1.0:  # Adjust the threshold as needed
        context = documents[indices[0][0]]
        answer = answer_question(user_query, context)
    else:
        answer = "No relevant information found."

    return render_template('index.html', answer=answer)


@app.route('/agent/', methods=['GET'])
def agent():
    user_query = request.args.get('text')


    agent_output = agent_response(user_query)
    if agent_output:
        return render_template('index.html', message=agent_output)


    if "calculate" in user_query.lower():
        expression = user_query.split("calculate")[-1].strip()
        calculation_result = calculate(expression)
        return render_template('index.html', message=calculation_result)

    return render_template('index.html', message="I'm sorry, I didn't understand that.")


@app.route('/')
def index():
    return render_template('index.html')


if __name__ == '__main__':
    # Create upload directory if it does not exist
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])

    app.run(debug=True)