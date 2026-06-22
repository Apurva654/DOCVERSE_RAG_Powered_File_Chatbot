#    ""DOCVERSE""- 🤖🤖🤖 RAG-Powered File Chatbot 

## clearLocal Document Question Answering System

This project is a simple terminal chatbot that answers questions about files on your computer. It reads '.txt' and '.pdf' files from a folder, splits them into chunks, stores searchable embeddings in ChromaDB, retrieves the most relevant chunks for each question, and asks an LLM to answer only from those chunks.

## What The Project Does-📖🔖

1. Loads local '.txt' and '.pdf' files with LangChain document loaders.
2. Splits documents with 'RecursiveCharacterTextSplitter'.
3. Creates embeddings with Hugging Face sentence-transformer embeddings.
4. Stores vectors locally in ChromaDB so indexing is reused between runs.
5. Retrieves the top matching chunks for a user question.
6. Sends those chunks to Gemini through a grounded prompt.
7. Runs an interactive terminal chat loop and shows source files for each answer.

##  Proposed System Architecture:-

The system has seven stages:
1. File loading
2. Chunking
3. Embedding
4. Vector storage
5. Retrieval
6. Prompting
7. Answer generation

## Setup

### Create and activate a virtual environment:

 python -m venv .venv

.venv\Scripts\activate 

### Install dependencies:

pip install -r requirements.txt


## Run With Google API Key And Hugging Face Embeddings

Create a `.env` file, then add your Google API key:

GOOGLE_API_KEY=your_google_api_key_here

## Run:

python src/rag_chatbot.py --docs sample_files --model gemini-2.5-flash --rebuild

## If u have to Use Your Own Folder

Place '.txt' and '.pdf' files in any local folder, then run:

python src/rag_chatbot.py --docs "C:\path\to\your\folder"

To rebuild the vector database after changing files:

python src/rag_chatbot.py --docs "C:\path\to\your\folder" --rebuild
 

## Example Questions And Answers

### Example 1:

You: Where is VJTI located and what  is it famous for?

Assistant:
VJTI is located in Matunga, Mumbai, Maharashtra, India. It is famous for being one of Maharashtra's oldest engineering colleges, having a strong placement record, good industry connections, a strong alumni network, and a high reputation among government engineering colleges.

Sources:
-sample_files\VJTI.txt

### Example 2:

You: what is the use of documentation agent and coder agent in KAIZEN? 

Assistant:
The Documentation agent in KAIZEN is used to keep project documentation synchronized with code changes. The Coder agent is used to implement changes with awareness of the existing codebase.

Sources:
- sample_files\kaizen_overview.txt

## Project Structure

├── src/
│   └── rag_chatbot.py
├── sample_files/
│   ├── kaizen_overview.txt
│   └── VJTI.txt
├── requirements.txt
├── README.md


### THANK YOU   o(=•ェ•=)m