"""
                                    DEPENDENCIES🔎
"""
import argparse   
#Imports Python command-line parser.
# allows changing folders/models/settings without editing code 
# or hardcoding the values again.
import os
#Used to access environment variables.
import shutil
#Provides file operations like deleting folders.
from pathlib import Path
#used for Modern path handling.
from typing import Iterable, List
#used for type hints.
from uuid import uuid4
#Used to create a fresh ChromaDB folder if Windows locks the old one.
from dotenv import load_dotenv
#Loads variables from .env
from langchain_core.documents import Document
#Document object used by LangChain.
from langchain_core.prompts import ChatPromptTemplate
#Creates prompt templates.
from langchain_chroma import Chroma
# Imports Chroma vector database for storing the embeddings
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
#used for loading multiple files from folder, pdf files, text files.
from langchain_google_genai import ChatGoogleGenerativeAI
# Creates Gemini chat models.
from langchain_huggingface import HuggingFaceEmbeddings
# Creates Hugging Face embeddings.
from langchain_text_splitters import RecursiveCharacterTextSplitter
# Splits large documents into smaller chunks

#----------------------------------------------------------------------------------------
"""                      STEP 1-   DOCUMENT LOADING 📚
    Load all supported files from a folder using LangChain document loaders.
"""
def load_documents(folder: str) -> List[Document]:
    
    folder_path = Path(folder)
    if not folder_path.exists() or not folder_path.is_dir():
        raise ValueError(f"Document folder does not exist: {folder}")

    documents: List[Document] = []

    txt_loader = DirectoryLoader(
        str(folder_path),
        glob="**/*.txt",
        loader_cls=TextLoader,               #Specifies loader type.
        loader_kwargs={"encoding": "utf-8"}, #Encoding for text files.
        show_progress=True,
    )
    pdf_loader = DirectoryLoader(
        str(folder_path),
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True,
    )

    for loader in (txt_loader, pdf_loader):
        documents.extend(loader.load())       #Loads all files and combines them.

    return documents

#----------------------------------------------------------------------------------------
"""                     STEP 2- DOCUMENT SPLITTING ⛓️‍💥
        Break large text into smaller pieces to optimize computational resources.
"""
def split_documents(
    documents: Iterable[Document],
    chunk_size: int = 900,
    chunk_overlap: int = 150,
) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter( #Prevents loss of context.
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(list(documents))

#----------------------------------------------------------------------------------------
"""                     STEP 3- BUILDING EMBEDDINGS 🐾
      used to convert text into vector databases for semantic search.
"""
def build_embeddings(model: str):
    cache_folder = Path(".cache") / "huggingface"
    cache_folder.mkdir(parents=True, exist_ok=True)
    return HuggingFaceEmbeddings(model_name=model, cache_folder=str(cache_folder))

#----------------------------------------------------------------------------------------
"""                 STEP 4-STORING AND LOADING EMBEDDINGS INTO CHROMA DB 👾
      it is responsible for reusing or rebuilding the Chroma vector database
"""
def build_or_load_vector_store(
    docs_folder: str,
    persist_dir: str,
    rebuild: bool,
    embedding_model: str,
) -> Chroma:
    embeddings = build_embeddings(embedding_model)
    persist_path = Path(persist_dir)
    collection_name = "rag_hf_embeddings_google_llm"
    
    if rebuild and persist_path.exists():
    #If user requested rebuilding AND the database folder already exists
        try:
            shutil.rmtree(persist_path) #Deletes old DB
        except PermissionError:
        #This catches an error if the folder cannot be deleted.
            persist_path = Path(f"{persist_dir}_fresh_{uuid4().hex[:8]}")
        #Instead of failing, it creates a new folder name.
            print(
                f"Could not clear locked ChromaDB folder. "
                f"Using fresh index folder instead: {persist_path}"
            )
        persist_dir = str(persist_path)
        #because Chroma expects a string path.

    if persist_path.exists() and any(persist_path.iterdir()) and not rebuild:
        #If DB folder exists AND folder contains data AND rebuild wasn't requested
        print(f"Using existing ChromaDB index at: {persist_dir}")
         #This loads the already-created database.
        return Chroma(
            collection_name=collection_name,
            persist_directory=persist_dir,
            embedding_function=embeddings,
        )

    print("Loading files...") #Loads files.
    documents = load_documents(docs_folder)
    if not documents:
        raise ValueError(
            f"No .txt or .pdf files found in {docs_folder}. Add files and try again."
        )

    print(f"Loaded {len(documents)} document pages/files.") #Chunks them.
    chunks = split_documents(documents)
    print(f"Created {len(chunks)} chunks.")

    print("Building ChromaDB index...") # chunking, embedding, and storing of vectors
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
        collection_name=collection_name,
    )
    print(f"Index saved at: {persist_dir}")
    return vector_store
#----------------------------------------------------------------------------------------
"""                   STEP 5- CREATING LANGUAGE MODELS 🧑‍💻
      used to Create the language model instance for generating responses.
"""
def build_llm(model: str):
    return ChatGoogleGenerativeAI(model=model, temperature=0)
    # Low temperature keeps answers deterministic.

#----------------------------------------------------------------------------------------
"""                   STEP 6- SOURCE TRACKING🗼
      used to extracts document metadata so the chatbot can show exactly 
      which files/pages were used to generate the answer,
      so the RAG system becomes traceable and grounded
"""
def format_sources(documents: List[Document]) -> str:
    source_lines = []
    seen = set()
    for doc in documents:
        source = doc.metadata.get("source", "unknown source")
        page = doc.metadata.get("page")
        label = f"{source}, page {page + 1}" if isinstance(page, int) else source
        if label not in seen:
 #since multiple chunks may come from the same page, we use set to get cleaner output
            source_lines.append(label)
            seen.add(label)
    return "\n".join(f"- {line}" for line in source_lines)
#----------------------------------------------------------------------------------------

"""               STEP 7- RETRIEVAL AND ANSWER GENERATION🤺
     This is the core RAG pipeline, it retrieves relevant document chunks,
       builds context, sends it to the LLM, and returns the answer with sources.
"""
def answer_question(vector_store: Chroma, llm, question: str, top_k: int) -> str:
    retriever = vector_store.as_retriever(search_kwargs={"k": top_k})
    #Searches vector DB and fetches relevant chunks.
    retrieved_docs = retriever.invoke(question)

    context = "\n\n".join( #Context Assembly
        f"Source: {doc.metadata.get('source', 'unknown')}\n{doc.page_content}"
        for doc in retrieved_docs
        #Combines retrieved chunks into one large text block
    )
    prompt = ChatPromptTemplate.from_messages( 
        [ #creates instructions for the LLM
            (
                "system",
                "You are a file-grounded assistant. Answer only from the provided "
                "context. If the context does not contain the answer, say you do "
                "not know based on the files. Keep the answer concise.",
            ),
            (
                "human",
                "Context:\n{context}\n\nQuestion: {question}\n\nGrounded answer:",
            ),
        ]
    )
    chain = prompt | llm   #Pipeline
    response = chain.invoke({"context": context, "question": question})
    answer = response.content if hasattr(response, "content") else str(response)

    return f"{answer.strip()}\n\nSources:\n{format_sources(retrieved_docs)}"

#----------------------------------------------------------------------------------------
"""                   STEP 8- CREATING THE CHAT ENGINE 🧑‍💻
      used to Create the language model instance for generating responses.
"""

def run_chat(args) -> None:
    load_dotenv() #Loads API keys.
    if not os.getenv("GOOGLE_API_KEY"):  #Stops execution if API key is missing.
        raise ValueError("GOOGLE_API_KEY is required. Add it to your .env file.")

    vector_store = build_or_load_vector_store( 
        #Loads an existing vector DB or creates a new one
        docs_folder=args.docs,
        persist_dir=args.persist_dir,
        rebuild=args.rebuild,
        embedding_model=args.embedding_model,
    )


    llm = build_llm(args.model) #Creates the Gemini LLM object.
    print("\n LESS GOO!!🥳 YOUR RAG file chatbot is ready.")
    print("Ask a question about your files. Type 'exit' or 'quit' to stop.\n")

    while True:
        question = input("You: ").strip()
        if question.lower() in {"exit", "quit"}:
            print("Goodbye! see you soon🫰")
            break
        if not question:   #Ignores blank input.
            continue

        try:
            print("\nAssistant:")
            print(answer_question(vector_store, llm, question, args.top_k))
            print()
        except Exception as exc:
            print(f"Error: {exc}\n")
#----------------------------------------------------------------------------------------
"""                   STEP 9- ARGUMENT PARSING 🕸️
      used to Read user inputs from the command line and 
      converts them into a configuration object for the chatbot.
"""
def parse_args():
    parser = argparse.ArgumentParser(description="RAG-powered local file chatbot")

    #Defines the folder from which files will be loaded
    parser.add_argument("--docs", default="sample_files", help="Folder containing .txt/.pdf files")

    # Defines where the Chroma vector database is stored.
    parser.add_argument("--persist-dir", default="chroma_db_hf_google", help="ChromaDB storage folder")

    # Defines which Gemini model should be used.
    parser.add_argument(
        "--model",
        default=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        help="Gemini model name, for example gemini-2.5-flash",)
    
    # Defines which embedding model generates vectors.
    parser.add_argument(
        "--embedding-model",
        default=os.getenv("HF_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        help="Hugging Face sentence-transformer embedding model",)
    
    #Defines how many chunks the retriever should return.
    parser.add_argument("--top-k", type=int, default=4, help="Number of chunks to retrieve")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild the ChromaDB index")
    return parser.parse_args()

#----------------------------------------------------------------------------------------
"""                  MAIN() FUNCTION   
"""
if __name__ == "__main__":
    run_chat(parse_args())
#                                AND WE ARE FINALLY DONE!!🥳🥳
#----------------------------------------------------------------------------------------------