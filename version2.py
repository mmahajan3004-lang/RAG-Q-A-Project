import os
import uuid
import shutil
import tempfile
import streamlit as st
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_mistralai import MistralAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader
)
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

#Load env
load_dotenv()

if not os.getenv("MISTRAL_API_KEY"):
    st.error("MISTRAL_API_KEY is missing in your .env file.")
    st.stop()

#Page detail
st.set_page_config(
    page_title="AI RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

#Session state
if "chats" not in st.session_state:
    st.session_state.chats = {}


if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = None


if "vector_db" not in st.session_state:
    st.session_state.vector_db = None


if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

#Function state
def create_new_chat():

    chat_id = str(uuid.uuid4())

    st.session_state.chats[chat_id] = {
        "title": "New Chat",
        "messages": []
    }

    st.session_state.current_chat_id = chat_id


def delete_chat(chat_id):

    if chat_id in st.session_state.chats:

        del st.session_state.chats[chat_id]

    if st.session_state.current_chat_id == chat_id:

        st.session_state.current_chat_id = None


def get_current_chat():

    chat_id = st.session_state.current_chat_id

    if chat_id is None:
        return None

    return st.session_state.chats.get(chat_id)

#Create llm 
llm = ChatMistralAI(
    model="mistral-small-latest",
    temperature=0
)
#Create embeddigs
embeddings = MistralAIEmbeddings(
    model="mistral-embed"
)

#Document load 
def load_document(uploaded_file):

    file_extension = uploaded_file.name.split(".")[-1].lower()

    # Temporary file
    temp_dir = tempfile.mkdtemp()

    file_path = os.path.join(
        temp_dir,
        uploaded_file.name
    )

    with open(file_path, "wb") as f:

        f.write(uploaded_file.getbuffer())

    # PDF
    if file_extension == "pdf":

        loader = PyPDFLoader(file_path)

      # TXT
    elif file_extension == "txt":

        loader = TextLoader(
            file_path,
            encoding="utf-8"
        )

    # DOCX
    elif file_extension == "docx":

        loader = Docx2txtLoader(file_path)
    else:

        shutil.rmtree(temp_dir)

        raise ValueError(
            "Unsupported file type."
        )
    
    documents = loader.load()

    shutil.rmtree(temp_dir)

    return documents

#Vector Database
def create_vector_database(files):

    all_documents = []

    # Load all uploaded documents
    for uploaded_file in files:

        documents = load_document(
            uploaded_file
        )

        all_documents.extend(
            documents
        )

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )

    chunks = text_splitter.split_documents(
        all_documents
    )

    # Create unique collection
    collection_name = (
        "documents_"
        + str(uuid.uuid4()).replace("-", "")
    )

    # Create Chroma database
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name
    )
    return vector_db

#Formate doc
def format_documents(documents):

    return "\n\n".join(
        document.page_content
        for document in documents
    )

#RAG chain
def create_rag_chain(vector_db):

    retriever = vector_db.as_retriever(
        search_kwargs={
            "k": 4
        }
    )

#Prompt
    prompt = ChatPromptTemplate.from_template(
        """
You are a helpful AI document assistant.

Answer the user's question using the provided
document context.

Rules:

1. Use the document context whenever possible.
2. Do not invent information.
3. If the answer is not available in the document,
   clearly say that the information was not found
   in the uploaded document.
4. Give a clear and easy-to-understand answer.

Document Context:

{context}

User Question:

{question}

Answer:
"""
    )

    rag_chain = (
        {
            "context": retriever | format_documents,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain

#Side bar
with st.sidebar:

    st.title("🤖 AI Chatbot")

#New chat
    if st.button(
        "➕ New Chat",
        use_container_width=True
    ):

        create_new_chat()

        st.rerun()

    st.divider()

#Doc loader
    st.subheader("📄 Documents")


    uploaded_files = st.file_uploader(
        "Upload your documents",
        type=[
            "pdf",
            "txt",
            "docx"
        ],
        accept_multiple_files=True
    )

    if uploaded_files:

        if st.button(
            "📚 Process Documents",
            use_container_width=True
        ):

            with st.spinner(
                "Processing documents..."
            ):

                try:

                    st.session_state.vector_db = (
                        create_vector_database(
                            uploaded_files
                        )
                    )


                    st.session_state.uploaded_files = [
                        file.name
                        for file in uploaded_files
                    ]


                    st.success(
                        "Documents processed successfully!"
                    )


                except Exception as e:

                    st.error(
                        f"Error: {str(e)}"
                    )

#Show upload pdf
    if st.session_state.uploaded_files:

        st.write("### Uploaded Files")

        for file_name in st.session_state.uploaded_files:

            st.write(
                f"📄 {file_name}"
            )

    st.divider()

#Chat history
    st.subheader("💬 Chat History")


    if not st.session_state.chats:

        st.caption(
            "No previous chats"
        )

    else:

        for chat_id, chat in list(
            st.session_state.chats.items()
        ):

            col1, col2 = st.columns(
                [5, 1]
            )

            with col1:

                title = chat["title"]

                if st.button(
                    f"💬 {title}",
                    key=f"chat_{chat_id}",
                    use_container_width=True
                ):

                    st.session_state.current_chat_id = (
                        chat_id
                    )

                    st.rerun()

            with col2:

                if st.button(
                    "🗑️",
                    key=f"delete_{chat_id}"
                ):

                    delete_chat(
                        chat_id
                    )

                    st.rerun()


    st.divider()

    st.caption(
        "🔒 Chats are stored only for this session."
    )

#Chat box
st.title("🤖 AI Document Chatbot")

st.caption(
    "Upload a document and ask questions about it."
)

#Create first chat auto
if st.session_state.current_chat_id is None:

    create_new_chat()


current_chat = get_current_chat()

#Show wlcm msg
if not current_chat["messages"]:

    st.info(
        """
        👋 Welcome!

        1. Upload a PDF, TXT, or DOCX document.
        2. Click **Process Documents**.
        3. Ask questions about your document.
        """
    )

#Chat history
for message in current_chat["messages"]:

    with st.chat_message(
        message["role"]
    ):

        st.write(
            message["content"]
        )

#Chat input
user_input = st.chat_input(
    "Ask something about your document..."
)


if user_input:

#Save user msg
    current_chat["messages"].append(
        {
            "role": "user",
            "content": user_input
        }
    )

#Display msg
    with st.chat_message("user"):

        st.write(
            user_input
        )

#Vector database
    if st.session_state.vector_db is None:

        answer = (
            "📄 Please upload and process a "
            "document first."
        )

    else:

#RAG chain
        rag_chain = create_rag_chain(
            st.session_state.vector_db
        )

#Generate msg
        with st.chat_message("assistant"):

            with st.spinner(
                "Thinking..."
            ):

                try:

                    answer = rag_chain.invoke(
                        user_input
                    )

                except Exception as e:

                    answer = (
                        f"❌ Error: {str(e)}"
                    )


            st.write(answer)

#Save user msg
    current_chat["messages"].append(
        {
            "role": "assistant",
            "content": answer
        }
    )

#Chat title
    if current_chat["title"] == "New Chat":

        current_chat["title"] = (
            user_input[:30]
            + ("..." if len(user_input) > 30 else "")
        )