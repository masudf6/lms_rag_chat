import os, json
from datetime import datetime, timedelta
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
import redis.asyncio as redis
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from azure.storage.blob import ContainerClient, generate_blob_sas, BlobSasPermissions

load_dotenv()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ENV VARS
REDIS_URL = os.getenv("REDIS_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_KEY = os.getenv("PINECONE_KEY")
AZURE_CONN_STR = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
CONTAINER_NAME = os.getenv("AZURE_CONTAINER_NAME")

# Clients
redis_client = redis.from_url(REDIS_URL, decode_responses=True)
llm = ChatOpenAI(temperature=0, api_key=OPENAI_API_KEY)

# Format redis key
def redis_key(session_id):
    return f"memory:{session_id}"


# ---- BLOB STORAGE ----
def generate_sas_url(blob_name):
    account_name = os.getenv("AZURE_STORAGE_ACCOUNT_NAME")
    account_key = os.getenv("AZURE_STORAGE_ACCOUNT_KEY")
    container_name = os.getenv("AZURE_CONTAINER_NAME")
    sas_token = generate_blob_sas(
        account_name=account_name,
        container_name=container_name,
        blob_name=blob_name,
        account_key=account_key,
        permission=BlobSasPermissions(read=True),
        expiry=datetime.utcnow() + timedelta(hours=1)  # Valid for 1 hour
    )
    url = f"https://{account_name}.blob.core.windows.net/{container_name}/{blob_name}?{sas_token}"
    return url

@app.get("/files")
async def list_files():
    container = ContainerClient.from_connection_string(AZURE_CONN_STR, CONTAINER_NAME)
    blob_list = container.list_blobs()
    files = []
    for blob in blob_list:
        sas_url = generate_sas_url(blob.name)
        parts = blob.name.split("/")
        course = parts[0] if len(parts) > 2 else ""
        mat_type = parts[1] if len(parts) > 2 else ""
        filename = parts[-1]
        files.append({
            "name": filename,
            "course": course,
            "material_type": mat_type,
            "full_path": blob.name,
            "size": blob.size,
            "last_modified": str(blob.last_modified),
            "sas_url": sas_url
        })
    return files

# Format memory to ChatPrompt style
def format_history(memory):
    return [
        {"role": "user", "content": m["text"]} if m["sender"] == "user"
        else {"role": "assistant", "content": m["text"]}
        for m in memory
    ]

# ---- STD CHAT PROMPT ----
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("placeholder", "{history}"),
    ("human", "{input}")
])

chat_chain = chat_prompt | llm | StrOutputParser()

@app.post("/chat/{session_id}")
async def chat(session_id: str, request: Request):
    data = await request.json()
    user_message = data.get("message", "")

    print(f'Chat session_id: {redis_key(session_id)}')

    raw = await redis_client.get(redis_key(session_id))
    memory = json.loads(raw) if raw else []
    memory.append({"sender": "user", "text": user_message})

    response = await chat_chain.ainvoke({
        "input": user_message,
        "history": format_history(memory)
    })

    memory.append({"sender": "bot", "text": response})
    await redis_client.set(redis_key(session_id), json.dumps(memory))
    return {"response": response}   # py obj, auto converts to JSON

@app.get("/chat/{session_id}")
async def get_history(session_id: str):
    raw = await redis_client.get(redis_key(session_id))
    memory = json.loads(raw) if raw else []
    return JSONResponse(memory)     # list, so need to tell FastAPI it's JSON

@app.delete("/chat/{session_id}")
async def clear_history(session_id: str):
    await redis_client.delete(redis_key(session_id))
    return {"status": "cleared"}

# ---- SESSION MGMT ----
@app.post("/new_session/{user_id}")
async def new_session(user_id: str):
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    session_id = f"{user_id}-{timestamp}"
    await redis_client.sadd(f"sessions:{user_id}", session_id)
    return {"session_id": session_id}

@app.get("/sessions/{user_id}")
async def list_sessions(user_id: str):
    session_ids = await redis_client.smembers(f"sessions:{user_id}")
    session_list = []
    for session_id in sorted(session_ids, reverse=True):
        name = await redis_client.get(f"session_name:{session_id}")
        session_list.append({
            "id": session_id,
            "name": name or session_id.replace(f"{user_id}-", "")
        })
    return JSONResponse(session_list)

@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    user_id = session_id.split("-")[0]
    await redis_client.delete(redis_key(session_id))
    await redis_client.delete(f"session_name:{session_id}")
    await redis_client.srem(f"sessions:{user_id}", session_id)
    return {"status": "deleted"}

@app.post("/rename_session/{session_id}")
async def rename_session(session_id: str, request: Request):
    data = await request.json()
    new_name = data.get("name", "")
    await redis_client.set(f"session_name:{session_id}", new_name)
    return {"status": "renamed", "name": new_name}




# ---- RAG ----
pc = Pinecone(api_key=PINECONE_KEY)
index = pc.Index("fyp")
embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")
vectorstore = PineconeVectorStore(index, embedding_model, "content")
retriever = vectorstore.as_retriever()

rag_prompt = ChatPromptTemplate([
    ("system", "You are a helpful assistant. Use the following information to answer the user's question. "
               "You can use both the chat history and retrieved documents. "
               "First identify the domain area of the question asked, then check if the domain area is present in the retrieved context. "
               "If the domain area is not present in the retrieved documents or in chat history, say 'I don’t know'."),
    ("placeholder", "{history}"),
    ("system", "Retrieved Documents:\n{context}"),
    ("human", "{question}")
])

# rag_chain = (
#     {
#         "context": RunnableLambda(lambda x: retriever.invoke(x["question"])),
#         "question": RunnableLambda(lambda x: x["question"]),
#         "history": RunnableLambda(lambda x: x.get("history", []))
#     }
#     | rag_prompt
#     | llm
#     | StrOutputParser()
# )

@app.post("/rag/{session_id}")
async def rag_chat(session_id: str, request: Request):
    data = await request.json()
    question = data.get("message", "")
    filename = data.get("filename", None)

    print(f'Rag session_id: {session_id}')

    raw = await redis_client.get(redis_key(session_id))
    memory = json.loads(raw) if raw else []
    memory.append({"sender": "user", "text": question})

    if filename:
        docs = retriever.invoke(question, filter={"filename": filename})
    else:
        docs = retriever.invoke(question)

    # get retrieved docs
    print(f"Retrieved {len(docs)} documents")
    # print(f"Doc: {docs[0].metadata}")
    context = "\n---\n".join([f"[{doc.metadata.get('filename', 'unknown')}]: {doc.page_content}" for doc in docs])

    response = await (rag_prompt | llm | StrOutputParser()).ainvoke({
        "question": question,
        "history": format_history(memory),
        "context": context
    })

    memory.append({"sender": "bot", "text": response})
    await redis_client.set(redis_key(session_id), json.dumps(memory))

    return {
        "response": response,
        "sources": [
            {"source": doc.metadata.get("filename", "unknown"), "text": doc.page_content}
            for doc in docs
        ]
    }
