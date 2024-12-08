import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from langchain_community.document_loaders.csv_loader import CSVLoader

DATA_DIR = "../../data/"

app = FastAPI()
loader = CSVLoader(file_path=f"{DATA_DIR}faq_final.csv",
                  source_column="id",
                  metadata_columns=["question", "context", "content"],
                  content_columns=["content", "context"],
                  csv_args={
                        "fieldnames": ["id", "context", "question", "content"],
                        }
                  )

faq_data = loader.load()[1:]

loader = CSVLoader(
    file_path=f"{DATA_DIR}khas_bank_branches.csv",
    source_column="name",  
    metadata_columns=["name", "name_en", "type", "address", "address_en", "phone", "timetable_en"],  
    content_columns=["timetable"],  
    csv_args={
        "fieldnames": [
            "name", "name_en", "type", "address", "address_en", "timetable", 
            "timetable_en", "phone", "mon", "tue", "wed", "thu", "fri", 
            "sat", "sun", "open_time", "close_time"
        ],
    }
)
branches_data = loader.load()[1:]

loader = CSVLoader(
    file_path=f"{DATA_DIR}product_info.csv",
    source_column="url",
    metadata_columns=["question", "context", "content"],
    content_columns=["content", "question"],
    csv_args={
        "fieldnames": ["url", "context", "question", "content"],
        }
)
products_data = loader.load()[1:]

data = products_data + branches_data + faq_data

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200
)
docs = text_splitter.split_documents(documents=data)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

vector_store = FAISS.from_documents(docs, embeddings, distance_strategy="COSINE")
vector_store.save_local("faiss_index")

vector_store = FAISS.load_local(
    "faiss_index", embeddings, allow_dangerous_deserialization=True
)

from huggingface_hub import notebook_login

notebook_login()

model_id = "meta-llama/Llama-3.1-8B"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    torch_dtype=torch.float16,
    device_map='auto'
)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    # llm-н үүсгэж болох токений дээд хязгаар
    max_new_tokens=1024,
    # max_new_tokens=512,
    # max_new_tokens=32,
    # хариултын randomization-г арилгах
    do_sample=True,
    # top_k=1,
    top_k=1,
    # repetition_penalty=1.15,
    repetition_penalty=1.15,
    # гаралт бидний өгсөн prompt-г хамт хэвлэхгүй байх
    return_full_text=False,
    pad_token_id=tokenizer.eos_token_id,
)

llm = HuggingFacePipeline(pipeline=pipe)

system_prompt = (
    """
    You are an assistant for question-answering tasks. 
    Use only the following pieces of retrieved context to answer the user question. 
    If the context does not contain an answer, say: "Миний мэдээлэлд таны асуултын хариулт байхгүй байна. Асуултаа тодруулна уу эсвэл лавлах төвтэй холбогдоно уу." 
    DO NOT MAKE UP ANSWERS.
    
    Use concise Mongolian (three sentences max) unless the user requests otherwise. 
    Context: {context}
    """
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("user", "question: \"\"\"{input}\"\"\""),
        ("assistant", "answer: "),
    ]
)

retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 2})


# docs = retriever.invoke("QPay үйлчилгээ гэж юу вэ?")
# docs = retriever.invoke("‘QPay үйлчилгээ’ гэж юу вэ?")
# docs = retriever.invoke("Тэмүүлэл дэд карт")
# docs = retriever.invoke("Энгийн кредит карт")
# docs = history_aware_retriever.invoke("Ням гарагт ажиллах салбарууд")
docs = retriever.invoke("Ням гарагт ажиллах салбарууд")

question_answer_chain = create_stuff_documents_chain(llm, prompt)

rag_chain = create_retrieval_chain(retriever, question_answer_chain)


# response = rag_chain.invoke({"input": "дебит карт яаж авах вэ"})
# response = rag_chain.invoke({"input": "нууц үг хэрхэн сэргээх"})
# response = rag_chain.invoke({"input": "хадгаламж нээх"})
# response = rag_chain.invoke({"input": "Цалингийн зээлийн шалгуур"})
# response = rag_chain.invoke({"input": "Тэмүүлэл "})
# response = rag_chain.invoke({"input": "Цалингийн зээлийн хугацаа"})
# response = rag_chain.invoke({"input": "QPay үйлчилгээ хэн ашиглах боломжтой вэ"})
# response = rag_chain.invoke({"input": "QPay үйлчилгээ "})
# response = rag_chain.invoke({"input": "Хагассайн өдөр ажиллах салбарууд"})
# response = rag_chain.invoke({"input": "Ням гарагт ажиллах салбарууд"})
response = rag_chain.invoke({"input": "Энгийн кредит карт давуу тал"})

print(response)
response["answer"]

origins = [
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str


@app.post("/query")
async def query_handler(request: QueryRequest):
    try:
        response = rag_chain.invoke({"input": request.query})
        print(response)
        result = response.get("answer", "No answer found.")
        return {"response": result}
    
    except Exception as e:
        print(f"Error during query processing: {str(e)}")
        
        raise HTTPException(status_code=500, detail={"error": str(e)})
