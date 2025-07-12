import rag_pdf, rag_ia
from dotenv import load_dotenv
import os

load_dotenv()
os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = os.getenv("OPENAI_MODEL_EMBEDDING")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
OPENAI_MODEL = os.getenv("OPENAI_MODEL")

persist_directory = 'db'
pdf_path = r"os-sertoes.pdf"

chunks = rag_pdf.read_pdf(pdf_path)
vector_store = rag_pdf.create_vector_store(chunks, persist_directory, EMBEDDING_MODEL, COLLECTION_NAME)
print("Vector store created successfully.")

prompt, retriever = rag_ia.create_prompt(vector_store)
chain = rag_ia.create_chain(prompt, retriever, OPENAI_MODEL)


question = """Qual é a visão de Euclides da Cunha sobre o ambiente natural do sertão nordestino e como ele influencia a vida dos habitantes?"""
print("1º Question:")
print(question)
response = chain.invoke(
    {'input': question},
)
print('Response:')
print(response['answer'])
print('='*90)


question = """Quais são as principais características da população sertaneja descritas por Euclides da Cunha? Como ele relaciona essas características com o ambiente em que vivem?"""
print("2º Question:")
print(question)
response = chain.invoke(
    {'input': question},
)
print('Response:')
print(response['answer'])
print('='*90)


question = """Qual foi o contexto histórico e político que levou à Guerra de Canudos, segundo Euclides da Cunha?"""
print("3º Question:")
print(question)
response = chain.invoke(
    {'input': question},
)
print('Response:')
print(response['answer'])
print('='*90)


question = """Como Euclides da Cunha descreve a figura de Antônio Conselheiro e seu papel na Guerra de Canudos?"""
print("4º Question:")
print(question)
response = chain.invoke(
    {'input': question},
)
print('Response:')
print(response['answer'])
print('='*90)


question = """Quais são os principais aspectos da crítica social e política presentes em "Os Sertões"? Como esses aspectos refletem a visão do autor sobre o Brasil da época?"""
print("4º Question:")
print(question)
response = chain.invoke(
    {'input': question},
)
print('Response:')
print(response['answer'])
print('='*90)