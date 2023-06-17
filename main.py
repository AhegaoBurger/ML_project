import os
import configparser
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Weaviate
from langchain.document_loaders import TextLoader

config = configparser.ConfigParser()
config.read('config.ini')

os.environ["OPENAI_API_KEY"] = config.get('OPENAI', 'api_key')
os.environ["WEAVIATE_URL"] = config.get('WEAVIATE', 'url')
os.environ["WEAVIATE_API_KEY"] = config.get('WEAVIATE', 'api_key')

loader = TextLoader('/home/artur/Downloads/babylon.txt')
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)
print(f'You now have {len(docs)} documents')

embeddings = OpenAIEmbeddings()

db = Weaviate.from_documents(docs, embeddings, weaviate_url=WEAVIATE_URL, by_text=False)

query = 'What is the most useful wisdom from this book'
docs = db.similarity_search(query)
