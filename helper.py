import os
import langchain
from langchain.vectorstores import FAISS
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter


def create_vector_store(
    path_to_data: str,
    path_to_vector_store: str,
    embeddings: langchain.embeddings.huggingface.HuggingFaceEmbeddings,
) -> langchain.vectorstores.faiss.FAISS:
    if os.path.exists(path_to_vector_store):
        print("Reusing vector store...\n")
        vector_store = FAISS.load_local(path_to_vector_store, embeddings)
    else:
        # loader = TextLoader("data/data.txt") # Use this line if you only need data.txt
        loader = DirectoryLoader(path_to_data)
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        docs = text_splitter.split_documents(documents)
        vector_store = FAISS.from_documents(docs, embeddings)
        vector_store.save_local(path_to_vector_store)
    return vector_store
