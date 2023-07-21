import os
import sys
import yaml
from dotenv import load_dotenv, find_dotenv
from helper import create_vector_store
from langchain import HuggingFaceHub
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
import urllib3


if __name__ == "__main__":
    os.environ[
        "CURL_CA_BUNDLE"
    ] = ""  # fixes VPN issue when connecting to hugging face hub
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    load_dotenv(find_dotenv())

    with open("static.yaml", "r") as file:
        data = yaml.safe_load(file)

    embeddings = HuggingFaceEmbeddings()
    llm = HuggingFaceHub(repo_id=data["hugging_face_repo_id"])
    vector_store = create_vector_store(
        path_to_data=data["path_to_data"],
        path_to_vector_store=data["path_to_vector_store"],
        embeddings=embeddings,
    )
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(search_kwargs={"k": 1}),
    )

    query = None
    chat_history = []
    while True:
        if not query:
            query = input("Prompt: ")
        if query in ["quit", "q", "exit"]:
            sys.exit()
        result = chain({"question": query, "chat_history": chat_history})
        print(result["answer"])
        chat_history.append((query, result["answer"]))
        query = None
