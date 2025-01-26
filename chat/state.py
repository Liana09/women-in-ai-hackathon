import os
import reflex as rx
import ollama
import asyncio
import requests
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType
from sentence_transformers import SentenceTransformer
from langchain.document_loaders import JSONLoader
import dotenv


import os
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, MilvusClient
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import JSONLoader  # Updated import
import json


dotenv.load_dotenv()

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

def mistral_request(prompt):
    url = "https://api.mistral.ai/v1/chat"  # Replace with the actual endpoint
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "mistral-large",  # Adjust model name as needed
        "prompt": prompt,
    }
    response = requests.post(url, json=payload, headers=headers)
    return response.json()


def load_data_from_milvus(chat_data: str):

    # Initialize Milvus client
    client = MilvusClient("my-milvus.db")

    # Define Milvus collection schema
    schema = CollectionSchema(
        fields=[
            FieldSchema("id", DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema("embedding", DataType.FLOAT_VECTOR, dim=768),
            FieldSchema("text", DataType.VARCHAR, max_length=65535)
        ]
    )

    # Path to the JSON file
    json_path = "california_report.json"

    # Define a JQ query string based on your JSON structure
    jq_query = '.lines[]'  # Adjusted to match the "lines" array

    # Initialize JSONLoader with the correct JQ query
    loader = JSONLoader(json_path, jq_schema=jq_query)
    print("loader", loader)
    try:
        documents = loader.load()
        texts = [doc.page_content for doc in documents]
    except json.JSONDecodeError as e:
        print(f"JSON Decode Error: {e}")
        exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        exit(1)

    # Create embeddings
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(texts)

    # Check and manage Milvus collection
    collection_name = "json_collection"
    if client.has_collection(collection_name=collection_name):
        client.drop_collection(collection_name=collection_name)

    # Create a new Milvus collection
    collection = client.create_collection(
        collection_name=collection_name,
        dimension=768,  # The vectors we will use in this demo has 768 dimensions
    )

    data = [
        {"id": i, "vector": embeddings[i], "text": texts[i], "subject": "history"}
        for i in range(len(embeddings))
    ]
    print("texts",texts)

    # Insert embeddings and texts into the collection
    client.insert(
        collection_name=collection_name,
        data=data
    )


    embeddings2 = model.encode(chat_data)
    chat_data = [
        {"id": i, "vector": embeddings2[i], "text": chat_data, "subject": "history"}
        for i in range(len(embeddings2))
    ]

    search_results = client.search(
        collection_name=collection_name,
        data=[chat_data],
        search_params={"metric_type": "COSINE", "params": {}},  # Search parameters
    )[0]
    return search_results

class QA(rx.Base):
    """A question and answer pair."""

    question: str
    answer: str


DEFAULT_CHATS = {
    "Intros": [],
}


class State(rx.State):
    """The app state."""

    # A dict from the chat name to the list of questions and answers.
    chats: dict[str, list[QA]] = DEFAULT_CHATS

    # The current chat name.
    current_chat = "Intros"

    img: list[str] = []
    

    # The current question.
    question: str

    # Whether we are processing the question.
    processing: bool = False

    # The name of the new chat.
    new_chat_name: str = ""

    def create_chat(self):
        """Create a new chat."""
        # Add the new chat to the list of chats.
        self.current_chat = self.new_chat_name
        self.chats[self.new_chat_name] = []

    def delete_chat(self):
        """Delete the current chat."""
        del self.chats[self.current_chat]
        if len(self.chats) == 0:
            self.chats = DEFAULT_CHATS
        self.current_chat = list(self.chats.keys())[0]

    def set_chat(self, chat_name: str):
        """Set the name of the current chat.

        Args:
            chat_name: The name of the chat.
        """
        self.current_chat = chat_name

    @rx.var(cache=True)
    def chat_titles(self) -> list[str]:
        """Get the list of chat titles.

        Returns:
            The list of chat names.
        """
        return list(self.chats.keys())

    async def process_question(self, form_data: dict[str, str]):
        # Get the question from the form
        question = form_data["question"]

        # Check if the question is empty
        if question == "":
            return

        model = self.openai_process_question

        async for value in model(question):
            yield value

    async def openai_process_question(self, question: str):
        """Get the response from Ollama API.

        Args:
            question: The current question.
        """
        # Add the question to the list of questions.
        qa = QA(question=question, answer="")
        self.chats[self.current_chat].append(qa)

        # Clear the input and start the processing.
        self.processing = True
        yield

        # Build the messages
        messages = []
        for qa in self.chats[self.current_chat]:
            messages.append({"role": "user", "content": qa.question})
            messages.append({"role": "assistant", "content": qa.answer})

        response_rag = load_data_from_milvus(question)
        
        # Update image paths to include full path
        full_path_images = [f"/Users/lianasoima/Documents/women-in-ai-hackathon/uploaded_files{img}" for img in self.img]
        # Create the prompt from chat history
        # retrieving, response
        prompt = """
        You are an expert in U.S. environmental laws, specializing in the National Environmental Policy Act (NEPA) and the California Environmental Quality Act (CEQA). Your expertise includes understanding compliance requirements, evaluating environmental impact assessments (EIAs), and identifying potential regulatory risks for federal, state, and local projects.

        When I upload a plan or document, your task is to:

        Analyze the project and determine which parts are subject to NEPA or CEQA compliance.
        Provide guidance on required documents, such as Environmental Impact Statements (EISs), Environmental Assessments (EAs), Initial Studies (IS), or Environmental Impact Reports (EIRs).
        Highlight areas where the plan may fall short of compliance with NEPA or CEQA regulations.
        Suggest practical recommendations for ensuring environmental compliance, including alternatives, mitigation measures, or adjustments to the plan.
        Identify any additional federal, state, or local regulatory considerations, such as zoning, seismic safety, or green building standards, relevant to the project.
        Speak in clear and actionable language and include references to specific legal or regulatory requirements where applicable.

        This is the input you need to analyze:
        {response_rag}
        """ 
        for msg in messages[:-1]:  # Exclude the last empty answer
            prompt += f"{msg['role']}: {msg['content']}\n"
        prompt += f"user: {question}"
        print("prompt sent")
        try:
            # Stream the response from Ollama
            stream = ollama.chat(
                model='llama3.2-vision',  # or your preferred model
                messages=[{'role': 'user', 'content': prompt,'images': full_path_images},],
                stream=True,
            )

            # Process the streaming response
            for chunk in stream:
                if 'message' in chunk and 'content' in chunk['message']:
                    answer_text = chunk['message']['content']
                    if answer_text is not None:
                        self.chats[self.current_chat][-1].answer += answer_text
                        self.chats = self.chats
                        yield

        except Exception as e:
            # Handle any errors
            self.chats[self.current_chat][-1].answer = f"Error: {str(e)}"
            self.chats = self.chats
            yield
        print("prompt received")
        # Toggle the processing flag
        self.processing = False

    

    @rx.event
    async def handle_upload(
        self, files: list[rx.UploadFile]
    ):
        """Handle the upload of file(s).

        Args:
            files: The uploaded files.
        """
        for file in files:
            upload_data = await file.read()
            outfile = rx.get_upload_dir() / file.filename

            # # # Save the file.
            # with outfile.open("wb") as file_object:
            #     file_object.write(upload_data)

            # Update the img var.
            self.img.append(file.filename)

        # Wait for 3 seconds before redirecting
        await asyncio.sleep(3)
        
        # Add navigation after upload is complete
        return rx.redirect("/chat")

    print("images",img)