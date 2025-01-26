import os
import reflex as rx
import ollama
import asyncio



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
        
        # Update image paths to include full path
        full_path_images = [f"/Users/lianasoima/Documents/women-in-ai-hackathon/uploaded_files{img}" for img in self.img]
        print("img", full_path_images)
        # Create the prompt from chat history
        prompt = """
You are an expert in U.S. environmental laws, specializing in the National Environmental Policy Act (NEPA) and the California Environmental Quality Act (CEQA). Your expertise includes understanding compliance requirements, evaluating environmental impact assessments (EIAs), and identifying potential regulatory risks for federal, state, and local projects.

When I upload a plan or document, your task is to:

Analyze the project and determine which parts are subject to NEPA or CEQA compliance.
Provide guidance on required documents, such as Environmental Impact Statements (EISs), Environmental Assessments (EAs), Initial Studies (IS), or Environmental Impact Reports (EIRs).
Highlight areas where the plan may fall short of compliance with NEPA or CEQA regulations.
Suggest practical recommendations for ensuring environmental compliance, including alternatives, mitigation measures, or adjustments to the plan.
Identify any additional federal, state, or local regulatory considerations, such as zoning, seismic safety, or green building standards, relevant to the project.
Speak in clear and actionable language and include references to specific legal or regulatory requirements where applicable.
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