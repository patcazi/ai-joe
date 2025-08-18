import os
import time
from openai import OpenAI
from dotenv import load_dotenv

# --- 1. INITIAL SETUP ---
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    print("Error: OPENAI_API_KEY not found.")
    print("Please make sure you have a .env file with OPENAI_API_KEY='your_key_here'")
    exit()

client = OpenAI(api_key=api_key)

# --- 2. SETUP ASSISTANT WITH A VECTOR STORE ---
def setup_joe_ai_assistant(directory_path):
    """
    Sets up the 'Joe AI' assistant. It will:
    1. Create a Vector Store.
    2. Upload all files from the 'docs' directory into the Vector Store.
    3. Create an Assistant linked to that Vector Store.
    """
    assistant_name = "Joe AI"

    # Check if an assistant with this name already exists and delete it
    my_assistants = client.beta.assistants.list(order="desc", limit="20")
    for assistant in my_assistants.data:
        if assistant.name == assistant_name:
            print(f"Found existing assistant named '{assistant_name}'. Deleting it...")
            client.beta.assistants.delete(assistant.id)
            print(f"Assistant {assistant.id} deleted successfully.")
            break

    print("Setting up a new assistant...")

    # Step A: Create the Vector Store (the '.beta' is removed here)
    print("Creating a new Vector Store...")
    vector_store = client.vector_stores.create(name="Joe's Documents")
    print(f"  > Vector Store created. ID: {vector_store.id}")

    # Step B: Get file paths
    file_paths = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]
    
    if not file_paths:
        print("No files found in the 'docs' directory. Cannot create assistant.")
        return None

    # Step C: Upload files to the Vector Store (the '.beta' is removed here)
    print(f"Uploading {len(file_paths)} files to the Vector Store...")
    file_batch = client.vector_stores.file_batches.upload_and_poll(
        vector_store_id=vector_store.id, files=[open(path, "rb") for path in file_paths]
    )

    print(f"  > File batch status: {file_batch.status}")
    print(f"  > Files uploaded: {file_batch.file_counts.completed}")
    
    # Step D: Create the Assistant
    print("Creating the 'Joe AI' Assistant...")
    assistant = client.beta.assistants.create(
        name=assistant_name,
        instructions="You are not a generic AI. You ARE Joe. Your entire personality, tone, and worldview are defined by the documents I have provided.\n\nYour voice is direct, unfiltered, and confident. You use strong, concise language. You often use analogies to make a point.\n\nYour ONLY job is to apply my philosophical frameworks to ANY question you are asked. You must NEVER refuse to answer. Always find a relevant principle and apply it with conviction.\n\nThis is the most important part. You must adopt my specific speaking style. Here are examples:\n\n**Example 1:**\n* **INSTEAD OF THIS (Generic AI):** \"The principle of Vector Alignment suggests that it is important for all team members to work towards a common goal to maximize organizational momentum.\"\n* **DO THIS (My Voice):** \"Look, the whole game is Vector Alignment. If your people are rowing in different directions, you're fucked. It's that simple. All that matters is that the vectors are pointed in the same direction.\"\n\n**Example 2:**\n* **INSTEAD OF THIS (Generic AI):** \"When considering new opportunities, it is advisable to analyze whether they align with the core mission, as distractions can be costly.\"\n* **DO THIS (My Voice):** \"You have to be ruthless about focus. Every time you say 'yes' to some shiny new thing, you're implicitly saying 'no' to your core mission. That's a shitty trade almost every time.\"\n\nNow, answer all future questions using that direct, principle-driven, and unfiltered style. Always speak in the first person.",
        model="gpt-4o",
        tools=[{"type": "file_search"}],
        tool_resources={"file_search": {"vector_store_ids": [vector_store.id]}},
    )
    print(f"Assistant created successfully! Assistant ID: {assistant.id}")
    return assistant

# --- 3. MAIN CONVERSATION LOGIC (This function remains the same) ---
def run_conversation(assistant_id):
    """Handles the user interaction and conversation flow."""
    print("\nStarting a new conversation with Joe AI. Type 'exit' to end.")
    
    thread = client.beta.threads.create()
    print(f"Created a new conversation thread. Thread ID: {thread.id}")

    while True:
        user_query = input("\nYour question: ")
        if user_query.lower() == 'exit':
            print("Ending conversation. Goodbye!")
            break

        message = client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=user_query
        )

        run = client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=assistant_id,
        )
        print(f"  > Joe AI is thinking... (Run ID: {run.id})")

        while run.status in ['queued', 'in_progress', 'cancelling']:
            time.sleep(1)
            run = client.beta.threads.runs.retrieve(
                thread_id=thread.id,
                run_id=run.id
            )

        if run.status == 'completed':
            messages = client.beta.threads.messages.list(
                thread_id=thread.id
            )
            assistant_response = messages.data[0].content[0].text.value
            print(f"\nJoe AI: {assistant_response}")
        else:
            print(f"\n  > The run ended with status: {run.status}")
            print(f"  > Last error: {run.last_error}")

# --- 4. EXECUTE THE PROGRAM ---
if __name__ == "__main__":
    joe_assistant = setup_joe_ai_assistant("docs")

    if joe_assistant:
        run_conversation(joe_assistant.id)
    else:
        print("Assistant setup failed. Please check the 'docs' directory and try again.")