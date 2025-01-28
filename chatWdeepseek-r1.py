import time
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.manager import CallbackManager

# Load environment variables from .env (optional, if needed for other configs)
load_dotenv()

##############################################################################
# 1) Define a Custom Callback for Streaming
##############################################################################
class StreamingCallbackHandler(BaseCallbackHandler):
    """
    Prints tokens in real time, tracks total tokens/time, and prints stats.
    Also holds the final text so it can be stored in chat history.
    """

    def __init__(self):
        super().__init__()
        self.reset()

    def reset(self):
        """Reset counters and buffers for each new generation."""
        self.start_time = None
        self.token_count = 0
        self.accumulated_tokens = []

    def on_llm_start(self, serialized, prompts, **kwargs):
        """Called before LLM starts generating."""
        self.start_time = time.time()
        self.token_count = 0
        self.accumulated_tokens = []
        # Optional: print a small header
        # print("AI is thinking...")

    def on_llm_new_token(self, token: str, **kwargs):
        """Called for each newly generated token."""
        self.token_count += 1
        self.accumulated_tokens.append(token)
        # Print token as it appears (no newline, flush so it shows immediately)
        print(token, end="", flush=True)

    def on_llm_end(self, response, **kwargs):
        """Called when the LLM finishes generating."""
        end_time = time.time()
        total_time = end_time - self.start_time
        tokens_per_second = (self.token_count / total_time) if total_time > 0 else 0
        # Print final stats
        print(
            f"\n---\n"
            f"Time taken: {total_time:.2f} seconds\n"
            f"Tokens: {self.token_count}\n"
            f"Tokens/sec: {tokens_per_second:.2f}\n"
            f"---"
        )

    def get_final_text(self) -> str:
        """Utility to retrieve the entire generated text for storing in history."""
        return "".join(self.accumulated_tokens)


##############################################################################
# 2) Create an Ollama Model in Streaming Mode with the Custom Callback
##############################################################################
# Create our callback manager
streaming_callback = StreamingCallbackHandler()
callback_manager = CallbackManager([streaming_callback])

# Create ChatOllama model with streaming enabled
model = ChatOllama(
    model="deepseek-r1",          # Replace with your local model name
    callback_manager=callback_manager,
    streaming=True                # Important: enable streaming
)

##############################################################################
# 3) Chat Loop
##############################################################################
chat_history = []

# Optional initial system message
system_msg = SystemMessage(content="You are a helpful AI assistant.")
chat_history.append(system_msg)

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break

    # Append user's query to chat history
    chat_history.append(HumanMessage(content=user_input))

    # Reset the callback so counters/timers start fresh for this generation
    streaming_callback.reset()

    # Invoke the model in streaming mode
    # The tokens will be printed live as they are generated
    result = model.invoke(chat_history)

    # Once generation is done, retrieve the final text from our callback
    final_text = streaming_callback.get_final_text()

    # Add the AI's response to chat history
    chat_history.append(AIMessage(content=final_text))

    # (Optionally, you could compare `result.content` vs. `final_text`,
    #  but they should match. The main difference is that `final_text`
    #  was assembled from streamed tokens.)

    # Print a blank line or separator if you like
    print()

print("---- Message History ----")
for msg in chat_history:
    print(f"{msg.type}: {msg.content}")
