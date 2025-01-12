import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from openwebui import WebUI, ChatInterface, Sidebar

# Load the Hugging Face token
hf_token = "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"  # Replace with your actual token



if not hf_token:
    raise ValueError("Hugging Face token not found. Please set the HF_TOKEN.")

# Load the model and tokenizer
model_path = "Tatakaiiii/Mouto"  # Smaller and more efficient model
try:
    tokenizer = AutoTokenizer.from_pretrained(model_path, token=hf_token)
    
    # Fix: Set pad_token to eos_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # Use eos_token as pad_token

    device = "cuda" if torch.cuda.is_available() else "cpu"  # Explicitly set device
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        token=hf_token,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,  # Use FP16 on GPU, FP32 on CPU
    ).to(device)  # Move model to the appropriate device
except Exception as e:
    raise RuntimeError(f"Failed to load the model: {e}")

# Function to generate responses using your model
def generate_response(message: str) -> str:
    inputs = tokenizer(message, return_tensors="pt", max_length=512, truncation=True, padding=True).to(device)
    output = model.generate(
        inputs["input_ids"],
        max_new_tokens=128,
        temperature=0.7,
        do_sample=True,
        top_p=0.95,
        attention_mask=inputs["attention_mask"],  # Add attention mask
        pad_token_id=tokenizer.pad_token_id,  # Use pad_token_id
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Create the WebUI instance
web_ui = WebUI(title="🤖 My AI Chatbot", layout="wide")

# Create the Sidebar
sidebar = Sidebar(title="Options")
sidebar.add_button("New Chat", on_click=lambda: web_ui.clear_chat_history())
sidebar.add_section("Modelfiles", content="Manage your model files here.")
sidebar.add_section("Prompts")
sidebar.add_button("Tell me a fun fact", on_click=lambda: web_ui.set_user_input("Tell me a fun fact"))
sidebar.add_button("Give me ideas", on_click=lambda: web_ui.set_user_input("Give me ideas"))
sidebar.add_button("Show me a code snippet", on_click=lambda: web_ui.set_user_input("Show me a code snippet"))
sidebar.add_button("Overcome procrastination", on_click=lambda: web_ui.set_user_input("Give me tips to overcome procrastination"))

# Add the Sidebar to the WebUI
web_ui.add_sidebar(sidebar)

# Create the ChatInterface
chat_interface = ChatInterface()

# Add the ChatInterface to the WebUI
web_ui.add_chat_interface(chat_interface)

# Main function to run the app
def main():
    # Set up the chat interface
    chat_interface.set_generate_response_function(generate_response)
    
    # Run the WebUI
    web_ui.run()

if __name__ == "__main__":
    main()
