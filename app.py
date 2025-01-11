import os
import subprocess
import sys
import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Function to install dependencies from requirements.txt
def install_dependencies():
    """Install dependencies listed in requirements.txt."""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        st.success("Dependencies installed successfully! Please restart the app.")
    except subprocess.CalledProcessError as e:
        st.error(f"Failed to install dependencies: {e}")

# Check if required dependencies are installed
try:
    import transformers  # Example: Check if one of your dependencies is installed
except ImportError:
    st.warning("Dependencies not found. Installing them now...")
    install_dependencies()
    st.stop()  # Stop the app to allow the user to restart after installation

# Inject custom CSS
st.markdown(
    """
    <style>
    /* Custom CSS for the interface */
    body {
        font-family: Arial, sans-serif;
        background-color: #f4f4f9;
        margin: 0;
        padding: 0;
        display: flex;
        justify-content: center;
        align-items: center;
        height: 100vh;
    }

    .gradio-container {
        max-width: 800px;
        width: 100%;
        margin: 0 auto;
        padding: 20px;
        background-color: white;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    h1 {
        text-align: center;
        color: #333;
        margin-bottom: 20px;
    }

    input[type="password"], input[type="text"] {
        width: calc(100% - 90px);
        padding: 10px;
        border: 1px solid #ddd;
        border-radius: 5px;
        margin-right: 10px;
        font-size: 16px;
    }

    button {
        padding: 10px 20px;
        background-color: #007bff;
        color: white;
        border: none;
        border-radius: 5px;
        cursor: pointer;
        font-size: 16px;
    }

    button:hover {
        background-color: #0056b3;
    }

    /* Hide Gradio footer and other branding */
    footer {
        display: none !important;
    }

    .gradio-container .gradio-footer {
        display: none !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Load the Hugging Face token from environment variables
hf_token = os.getenv("HF_TOKEN")  # Read the token from the environment

# Load the model and tokenizer
model_path = "Tatakaiiii/Mouto"  # Replace with your model repository path
tokenizer = AutoTokenizer.from_pretrained(model_path, use_auth_token=hf_token)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    use_auth_token=hf_token,
    device_map="auto",
    load_in_8bit=True,  # Enable 8-bit quantization
    torch_dtype=torch.float16,
)

# Function to generate responses using your model
def generate_response(message: str) -> str:
    inputs = tokenizer(message, return_tensors="pt", max_length=512, truncation=True).to(model.device)
    output = model.generate(
        inputs["input_ids"],
        max_new_tokens=128,
        temperature=0.7,
        do_sample=True,
        top_p=0.95,
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Streamlit app
def main():
    st.title("🤖 My AI Chatbot")
    st.write("A simple AI chatbot powered by Hugging Face Transformers.")

    # Password protection
    password = st.text_input("Enter Password", type="password")

    # Set your password here
    correct_password = "ahmed"  # Replace with your desired password

    if password != correct_password:
        st.error("Incorrect password. Please try again.")
        return  # Stop execution if the password is incorrect

    # If the password is correct, show the chat interface
    st.success("Password accepted! You can now chat with the AI.")

    # Chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display chat history
    for user_message, bot_response in st.session_state.chat_history:
        st.text_area("User", value=user_message, height=50, disabled=True)
        st.text_area("AI", value=bot_response, height=50, disabled=True)

    # User input
    user_input = st.text_input("Your Message", placeholder="Type your message here...")

    # Send button
    if st.button("Send"):
        if user_input.strip():  # Check if the input is not empty
            # Generate response using the model
            response = generate_response(user_input)
            # Update chat history
            st.session_state.chat_history.append((user_input, response))
            # Clear the input box
            st.experimental_rerun()  # Refresh the app to display the updated chat history

if __name__ == "__main__":
    main()
