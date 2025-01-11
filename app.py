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
    st.stop()

# Load the Hugging Face token
hf_token = "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"  # Replace with your actual token


if not hf_token:
    st.error("Hugging Face token not found. Please set the HF_TOKEN.")
    st.stop()

# Load the model and tokenizer
model_path = "microsoft/DialoGPT-small"  # Smaller and more efficient model
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
    st.error(f"Failed to load the model: {e}")
    st.stop()

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
        st.text_area("User", value=user_message, height=68, disabled=True)  # Updated height to 68
        st.text_area("AI", value=bot_response, height=68, disabled=True)  # Updated height to 68

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
            st.rerun()  # Refresh the app to display the updated chat history

if __name__ == "__main__":
    main()
