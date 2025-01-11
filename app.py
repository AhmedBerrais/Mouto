import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

# Load the model and tokenizer from Hugging Face
model_path = "Tatakaiiii/Mouto"  # Replace with your model repository path
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")

# Function to generate responses
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

# Custom HTML and CSS for the interface
custom_css = """
footer { display: none !important; }
.gradio-container { max-width: 800px; margin: auto; padding: 20px; }
"""

# Gradio interface
with gr.Blocks(css=custom_css) as demo:
    with gr.Column():
        gr.Markdown("# 🤖 My AI Chatbot")  # Title
        chatbot = gr.Chatbot(label="Chat History", elem_id="chatbot")
        message = gr.Textbox(label="Your Message", placeholder="Type your message here...")
        submit_btn = gr.Button("Send")

    # Function to handle chat interaction
    def chat(message: str, history: list) -> tuple[list, str]:
        response = generate_response(message)
        history.append((message, response))
        return history, ""

    # Link the button to the function
    submit_btn.click(chat, inputs=[message, chatbot], outputs=[chatbot, message])

# Launch Gradio app
if __name__ == "__main__":
    demo.launch(share=False)
