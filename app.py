from flask import Flask, request, redirect, url_for, render_template
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

# Initialize Flask app
app = Flask(__name__)

# Set your password here
PASSWORD = "ahmed"
hf_token = os.getenv("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")
# Load the Hugging Face model and tokenizer
model_path = "Tatakaiiii/Mouto"  # Your Hugging Face model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")

# Custom CSS for the interface
custom_css = """
footer { display: none !important; }
.gradio-container { max-width: 800px; margin: auto; padding: 20px; }
"""

# Function to check the password
def check_password(password):
    return password == PASSWORD

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

# Gradio interface
def create_chat_interface():
    with gr.Blocks(css=custom_css) as demo:
        gr.Markdown("# 🤖 My AI Chatbot")
        chatbot = gr.Chatbot(label="Chat History", elem_id="chatbot")
        message = gr.Textbox(label="Your Message", placeholder="Type your message here...")
        submit_btn = gr.Button("Send")

        def chat(message: str, history: list) -> tuple[list, str]:
            response = generate_response(message)  # Use your model here
            history.append((message, response))
            return history, ""

        submit_btn.click(chat, inputs=[message, chatbot], outputs=[chatbot, message])

    return demo

# Flask routes
@app.route("/")
def home():
    return redirect(url_for("login"))

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        password = request.form.get("password")
        if check_password(password):
            return redirect(url_for("chat"))
        else:
            return render_template("login.html", error="Incorrect password. Please try again.")
    return render_template("login.html")

@app.route("/chat")
def chat():
    # Render the chat interface wrapper
    return render_template("chat.html")

# Route to serve the Gradio app
@app.route("/gradio")
def gradio():
    # Create and launch the Gradio interface
    gradio_app = create_chat_interface()
    return gradio_app.launch(share=False, inline=True)

if __name__ == "__main__":
    app.run(debug=True)
