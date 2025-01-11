from flask import Flask, request, redirect, url_for, render_template_string
import gradio as gr

app = Flask(__name__)

# Set your password here
PASSWORD = "ahmed"

# Custom CSS for the interface
custom_css = """
footer { display: none !important; }
.gradio-container { max-width: 800px; margin: auto; padding: 20px; }
"""

# Function to check the password
def check_password(password):
    return password == PASSWORD

# Gradio interface
def create_chat_interface():
    with gr.Blocks(css=custom_css) as demo:
        gr.Markdown("# 🤖 My AI Chatbot")
        chatbot = gr.Chatbot(label="Chat History", elem_id="chatbot")
        message = gr.Textbox(label="Your Message", placeholder="Type your message here...")
        submit_btn = gr.Button("Send")

        def chat(message: str, history: list) -> tuple[list, str]:
            # Simulate a response (replace with your model's logic)
            response = f"AI: You said '{message}'"
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
            return render_template_string("""
                <h1>Login</h1>
                <form method="post">
                    <label for="password">Password:</label>
                    <input type="password" id="password" name="password" required>
                    <button type="submit">Submit</button>
                </form>
                <p style="color: red;">Incorrect password. Please try again.</p>
            """)
    return render_template_string("""
        <h1>Login</h1>
        <form method="post">
            <label for="password">Password:</label>
            <input type="password" id="password" name="password" required>
            <button type="submit">Submit</button>
        </form>
    """)

@app.route("/chat")
def chat():
    # Create and launch the Gradio interface
    gradio_app = create_chat_interface()
    return gradio_app.launch(share=False, inline=True)

if __name__ == "__main__":
    app.run(debug=True)
