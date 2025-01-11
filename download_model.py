from transformers import AutoModelForCausalLM, AutoTokenizer
import os

# Load the Hugging Face token from environment variables
hf_token = os.getenv("HF_TOKEN")

# Load the model and tokenizer
model_path = "Tatakaiiii/Mouto"  # Replace with your model repository path
tokenizer = AutoTokenizer.from_pretrained(model_path, use_auth_token=hf_token)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    use_auth_token=hf_token,
    device_map="auto",
    load_in_8bit=True,  # Enable 8-bit quantization
    torch_dtype="auto",
)

# Save the model and tokenizer locally
model.save_pretrained("./model")
tokenizer.save_pretrained("./model")

print("Model and tokenizer downloaded and saved successfully!")
