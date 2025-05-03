import os
from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.utils.local_file_handler import LocalFileHandler


class Setup:
    def __init__():
        pass

    # Function to load or download model
    def get_model_and_tokenizer(model_name, local_path):
        if os.path.exists(os.path.join(local_path, "config.json")):
            print(f"Loading model from {local_path}...")
            tokenizer = AutoTokenizer.from_pretrained(local_path)
            model = AutoModelForCausalLM.from_pretrained(
                local_path, torch_dtype="auto", device_map="auto"
            )
        else:
            print(f"Downloading model {model_name}...")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            tokenizer.save_pretrained(local_path)

            model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype="auto", device_map="auto"
            )
            model.save_pretrained(local_path)
            print(f"Model saved to {local_path}")

        return model, tokenizer


class RequestHandle:
    def __init__(self):
        self._local_file_handler = LocalFileHandler()

    def get_gcs_file(self, entry_id):
        file = self._local_file_handler.load_file(f"src/api/data/{entry_id}.md")
        return file


class InferenceRequest(BaseModel):
    entry_id: str


# Get model and tokenizer


# Load model and tokenizer
s = Setup()
MODEL_NAME = "Qwen/Qwen3-0.6B"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype="auto", device_map="auto")

# API
app = FastAPI()
handle = RequestHandle()


@app.get("/")
def health_check():
    return {"message": "ML Cloud API is up and running!"}


@app.post("/summarize")
def infer(request: InferenceRequest):
    paper_md = handle.get_gcs_file(request.entry_id)

    prompt = f"Provide a 3-5 paragraph summary of this research paper from arxiv:\n{paper_md}"
    messages = [{"role": "user", "content": prompt}]

    print("retrieved message")
    print(prompt)

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    print("applied chat template")

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    print("tokenized text")

    generated_ids = model.generate(**model_inputs, max_new_tokens=100)
    print("generated ids")

    output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()
    print("output ids")

    try:
        # rindex finding 151668 (</think>)
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0

    thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
    print("thinking content done")

    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
    print("content done")

    print("\nthinking content:", thinking_content)
    print("content:", content)

    # return {"thinking content": thinking_content, "content": content}

    return {"response": content, "input": request.entry_id}
