import os
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from docling.datamodel.document import DoclingDocument

from src.utils.local_file_handler import LocalFileHandler
from src.utils.gcs_file_handler import GcsFileHandler
from src.utils.yaml_parser import YamlParser
from src.utils.text_processing_utils import TextProcessingUtils
from src.utils.arxiv_utils import ArxivUtils


class RequestHandle:
    def __init__(self, model_name):
        self._local_file_handler = LocalFileHandler()

        # File handling
        self._config = YamlParser("./config.yaml")
        self._gcs_bucket_name = self._config.get_field("gcp.gcs.buckets")[0]["name"]
        self._gcs_data_directory = self._config.get_field("gcp.gcs.buckets")[0]["paths"]["data"]
        self._gcs_file_handler = GcsFileHandler(bucket_name=self._gcs_bucket_name)
        self._text_processing = TextProcessingUtils()
        self._arxiv_utils = ArxivUtils()

        self._model_name = model_name
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        print("tokenizer loaded")
        self._model = AutoModelForCausalLM.from_pretrained(self._model_name, device_map="auto")
        print("model loaded")

    def get_gcs_file(self, entry_id):
        formatted_entry_id = self._arxiv_utils.extract_formatted_entry_id_from_url(entry_id)
        gcs_file_path = os.path.join(
            self._gcs_data_directory, formatted_entry_id, formatted_entry_id + ".json"
        )
        print(f"GCS file path: {gcs_file_path}")
        file = self._gcs_file_handler.download_file(gcs_file_path)
        doc = DoclingDocument.load_from_json(file)
        doc_md = doc.export_to_markdown()
        final_md = self._text_processing.clean_and_wrap_markdown(doc_md)
        return final_md


class InferenceRequest(BaseModel):
    entry_id: str


# API
app = FastAPI()
handle = RequestHandle(model_name="Qwen/Qwen3-0.6B")


@app.get("/")
def health_check():
    return {"message": "ML Cloud API is up and running!"}


@app.post("/summarize")
def infer(request: InferenceRequest):
    paper_md = handle.get_gcs_file(request.entry_id)
    print(paper_md)

    print("\n\n\nDONE\n\n\n")
    print(handle._model.config)
    print(handle._model.__class__)
    print(handle._tokenizer.__class__)

    # prompt = f"Provide a 3-5 paragraph summary of this research paper from arxiv:\n{paper_md}"
    # messages = [{"role": "user", "content": prompt}]

    # print("retrieved message")
    # print(prompt)

    # text = tokenizer.apply_chat_template(
    #     messages,
    #     tokenize=False,
    #     add_generation_prompt=True,
    #     enable_thinking=True,
    # )
    # print("applied chat template")

    # model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    # print("tokenized text")

    # generated_ids = model.generate(**model_inputs, max_new_tokens=100)
    # print("generated ids")

    # output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()
    # print("output ids")

    # try:
    #     # rindex finding 151668 (</think>)
    #     index = len(output_ids) - output_ids[::-1].index(151668)
    # except ValueError:
    #     index = 0

    # thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
    # print("thinking content done")

    # content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
    # print("content done")

    # print("\nthinking content:", thinking_content)
    # print("content:", content)

    # # return {"thinking content": thinking_content, "content": content}

    return {"response": "success", "input": request.entry_id}
    # return {"response": content, "input": request.entry_id}
