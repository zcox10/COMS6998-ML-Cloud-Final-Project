import os
from typing import List
from transformers import AutoModelForCausalLM, AutoTokenizer
from docling.datamodel.document import DoclingDocument

from src.utils.embedding_model_utils import EmbeddingModelUtils
from src.utils.secrets_utils import SecretsUtils
from src.utils.yaml_parser import YamlParser
from src.utils.gcs_file_handler import GcsFileHandler
from src.utils.text_processing_utils import TextProcessingUtils


class PaperSummarizer:
    def __init__(self, model_name: str, vector_db_url: str):
        self.embed_utils = EmbeddingModelUtils(vector_db_url)
        self.qdrant_client = self.embed_utils.vector_db_client
        self.collection = self.embed_utils._collection_name
        self.embedding_model = self.embed_utils.embedding_model

        # load the tokenizer and the model
        self.hf_token = SecretsUtils().get_secret(secret_id="HUGGINGFACE_TOKEN")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype="auto", device_map="auto"
        )

        # Text processing utils
        self._text_processing = TextProcessingUtils()

        # Load configuration from YAML
        self._config = YamlParser("./config.yaml")

        # Setup GCS paths
        self._local_dataset_directory = "downloads/json"
        self._gcs_bucket_name = self._config.get_field("gcp.gcs.buckets")[0]["name"]
        self._gcs_data_directory = self._config.get_field("gcp.gcs.buckets")[0]["paths"]["data"]
        self._gcs_file_handler = GcsFileHandler(self._gcs_bucket_name)

        # Download JSON files
        self._gcs_file_handler.download_docling_json_files(self._gcs_data_directory)

    def retrieve_document(self, entry_id: str):
        print("Start here\n")

        file = os.path.join(self._local_dataset_directory, f"{entry_id}.json")
        doc = DoclingDocument.load_from_json(file)
        doc_md = doc.export_to_markdown()
        final_md = self._text_processing.clean_and_wrap_markdown(doc_md)

        prompt = f"Provide a 3-5 paragraph summary of this arxiv paper: \n{final_md}"
        messages = [{"role": "user", "content": prompt}]
        print("retrieved message")

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        print("applied chat template")

        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        print("tokenized text")

        generated_ids = self.model.generate(**model_inputs, max_new_tokens=1000)
        print("generated ids")

        output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()
        print("output ids")

        try:
            # rindex finding 151668 (</think>)
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0

        # thinking_content = self.tokenizer.decode(
        #     output_ids[:index], skip_special_tokens=True
        # ).strip("\n")
        # print("thinking content done")

        content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
        print("content done")

        # print("\nthinking content:", thinking_content)
        # print("content:", content)
        return content
