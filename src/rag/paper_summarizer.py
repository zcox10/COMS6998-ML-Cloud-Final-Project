import os
from typing import List
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.utils.secrets_utils import SecretsUtils
from src.utils.yaml_parser import YamlParser
from src.utils.gcs_file_handler import GcsFileHandler
from src.utils.arxiv_utils import ArxivUtils
from src.utils.text_processing_utils import TextProcessingUtils
from src.utils.local_file_handler import LocalFileHandler


class PaperSummarizer:
    def __init__(self):
        self._arxiv_utils = ArxivUtils()
        self._text_processing = TextProcessingUtils()
        self._text_processing.process_docling_files_to_markdown()
        self._local_markdown_directory = "downloads/markdown"
        self._local_file_handler = LocalFileHandler()

        # load tokenizer and model
        model_name = "Qwen/Qwen3-0.6B"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype="auto", device_map="auto"
        )
        logging.info(f"Model device: {next(self.model.parameters()).device}")

    def summarize(self, entry_id: str):
        formatted_entry_id = self._arxiv_utils.extract_formatted_entry_id_from_url(entry_id)
        logging.info(f"Run `summarize` on {formatted_entry_id}")

        file = os.path.join(self._local_markdown_directory, f"{formatted_entry_id}.md")
        md_file = self._local_file_handler.load_file(file)

        prompt = f"Provide a 5-10 paragraph summary of this arxiv paper: \n{md_file}"
        messages = [{"role": "user", "content": prompt}]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )

        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(**model_inputs, max_new_tokens=20000)
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()
        logging.info(f"{formatted_entry_id} output generated")

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
        return content
