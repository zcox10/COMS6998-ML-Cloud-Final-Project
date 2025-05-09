# Kubeflow RAG Pipeline & arXiv Summarization API

This repository provides an end‑to‑end framework for ingesting, processing, and summarizing scientific papers from arXiv using a Kubeflow pipeline on Kubernetes, a Qdrant vector database for retrieval‑augmented generation (RAG), and a standalone API service for delivering LLM‑generated summaries.

## Architecture Overview

- **Kubernetes Namespaces**  
  - `vector-db`: Hosts a Qdrant collection for storing and querying text embeddings.  
  - `arxiv-summarization-api`: Runs an HTTP service that accepts paper identifiers and returns LLM‑driven summaries.
  - `kubeflow`: Runs all Kubeflow pipelines.
- **Kubeflow Pipeline**  
  1. **arxiv_data_collection**: Scrapes metadata from arXiv (via the `arxiv` Python package) and writes JSON metadata files to GCS.  
  2. **docling_pdf_processing**: Converts PDF documents into Docling JSON assets (text, tables, images) using OCR and enrichment, then uploads them to GCS.  
  3. **generate_fine_tune_dataset**: Aggregates Docling JSON, cleans and wraps text, and produces a combined Markdown file for fine‑tuning.  
  4. **embed_text_chunks**: Splits documents into overlapping text chunks, computes embeddings via Google Generative AI embeddings (model `models/embedding-001`), and upserts them into Qdrant.  

## Pipeline Components

### 1. arxiv_data_collection

- **Function**: `arxiv_data_collection` component
- **Action**: Queries the arXiv API for recent papers, extracts metadata fields (title, entry_id, timestamps, summary, categories, DOI, URLs) and writes per‑paper JSON to a GCS bucket (metadata only, no raw PDFs).

### 2. docling_pdf_processing

- **Class**: `DoclingPdfProcessing`
- **Action**: Reads metadata JSON paths, downloads corresponding PDFs from `https://arxiv.org/pdf/{entry_id}`, runs Docling's PDF converter (OCR, table extraction, formula/code enrichment), saves JSON assets locally, and uploads the asset directories to GCS.

### 3. generate_fine_tune_dataset

- **Class**: `GenerateDataset`
- **Action**: Downloads all Docling JSON files from GCS, loads each as a `DoclingDocument`, exports to Markdown, applies text‑cleaning utilities, concatenates into a single `.md` file, and uploads the final dataset back to GCS.

### 4. embed_text_chunks

- **Class**: `VectorEmbeddings`
- **Action**: Fetches Docling JSON and associated metadata from GCS, uses LangChain's `RecursiveCharacterTextSplitter` (800‑token chunks, 100‑token overlap) to segment content, wraps each chunk in a LangChain `Document` with metadata, and calls `EmbeddingModelUtils.upsert_document_embedding`.

## EmbeddingModelUtils & Qdrant Integration

- **EmbeddingModelUtils**: Initializes `GoogleGenerativeAIEmbeddings` (with key from Google Secret Manager), measures embedding size, and (re)creates a Qdrant collection using `qdrant_client`.  
- **Upsert Logic**: Deterministic UUIDs (`uuid5` of `entry_id` + chunk index) ensure idempotent writes.  
- **Distance Metric**: COSINE similarity.

## arxiv-summarization-api Service

- **Namespace**: `arxiv-summarization-api`  
- **Functionality**: Receives an arXiv identifier, retrieves relevant embeddings from Qdrant, constructs a RAG prompt, calls the LLM, and returns a concise paper summary over HTTP (via FastAPI).

## LLM Training & Inference Setup

### LLaMA Factory Setup

LLaMA Factory is used for fine-tuning language models in this project.

#### Prerequisites
- GPU with CUDA support
- Verify installation with `nvidia-smi`

#### Installation Steps
```bash
git clone https://github.com/hiyouga/LLaMA-Factory.git
conda create -n llama_factory python=3.10
conda activate llama_factory
cd LLaMA-Factory
pip install -e '.[torch,metrics]'
```

#### Verification
```python
import torch
torch.cuda.current_device()
torch.cuda.get_device_name(0)
torch.__version__
```

You can also verify with the CLI tools:
```bash
llamafactory-cli train -h
llamafactory-cli webui
```

### Qwen3-4B Inference API & Frontend

This section explains how to serve your fine-tuned model with a FastAPI backend and simple HTML frontend.

#### 1. Merge LoRA Weights
To make inference easier, merge LoRA weights with the base model:
```bash
mkdir -p Models/qwen3-4b-merged
# Export merged model using your training framework (e.g., LLaMA-Factory)
```

#### 2. Set Up FastAPI Service

Create and activate environment:
```bash
conda create -n fastapi-qwen python=3.10
conda activate fastapi-qwen
conda install -c conda-forge fastapi uvicorn transformers pytorch
pip install safetensors sentencepiece protobuf
```

Sample code how to use trained model
```python

# Load model
model_path = "/root/autodl-tmp/Models/qwen3-4b-merged"
tokenizer = AutoTokenizer.from_pretrained(model_path)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(model_path).to(device)

@app.get("/generate")
async def generate_text(prompt: str):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(inputs["input_ids"], max_length=4096)
    return {"generated_text": tokenizer.decode(outputs[0], skip_special_tokens=True)}
```


## Getting Started

### Prerequisites

- GCP account with:
  - Secret Manager access for `GOOGLE_GEMINI_KEY`  
  - GCS bucket configured in `config.yaml`  
- Kubernetes cluster with Kubeflow installed  
- `kubectl`, `kfctl`, and Terraform  
- Python 3.9+ and Docker
- GPU with CUDA support (for LLM training and inference)

### Setup

1. Create a new Google Cloud Project

2. Create a service account with the necessary permissions to create service accounts, create storage buckets, configure network settings, accessing Secrets Manager, and creating GKE clusters / configuring namespaces. This service account will run the Terraform setup by setting IAM permissions, creating service accounts for Kubernetes, setting up a GCS storage bucket, configuring network settings, creating a GKE cluster, and configuring Kubernetes namespaces for Kubeflow, Qdrant, and API serving.

3. Configure `config.yaml` with your GCP project, GCS bucket names, paths, and Qdrant endpoint.

4. Enable the Gemini API, generate an API key on [aistudio.google.com](https://aistudio.google.com/), and store the API key in the project's Secrets Manger. Name the secret `GOOGLE_GEMINI_KEY`.

5. Install `terraform` and `helm` via brew (on Mac) with the following commands

  ```bash
  brew update && brew upgrade # optional, but recommended
  brew tap hashicorp/tap
  brew install hashicorp/tap/terraform
  brew install helm # for qdrant vector db installation
  ```

6. Run the Kubernetes setup script from root with the following commands

   ```bash
   chmod +x ./scripts/setup/kubernetes_setup.sh
   ./scripts/kubernetes_setup.sh
   ```

7. Run Kubeflow Pipeline with:

   ```bash
   chmod +x ./scripts/kubeflow/run_kubeflow_pipeline.sh
   ./scripts/kubeflow/run_kubeflow_pipeline.sh
   ```

8. For LLM fine-tuning setup, follow the LLaMA Factory installation steps in the section above.

9. For model serving, follow the Qwen3-4B Inference API & Frontend setup instructions.
