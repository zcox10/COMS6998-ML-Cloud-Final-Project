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
  5. **fine_tune_model**: Uses the Markdown dataset to fine‑tune a target LLM model on domain‑specific summaries.

## Pipeline Components

### 1. arxiv_data_collection

- **Function**: `arxiv_data_collection` component
- **Action**: Queries the arXiv API for recent papers, extracts metadata fields (title, entry_id, timestamps, summary, categories, DOI, URLs) and writes per‑paper JSON to a GCS bucket (metadata only, no raw PDFs).

### 2. docling_pdf_processing

- **Class**: `DoclingPdfProcessing`
- **Action**: Reads metadata JSON paths, downloads corresponding PDFs from `https://arxiv.org/pdf/{entry_id}`, runs Docling’s PDF converter (OCR, table extraction, formula/code enrichment), saves JSON assets locally, and uploads the asset directories to GCS.

### 3. generate_fine_tune_dataset

- **Class**: `GenerateDataset`
- **Action**: Downloads all Docling JSON files from GCS, loads each as a `DoclingDocument`, exports to Markdown, applies text‑cleaning utilities, concatenates into a single `.md` file, and uploads the final dataset back to GCS.

### 4. embed_text_chunks

- **Class**: `VectorEmbeddings`
- **Action**: Fetches Docling JSON and associated metadata from GCS, uses LangChain’s `RecursiveCharacterTextSplitter` (800‑token chunks, 100‑token overlap) to segment content, wraps each chunk in a LangChain `Document` with metadata, and calls `EmbeddingModelUtils.upsert_document_embedding`.

### 5. fine_tune_model

- **Class**: `TBD`
- **Action**: Needs to be written...

## EmbeddingModelUtils & Qdrant Integration

- **EmbeddingModelUtils**: Initializes `GoogleGenerativeAIEmbeddings` (with key from Google Secret Manager), measures embedding size, and (re)creates a Qdrant collection using `qdrant_client`.  
- **Upsert Logic**: Deterministic UUIDs (`uuid5` of `entry_id` + chunk index) ensure idempotent writes.  
- **Distance Metric**: COSINE similarity.

## arxiv-summarization-api Service

- **Namespace**: `arxiv-summarization-api`  
- **Functionality**: Receives an arXiv identifier, retrieves relevant embeddings from Qdrant, constructs a RAG prompt, calls the LLM, and returns a concise paper summary over HTTP (via FastAPI).

## Getting Started

### Prerequisites

- GCP account with:
  - Secret Manager access for `GOOGLE_GEMINI_KEY`  
  - GCS bucket configured in `config.yaml`  
- Kubernetes cluster with Kubeflow installed  
- `kubectl`, `kfctl`, and Terraform  
- Python 3.9+ and Docker

### Setup

1. Create a new Google Cloud Project

2. Create a service account with the necessary permissions to create service accounts, create storage buckets, configure network settings, and creating GKE clusters / configuring namespaces. This service account will run the Terraform setup by setting IAM permissions, creating service accounts for Kubernetes, setting up a GCS storage bucket, configuring network settings, creating a GKE cluster, and configuring Kubernetes namespaces for Kubeflow, Qdrant, and API serving.

3. Configure `config.yaml` with your GCP project, GCS bucket names, paths, and Qdrant endpoint.

4. Install `terraform` and `helm` via brew (on Mac) with the following commands

  ```bash
  brew update && brew upgrade # optional, but recommended
  brew tap hashicorp/tap
  brew install hashicorp/tap/terraform
  brew install helm # for qdrant vector db installation
  ```

5. Run the Kubernetes setup script from root with the following commands

   ```bash
   chmod +x ./scripts/setup/kubernetes_setup.sh
   ./scripts/kubernetes_setup.sh
   ```

6. Run Kubeflow Pipeline with:

   ```bash
   chmod +x ./scripts/kubeflow/run_kubeflow_pipeline.sh
   ./scripts/kubeflow/run_kubeflow_pipeline.sh
   ```
