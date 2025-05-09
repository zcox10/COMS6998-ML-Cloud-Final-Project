from fastapi import FastAPI
from pydantic import BaseModel
from src.rag.paper_summarizer import PaperSummarizer


class InferenceRequest(BaseModel):
    entry_id: str


# API
app = FastAPI()
summarizer = PaperSummarizer()


@app.get("/")
def health_check():
    return {"message": "ML Cloud API is up and running!"}


@app.post("/summarize")
def infer(request: InferenceRequest):
    return {"response": summarizer.summarize(request.entry_id), "input": request.entry_id}
