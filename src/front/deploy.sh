#! /bin/bash
kubectl delete -f pdfusion_app.yaml -n arxiv-summarization-api
kubectl apply -f pdfusion_app.yaml -n arxiv-summarization-api
