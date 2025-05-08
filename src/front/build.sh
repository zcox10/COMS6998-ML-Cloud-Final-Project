#! /bin/bash

docker buildx build \
  --file Dockerfile \
  --tag gcr.io/zsc-personal/pdfusion-app:latest \
  --cache-from=type=registry,ref=gcr.io/zsc-personal/pdfusion-app:cache \
  --cache-to=type=registry,ref=gcr.io/zsc-personal/pdfusion-app:cache,mode=max \
  --push \
  --platform linux/amd64 .
