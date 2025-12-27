# vLLM quickstart (local install + model run)

This note is a standalone, educational guide for installing vLLM, downloading a model from Hugging Face, and running a local server.

## Prerequisites
- Use a Python version supported by your vLLM release. Python 3.12 is used in the examples below, but it is not strictly required.
- A Hugging Face account and access token if you need gated models.

## 1) Create and activate a virtual environment
Optional (macOS with Homebrew): install Python 3.12 first
```bash
brew install python@3.12
```

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

## 2) Install vLLM (and the Hugging Face CLI)
```bash
pip install vllm huggingface_hub
```

## 3) Log in to Hugging Face
```bash
hf auth login
```

## 4) Download a model
Example: Qwen/Qwen2.5-0.5B-Instruct
```bash
hf download Qwen/Qwen2.5-0.5B-Instruct --repo-type model
```

## 5) Run the vLLM server
```bash
vllm serve Qwen/Qwen2.5-0.5B-Instruct
```

## 6) Query the API (OpenAI-compatible)
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "messages": [{"role":"user","content":"Say hello in one sentence."}],
    "temperature": 0.2
  }'
```
