#!/bin/bash
# Lightning AI Studio on_start.sh (4-bit Edition)
# Runs automatically when the studio starts or restarts

echo "🚀 WfAgent Pro startup script (4-bit)"

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# Pre-download the tokenizer to avoid first-run delay
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('Qwen/Qwen2.5-Coder-7B-Instruct', trust_remote_code=True)" || true

echo "✅ Startup complete. Run: python app.py"
