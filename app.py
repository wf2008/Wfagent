"""
WfAgent Pro — Lightning AI Studio Edition (4-bit Quantized)
Self-hosted Qwen2.5-Coder-7B-Instruct (4-bit) with live sandbox + zero-key web search.
Optimized for Lightning AI Studio GPU environments (L40S, A10G, etc.)

Deployment:
    1. Upload app.py + requirements.txt to Lightning AI Studio
    2. Install the Gradio plugin in the Studio UI
    3. Run: python app.py
    4. Expose port 8080 via the Studio "Expose" button or Gradio plugin
"""

import json
import os
import re
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from threading import Thread
from typing import List, Optional

import gradio as gr
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
    BitsAndBytesConfig,
)

# ─────────────────────────────────────────────────────────────────────────────
# 0.  LIGHTNING AI STUDIO COMPATIBILITY
# ─────────────────────────────────────────────────────────────────────────────
PORT = int(os.environ.get("PORT", "8080"))
HOST = os.environ.get("HOST", "0.0.0.0")

CACHE_DIR = os.environ.get("HF_HOME", "/tmp/hf_cache")
Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)

SANDBOX_DIR = os.path.join(tempfile.gettempdir(), "wfagent_sandbox")
Path(SANDBOX_DIR).mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1.  LAZY MODEL LOADING (4-bit Quantized)
# ─────────────────────────────────────────────────────────────────────────────
MODEL_NAME: str = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-Coder-7B-Instruct")

_model: Optional[AutoModelForCausalLM] = None
_tokenizer: Optional[AutoTokenizer] = None
_model_lock = threading.Lock()


def load_model() -> None:
    """Thread-safe lazy loader for the 4-bit quantized causal LM."""
    global _model, _tokenizer
    with _model_lock:
        if _model is not None:
            return
        print(f"🧠 Loading {MODEL_NAME} (4-bit / NF4) …")

        _tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            cache_dir=CACHE_DIR,
        )

        # 4-bit quantization config — ~4 GB VRAM instead of ~14-16 GB
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        _model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            cache_dir=CACHE_DIR,
            torch_dtype=torch.bfloat16,
        )
        print("✅ Model ready (4-bit).")


# ─────────────────────────────────────────────────────────────────────────────
# 2.  LIVE SANDBOX – Execute Python line‑by‑line
# ─────────────────────────────────────────────────────────────────────────────
def live_sandbox_exec(code: str, sandbox_dir: str) -> str:
    """Run Python code in a subprocess and capture stdout/stderr."""
    log_lines: List[str] = []

    def _io_thread(pipe, prefix: str):
        try:
            for line in iter(pipe.readline, ""):
                line = line.rstrip()
                if line:
                    log_lines.append(f"{prefix}{line}")
        except Exception:
            pass

    try:
        proc = subprocess.Popen(
            [sys.executable, "-u", "-c", code],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            cwd=sandbox_dir,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
    except Exception as exc:
        return f"❌ Failed to start subprocess: {exc}"

    t_out = Thread(target=_io_thread, args=(proc.stdout, ""), daemon=True)
    t_err = Thread(target=_io_thread, args=(proc.stderr, "STDERR: "), daemon=True)
    t_out.start()
    t_err.start()

    try:
        proc.wait(timeout=30)
        t_out.join(timeout=2)
        t_err.join(timeout=2)
    except subprocess.TimeoutExpired:
        proc.kill()
        log_lines.append("⏱ Timeout – code ran longer than 30 s and was killed.")

    return "\n".join(log_lines) if log_lines else "✅ Code ran with no output."


# ─────────────────────────────────────────────────────────────────────────────
# 3.  WEB SEARCH (Zero‑API‑Key) – DuckDuckGo via zero‑api‑key‑web‑search
# ─────────────────────────────────────────────────────────────────────────────
def web_search(query: str, max_results: int = 5) -> str:
    """Run zero-api-key-web-search and return formatted top results."""
    try:
        result = subprocess.run(
            ["zero-search", query, "--max-results", str(max_results)],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode != 0:
            return f"Search error: {result.stderr.strip()}"

        data = json.loads(result.stdout)
        if not data:
            return "No results found."

        formatted = []
        for i, entry in enumerate(data, 1):
            title = entry.get("title", "No title")
            snippet = entry.get("snippet", "No snippet")
            url = entry.get("url", "")
            formatted.append(f"{i}. {title}\n   {snippet}\n   {url}")
        return "\n\n".join(formatted)
    except subprocess.TimeoutExpired:
        return "Search timed out."
    except Exception as e:
        return f"Search failed: {e}"


# ─────────────────────────────────────────────────────────────────────────────
# 4.  BLOCK PARSER – Robust fenced‑block extractor
# ─────────────────────────────────────────────────────────────────────────────
CODE_BLOCK_RE = re.compile(
    r"```(?:python|py)?\s*\n(.*?)\n```",
    re.DOTALL | re.IGNORECASE,
)
SEARCH_BLOCK_RE = re.compile(
    r"```search\s*\n(.*?)\n```",
    re.DOTALL | re.IGNORECASE,
)


def extract_blocks(text: str):
    """Extract code and search blocks from assistant text."""
    code_blocks = CODE_BLOCK_RE.findall(text)
    search_blocks = SEARCH_BLOCK_RE.findall(text)
    return code_blocks, search_blocks


# ─────────────────────────────────────────────────────────────────────────────
# 5.  AGENT LOOP – Streaming + per‑block execution + web search
# ─────────────────────────────────────────────────────────────────────────────
def agent_chat(
    message: str,
    history: list,
    system_prompt: str,
    max_iter: int,
):
    load_model()

    transcript: List[dict] = [{"role": "system", "content": system_prompt}]

    for turn in history:
        if isinstance(turn, (list, tuple)) and len(turn) >= 2:
            transcript.append({"role": "user", "content": turn[0]})
            if turn[1]:
                transcript.append({"role": "assistant", "content": turn[1]})

    transcript.append({"role": "user", "content": message})
    history.append([message, ""])

    iteration = 0
    pending_feedback: Optional[str] = None
    feedback_type: Optional[str] = None

    while iteration < max_iter:
        iteration += 1

        if pending_feedback:
            if feedback_type == "error":
                transcript.append({
                    "role": "user",
                    "content": f"Execution error:\n{pending_feedback}\nFix ONLY that block."
                })
            elif feedback_type == "search":
                transcript.append({
                    "role": "user",
                    "content": f"Web search results:\n{pending_feedback}\nNow continue with your original task."
                })
            pending_feedback = None
            feedback_type = None

        text_prompt = _tokenizer.apply_chat_template(
            transcript, tokenize=False, add_generation_prompt=True
        )
        inputs = _tokenizer(text_prompt, return_tensors="pt").to(_model.device)

        streamer = TextIteratorStreamer(
            _tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        gen_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=2048,
            do_sample=True,
            temperature=0.2,
            top_p=0.95,
        )
        thread = Thread(target=_model.generate, kwargs=gen_kwargs)
        thread.start()

        reply_text = ""
        for new_token in streamer:
            reply_text += new_token
            history[-1] = [message, reply_text]
            yield history, SANDBOX_DIR, "⏳ Generating…"

        code_blocks, search_blocks = extract_blocks(reply_text)

        if search_blocks:
            query = search_blocks[0].strip()
            status_msg = f"🔍 Searching the web for: {query[:60]}…"
            history[-1] = [message, reply_text + f"\n\n*({status_msg})*"]
            yield history, SANDBOX_DIR, status_msg

            search_result = web_search(query)
            pending_feedback = search_result
            feedback_type = "search"
            transcript.append({"role": "assistant", "content": reply_text})
            continue

        if code_blocks:
            code = code_blocks[0].strip()
            status_msg = "⚡ Executing block…"
            history[-1] = [message, reply_text + f"\n\n*({status_msg})*"]
            yield history, SANDBOX_DIR, status_msg

            output = live_sandbox_exec(code, SANDBOX_DIR)

            if "Error" in output or "Traceback" in output or "STDERR" in output:
                status_msg = "❌ Block failed – fixing…"
                history[-1] = [message, reply_text + f"\n\n```output\n{output}\n```"]
                yield history, SANDBOX_DIR, status_msg
                pending_feedback = output
                feedback_type = "error"
                transcript.append({"role": "assistant", "content": reply_text})
                continue
            else:
                status_msg = "✅ Block executed successfully"
                history[-1] = [message, reply_text + f"\n\n```output\n{output}\n```"]
                yield history, SANDBOX_DIR, status_msg
                transcript.append({"role": "assistant", "content": reply_text})
                transcript.append({"role": "user", "content": f"Output:\n{output}"})
                continue

        transcript.append({"role": "assistant", "content": reply_text})
        break

    yield history, SANDBOX_DIR, "✅ Done"


# ─────────────────────────────────────────────────────────────────────────────
# 6.  GRADIO UI
# ─────────────────────────────────────────────────────────────────────────────
CUSTOM_CSS = """
:root {
    --bg: #0d1117; --surface: #161b22; --surface2: #1c2128;
    --border: #30363d; --accent: #6366f1; --text: #e6edf3; --muted: #8b949e;
}
body, .gradio-container {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Segoe UI', sans-serif !important;
}
.app-header {
    display: flex; align-items: center; gap: 12px;
    padding: 1rem 1.5rem;
    background: linear-gradient(135deg, rgba(99,102,241,0.12), rgba(167,139,250,0.08));
    border: 1px solid rgba(99,102,241,0.25); border-radius: 16px; margin-bottom: 1rem;
}
.brand-dot { width: 12px; height: 12px; background: #22c55e; border-radius: 50%; box-shadow: 0 0 10px #22c55e; animation: pulse 2s infinite; }
@keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.4; } }
footer { display: none !important; }
"""

with gr.Blocks(css=CUSTOM_CSS, title="WfAgent · Lightning AI Studio") as demo:
    gr.HTML(
        """
        <div class="app-header">
            <div class="brand-dot"></div>
            <div>
                <h2 style="margin:0;">WfAgent Pro</h2>
                <p style="margin:0; color:#8b949e; font-size:0.85rem;">
                    Self‑hosted Qwen2.5‑Coder‑7B (4-bit) · Live execution · Auto‑fix · Web search (zero‑key)
                </p>
            </div>
        </div>
        """
    )

    with gr.Tabs():
        with gr.TabItem("🤖 Agent Chat"):
            with gr.Row():
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(height=500, show_label=False, render_markdown=True, type="messages")
                    with gr.Row():
                        msg = gr.Textbox(
                            placeholder="Describe what to build, scrape, or automate…",
                            show_label=False, scale=8, lines=1,
                        )
                        send = gr.Button("Send ▶", variant="primary", scale=1)
                        clear = gr.Button("Clear", variant="stop", scale=1)
                with gr.Column(scale=1):
                    status_box = gr.Textbox(
                        label="⚡ Live Status", value="Ready", interactive=False, lines=4,
                    )
                    sandbox_log = gr.Textbox(
                        label="📁 Sandbox Directory",
                        value=SANDBOX_DIR,
                        interactive=False,
                    )
                    with gr.Accordion("⚙️ Settings", open=False):
                        system_prompt = gr.Textbox(
                            label="System Prompt",
                            value=(
                                "You are an elite autonomous coding agent with web search. "
                                "Write Python code inside ```python\n...\n``` blocks. "
                                "When you need fresh information, search the web using a ```search\nyour query\n``` block. "
                                "You will receive the results and can continue with that knowledge. "
                                "When you generate a code block, the system executes it immediately. "
                                "If it fails, you will receive the error and must fix ONLY that block."
                            ),
                            lines=6,
                        )
                        max_iter_slider = gr.Slider(
                            minimum=1, maximum=12, value=6, step=1, label="Max Fix Iterations",
                        )

            send.click(
                agent_chat,
                inputs=[msg, chatbot, system_prompt, max_iter_slider],
                outputs=[chatbot, sandbox_log, status_box],
            ).then(lambda: "", outputs=msg)
            msg.submit(
                agent_chat,
                inputs=[msg, chatbot, system_prompt, max_iter_slider],
                outputs=[chatbot, sandbox_log, status_box],
            ).then(lambda: "", outputs=msg)
            clear.click(lambda: ([], "Ready"), outputs=[chatbot, status_box])

        with gr.TabItem("🛠️ Code Studio"):
            gr.Markdown("### ✏️ Manual Python Sandbox")
            code_editor = gr.Code(
                value="# Write / paste Python code here and run it instantly.",
                language="python", lines=10,
            )
            run_btn = gr.Button("▶ Execute", variant="primary")
            code_output = gr.Textbox(label="Output", interactive=False, lines=8)
            run_btn.click(
                fn=lambda code: live_sandbox_exec(code, SANDBOX_DIR),
                inputs=code_editor,
                outputs=code_output,
            )

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1, max_size=10).launch(
        server_name=HOST,
        server_port=PORT,
        show_error=True,
        share=False,
        )
    
