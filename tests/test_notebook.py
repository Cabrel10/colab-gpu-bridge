import ast
import json
import re
from pathlib import Path

ROOT = Path(__file__).parents[1]
NOTEBOOK = ROOT / "notebooks" / "genspark_code.ipynb"
LEGACY_NOTEBOOK = ROOT / "notebooks" / "RavenX_Chaos_Agent_Colab.ipynb"
OPENCODE_CONFIG = ROOT / "opencode.json"
README = ROOT / "README.md"


def load_notebook():
    return json.loads(NOTEBOOK.read_text())


def code_text():
    return "\n".join(
        "".join(cell["source"])
        for cell in load_notebook()["cells"]
        if cell["cell_type"] == "code"
    )


def test_notebook_structure_and_syntax():
    nb = load_notebook()
    assert nb["nbformat"] == 4
    assert nb["metadata"]["accelerator"] == "GPU"
    code = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    assert len(code) == 8
    assert all(cell.get("id") for cell in nb["cells"])
    # La cible garde les identifiants du notebook joint d'origine.
    assert nb["cells"][0]["id"] == "mZNJ9f0WCQKA"
    assert nb["cells"][-1]["id"] == "XSNVrwSVCQKV"
    for source in code:
        ast.parse(source)


def test_single_exact_model_and_single_python_instance():
    text = code_text()
    assert "RavenX-Chaos-Agent-Q4_K_M.gguf" in text
    assert "deadbydawn101/RavenXAiLabs" in text
    assert "from llama_cpp import Llama" in text
    assert text.count("Llama.from_pretrained(") == 1
    assert "llm.create_chat_completion(" in text
    assert "hf_hub_download" not in text
    assert "llama-server" not in text
    assert "mistral-24b" not in text.lower()
    assert "huihui-qwen35" not in text.lower()
    assert "qwen3.5" not in text.lower()
    assert "MODELS_LIST" not in text
    assert "MODEL_CHOICE" not in text


def test_attached_notebook_is_the_canonical_target():
    assert NOTEBOOK.exists()
    assert LEGACY_NOTEBOOK.exists()
    assert NOTEBOOK.name == "genspark_code.ipynb"


def test_only_two_required_colab_secrets():
    text = code_text()
    requested = set(re.findall(r'userdata\.get\(["\']([^"\']+)["\']\)', text))
    assert requested == {"hex", "vps"}
    assert "API_KEY = HB_SECRET" in text
    assert "print(API_KEY)" not in text
    assert "userdata.get(\"api_key\")" not in text
    assert "userdata.get(\"account\")" not in text
    assert '"RAVENX_API_KEY": "<valeur du secret Colab hex>"' in text


def test_security_runtime_and_gpu_guards():
    text = code_text()
    assert "EXPECTED_SIZE = 16_547_399_968" in text
    assert "llama_supports_gpu_offload" in text
    assert "GPU_LAYER_CANDIDATES" in text
    assert "VRAM_GIB < 18" in text
    assert "VRAM_GIB < 35" in text
    assert "GENERATION_LOCK" in text
    assert "hmac.compare_digest" in text
    assert "hmac.new" in text
    assert '"X-Signature": signature' in text
    assert '"Authorization": f"Bearer {API_KEY}"' in text


def test_openai_routes_and_live_checks_are_present():
    text = code_text()
    assert '@app.get("/health")' in text
    assert '@app.get("/v1/models")' in text
    assert '@app.post("/v1/chat/completions")' in text
    assert "StreamingResponse" in text
    assert '"data: [DONE]"' in text
    assert "models_response.raise_for_status()" in text
    assert "chat_response.raise_for_status()" in text
    assert "stream_ok" in text


def test_cloudflare_heartbeat_and_keepalive_are_automatic():
    text = code_text()
    assert "trycloudflare" in text
    assert "send_heartbeat()" in text
    assert "HEARTBEAT_THREAD" in text
    assert "while SERVER_THREAD.is_alive()" in text
    assert '"status": "READY"' in text


def test_heartbeat_reloads_and_validates_the_live_tunnel_url():
    text = code_text()
    assert 'Path("cloudflare_url.txt")' in text
    assert 'ROOT / "cloudflare_url.txt"' in text
    assert 'globals().get("PUBLIC_URL") or globals().get("public_url")' in text
    assert 'url + "/v1/models"' in text
    assert "validate_tunnel(PUBLIC_URL)" in text
    assert "PUBLIC_URL = resolve_public_url()" in text
    assert "survival-border-away-favors" not in text


def test_opencode_configuration_is_valid_and_secret_free():
    config = json.loads(OPENCODE_CONFIG.read_text())
    assert config["model"] == "ravenx/ravenx-chaos-agent-qwen3.8-27b"
    provider = config["provider"]["ravenx"]
    assert provider["npm"] == "@ai-sdk/openai-compatible"
    assert provider["options"]["baseURL"] == "{env:RAVENX_BASE_URL}"
    assert provider["options"]["apiKey"] == "{env:RAVENX_API_KEY}"
    assert "ravenx-chaos-agent-qwen3.8-27b" in provider["models"]
    context7 = config["mcp"]["context7"]
    assert context7 == {
        "type": "remote",
        "url": "https://mcp.context7.com/mcp",
        "enabled": True,
        "timeout": 15000,
    }
    serialized = OPENCODE_CONFIG.read_text().lower()
    assert "trycloudflare.com" not in serialized
    assert "bearer " not in serialized


def test_readme_documents_colab_opencode_ssh_and_mcp():
    text = README.read_text()
    assert "colab.research.google.com/github/Cabrel10/colab-gpu-bridge" in text
    assert "opencode . --model ravenx/ravenx-chaos-agent-qwen3.8-27b" in text
    assert "ssh user@vps" in text
    assert "opencode mcp list" in text
    assert "RAVENX_BASE_URL" in text
    assert "RAVENX_API_KEY" in text
