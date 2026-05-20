import importlib, os
from dotenv import load_dotenv

load_dotenv()

# Project root is one level above this file (src/)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LLM_PROVIDER  = os.getenv("LLM_PROVIDER",  "ollama")
LLM_MODEL     = os.getenv("LLM_MODEL",     "qwen2.5:1.5b")
EMBED_PROVIDER = os.getenv("EMBED_PROVIDER", "ollama")
EMBED_MODEL   = os.getenv("EMBED_MODEL",   "nomic-embed-text")
CHROMA_DIR    = os.getenv("CHROMA_DIR",    os.path.join(_PROJECT_ROOT, "chroma_db"))

# Registry: provider → (langchain module, class name, extra kwargs)
# model=LLM_MODEL / model=EMBED_MODEL is injected automatically by the build functions
LLM_REGISTRY = {
    "ollama":    ("langchain_ollama",    "OllamaLLM",     {}),
    "anthropic": ("langchain_anthropic", "ChatAnthropic", {"api_key": os.getenv("ANTHROPIC_API_KEY")}),
    "openai":    ("langchain_openai",    "ChatOpenAI",    {"api_key": os.getenv("OPENAI_API_KEY")}),
    "deepseek":  ("langchain_openai",    "ChatOpenAI",    {"api_key": os.getenv("DEEPSEEK_API_KEY"),
                                                           "base_url": "https://api.deepseek.com"}),
}

EMBED_REGISTRY = {
    "ollama": ("langchain_ollama", "OllamaEmbeddings", {}),
    "openai": ("langchain_openai", "OpenAIEmbeddings", {"api_key": os.getenv("OPENAI_API_KEY")}),
}


def build_llm():
    if LLM_PROVIDER not in LLM_REGISTRY:
        raise ValueError(f"Unknown LLM provider '{LLM_PROVIDER}'. Available: {list(LLM_REGISTRY)}")
    module_name, class_name, extra_kwargs = LLM_REGISTRY[LLM_PROVIDER]
    cls = getattr(importlib.import_module(module_name), class_name)
    return cls(model=LLM_MODEL, **extra_kwargs)


def build_embeddings():
    if EMBED_PROVIDER not in EMBED_REGISTRY:
        raise ValueError(f"Unknown embed provider '{EMBED_PROVIDER}'. Available: {list(EMBED_REGISTRY)}")
    module_name, class_name, extra_kwargs = EMBED_REGISTRY[EMBED_PROVIDER]
    cls = getattr(importlib.import_module(module_name), class_name)
    return cls(model=EMBED_MODEL, **extra_kwargs)
