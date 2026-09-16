"""Provider defaults shared by the config, the embedders and the setup wizard.

This module has no internal imports so it can be used from anywhere without
creating an import cycle.
"""

DEFAULT_PROVIDER = "openai"
DEFAULT_OPENAI_MODEL = "text-embedding-3-small"
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "nomic-embed-text"
DEFAULT_LMSTUDIO_URL = "http://localhost:1234"
DEFAULT_LMSTUDIO_MODEL = "text-embedding-nomic-embed-text-v1.5"
