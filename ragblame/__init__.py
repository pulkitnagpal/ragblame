"""ragblame: find which RAG stage caused each wrong answer."""
import os

# FAISS and PyTorch each ship their own copy of the OpenMP threading library on macOS. When both
# are loaded in one process, their worker threads can crash Python (a segfault in libomp).
# One thread for OpenMP avoids it; FAISS and the models are fast enough here without it.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

__version__ = "0.1.0"
