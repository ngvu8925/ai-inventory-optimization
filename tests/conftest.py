"""Global pytest configuration & fixtures."""

import os

# Thiet lap che do offline cho toan bo test suite de tranh network retry timeout
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
