"""
TCT Token Constants for Kubernetes

Defines TCT-specific token IDs and vocabulary sizes for Kubernetes manifests.
These constants are used across the TCT tokenization and training pipeline.
"""

# Base vocabulary size (before special tokens)
TCT_K8S_BASE_VOCAB_SIZE = 19999

# Special token IDs
TCT_K8S_PAD_TOKEN_ID = 19999   # Used for padding sequences to fixed length

# Total vocabulary size (includes base + special tokens)
TCT_K8S_TOTAL_VOCAB_SIZE = 20000


def get_vocab_info():
    """
    Get a dictionary with all TCT Kubernetes vocabulary information.

    Returns:
        dict: Contains base_vocab_size, pad_token_id, total_vocab_size
    """
    return {
        "base_vocab_size": TCT_K8S_BASE_VOCAB_SIZE,
        "pad_token_id": TCT_K8S_PAD_TOKEN_ID,
        "total_vocab_size": TCT_K8S_TOTAL_VOCAB_SIZE,
    }
