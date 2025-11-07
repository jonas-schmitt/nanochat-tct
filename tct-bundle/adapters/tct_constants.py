"""
TCT Token Constants

Defines TCT-specific token IDs and vocabulary sizes for GitHub Actions workflows.
These constants are used across the TCT tokenization and training pipeline.
"""

# Base vocabulary size (before special tokens)
TCT_BASE_VOCAB_SIZE = 8190

# Special token IDs
TCT_MASK_TOKEN_ID = 8190  # Used for FIM (Fill-in-the-Middle) masking
TCT_PAD_TOKEN_ID = 8191   # Used for padding sequences to fixed length

# Total vocabulary size (includes base + special tokens)
TCT_TOTAL_VOCAB_SIZE = 8192


def get_vocab_info():
    """
    Get a dictionary with all TCT vocabulary information.

    Returns:
        dict: Contains base_vocab_size, mask_token_id, pad_token_id, total_vocab_size
    """
    return {
        "base_vocab_size": TCT_BASE_VOCAB_SIZE,
        "mask_token_id": TCT_MASK_TOKEN_ID,
        "pad_token_id": TCT_PAD_TOKEN_ID,
        "total_vocab_size": TCT_TOTAL_VOCAB_SIZE,
    }
