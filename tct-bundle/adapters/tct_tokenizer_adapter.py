"""
TCT Tokenizer Adapter for Nanochat

Drop-in replacement for nanochat's RustBPETokenizer that uses TCT
for GitHub Actions workflow tokenization.

Usage:
    from tct_tokenizer_adapter import TCTTokenizer

    tokenizer = TCTTokenizer()
    tokens = tokenizer.encode(workflow_json)
    decoded = tokenizer.decode(tokens)
"""

from tct_github_workflow import (
    encode as tct_encode,
    decode as tct_decode,
    vocab_size as tct_vocab_size,
)


class TCTTokenizer:
    """
    Adapter to match nanochat's tokenizer interface using TCT.

    This adapter provides the same API as RustBPETokenizer but uses
    TCT's schema-aware tokenization for GitHub Actions workflows.
    """

    def __init__(self):
        """Initialize TCT tokenizer adapter"""
        self._vocab_size = tct_vocab_size()

    def encode(self, text, prepend=None, append=None):
        """
        Encode workflow JSON to tokens.

        Args:
            text: JSON string of GitHub Actions workflow
            prepend: Optional list of token IDs to prepend
            append: Optional list of token IDs to append

        Returns:
            List of token IDs
        """
        # Encode with TCT
        tokens = list(tct_encode(text))

        # Add prepend/append if provided
        if prepend is not None:
            tokens = list(prepend) + tokens
        if append is not None:
            tokens = tokens + list(append)

        return tokens

    def decode(self, ids):
        """
        Decode tokens back to workflow JSON.

        Args:
            ids: List of token IDs

        Returns:
            JSON string of workflow
        """
        json_str, consumed, total = tct_decode(list(ids))

        if consumed < total:
            print(f"Warning: {total - consumed} surplus tokens not consumed")

        return json_str

    def get_vocab_size(self):
        """
        Get vocabulary size.

        Returns:
            8192 (TCT vocabulary size)
        """
        return self._vocab_size

    def get_bos_token_id(self):
        """
        Get beginning-of-sequence token ID.

        Note: TCT doesn't use special BOS/EOS tokens like BPE.
        For workflow generation, we don't need explicit BOS markers.

        Returns:
            None (no BOS token)
        """
        return None

    def get_eos_token_id(self):
        """
        Get end-of-sequence token ID.

        Returns:
            None (no EOS token)
        """
        return None

    def get_special_tokens(self):
        """
        Get list of special tokens.

        TCT doesn't use special tokens for workflow generation.

        Returns:
            Empty list
        """
        return []

    def id_to_token(self, token_id):
        """
        Convert token ID to string representation.

        Note: TCT tokens are context-dependent (schema-aware),
        so there's no simple mapping from ID to string.

        Args:
            token_id: Token ID

        Returns:
            String representation
        """
        return f"<TCT:{token_id}>"

    def __call__(self, text, prepend=None, append=None):
        """
        Make tokenizer callable.

        Equivalent to self.encode(text, prepend, append)
        """
        return self.encode(text, prepend, append)

    def save(self, tokenizer_dir):
        """
        Save tokenizer (no-op for TCT).

        TCT tokenizer is embedded in the wheel, no need to save.
        """
        pass

    @classmethod
    def from_directory(cls, tokenizer_dir):
        """
        Load tokenizer from directory (just creates new instance).

        Args:
            tokenizer_dir: Ignored (TCT tokenizer is in wheel)

        Returns:
            New TCTTokenizer instance
        """
        return cls()


# =============================================================================
# Compatibility Helpers
# =============================================================================

def get_tokenizer():
    """
    Get TCT tokenizer instance.

    Returns:
        TCTTokenizer instance
    """
    return TCTTokenizer()


# =============================================================================
# Usage Example
# =============================================================================

if __name__ == "__main__":
    import json

    # Create tokenizer
    tokenizer = TCTTokenizer()

    # Example workflow
    workflow = {
        "name": "CI",
        "on": "push",
        "jobs": {
            "build": {
                "runs-on": "ubuntu-latest",
                "steps": [{"uses": "actions/checkout@v4"}],
            }
        },
    }

    # Encode
    json_str = json.dumps(workflow)
    tokens = tokenizer.encode(json_str)

    print(f"Vocabulary size: {tokenizer.get_vocab_size()}")
    print(f"Encoded tokens: {len(tokens)}")
    print(f"Token IDs: {tokens[:10]}..." if len(tokens) > 10 else f"Token IDs: {tokens}")

    # Decode
    decoded = tokenizer.decode(tokens)
    decoded_workflow = json.loads(decoded)

    # Verify round-trip
    assert decoded_workflow == workflow, "Round-trip mismatch!"
    print("✅ Round-trip successful")
