"""
UTF8-BPE tokenizer utilities for nanochat.

Provides:
1. UTF8BPEDecoder: Standalone decoder that works WITHOUT xgrammar
2. build_xgrammar_tokenizer_info: XGrammar integration (requires xgrammar)
3. compile_json_schema_grammar: Schema-constrained generation (requires xgrammar)

Usage (decoding only - no xgrammar required):
    from nanochat.xgrammar_tokenizer import UTF8BPEDecoder

    decoder = UTF8BPEDecoder("bpe-merges/kubernetes-utf8-bpe-matched.json")
    text = decoder.decode([123, 456, 789])  # Decode token IDs to string
    vocab_size = decoder.vocab_size()

Usage (constrained generation - requires xgrammar):
    from nanochat.xgrammar_tokenizer import (
        build_xgrammar_tokenizer_info,
        compile_json_schema_grammar,
    )

    tokenizer_info = build_xgrammar_tokenizer_info(
        merge_table_path="bpe-merges/kubernetes-utf8-bpe-matched.json"
    )
    schema = {"type": "object", "properties": {...}}
    compiled = compile_json_schema_grammar(tokenizer_info, schema)
"""

import json
from pathlib import Path
from typing import List, Optional, Union

try:
    import xgrammar
except ImportError:
    xgrammar = None


def _token_to_bytes(token_id: int, decode_table: dict, base_vocab_size: int = 256) -> bytes:
    """Recursively decode a token ID to its byte sequence.

    Args:
        token_id: The token ID to decode
        decode_table: Mapping from token_id -> (left_id, right_id) for merged tokens
        base_vocab_size: Size of base vocabulary (single bytes), default 256

    Returns:
        The byte sequence this token represents
    """
    if token_id < base_vocab_size:
        return bytes([token_id])
    elif token_id in decode_table:
        left, right = decode_table[token_id]
        return _token_to_bytes(left, decode_table, base_vocab_size) + \
               _token_to_bytes(right, decode_table, base_vocab_size)
    else:
        # Unknown token (e.g., special tokens added later)
        return b""


def build_vocabulary_from_merges(merge_table_path: Union[str, Path]) -> tuple:
    """Build vocabulary from BPE merge table.

    Args:
        merge_table_path: Path to merge table JSON file

    Returns:
        Tuple of (encoded_vocab: List[bytes], eos_token_id: int, vocab_size: int)
    """
    with open(merge_table_path) as f:
        merge_data = json.load(f)

    base_vocab_size = merge_data.get("base_vocab_size", 256)
    merges = merge_data.get("merges", [])

    # Build decode table: {new_token_id: (left_id, right_id)}
    decode_table = {}
    for merge in merges:
        new_token = merge["new_token"]
        left, right = merge["pair"]
        decode_table[new_token] = (left, right)

    # Total vocabulary size (base + merges)
    total_vocab = base_vocab_size + len(merges)

    # Build encoded vocabulary (list of byte sequences)
    encoded_vocab = []
    for i in range(total_vocab):
        token_bytes = _token_to_bytes(i, decode_table, base_vocab_size)
        encoded_vocab.append(token_bytes)

    # Add EOS token
    eos_token_id = total_vocab
    encoded_vocab.append(b"</s>")

    return encoded_vocab, eos_token_id, len(encoded_vocab)


class UTF8BPEDecoder:
    """UTF8-BPE decoder that works without xgrammar.

    Provides decode functionality for UTF8-BPE models using the merge table.
    Compatible with both constrained (XGrammar) and unconstrained generation.

    Usage:
        decoder = UTF8BPEDecoder("bpe-merges/kubernetes-utf8-bpe-matched.json")
        text = decoder.decode([123, 456, 789])  # Decode token IDs to string
        vocab_size = decoder.vocab_size()
    """

    def __init__(self, merge_table_path: Union[str, Path]):
        """Initialize decoder from BPE merge table.

        Args:
            merge_table_path: Path to merge table JSON file
        """
        self.merge_table_path = Path(merge_table_path)

        with open(merge_table_path) as f:
            merge_data = json.load(f)

        self.base_vocab_size = merge_data.get("base_vocab_size", 256)
        merges = merge_data.get("merges", [])

        # Build decode table: {new_token_id: (left_id, right_id)}
        self._decode_table = {}
        for merge in merges:
            new_token = merge["new_token"]
            left, right = merge["pair"]
            self._decode_table[new_token] = (left, right)

        # Total vocabulary size (base + merges + EOS)
        self._vocab_size = self.base_vocab_size + len(merges) + 1
        self._eos_token_id = self.base_vocab_size + len(merges)

    def vocab_size(self) -> int:
        """Return vocabulary size (for compatibility with tokenizer modules)."""
        return self._vocab_size

    def eos_token_id(self) -> int:
        """Return EOS token ID."""
        return self._eos_token_id

    def _token_to_bytes(self, token_id: int) -> bytes:
        """Recursively decode a token ID to its byte sequence."""
        if token_id < self.base_vocab_size:
            return bytes([token_id])
        elif token_id in self._decode_table:
            left, right = self._decode_table[token_id]
            return self._token_to_bytes(left) + self._token_to_bytes(right)
        elif token_id == self._eos_token_id:
            return b""  # EOS produces empty bytes
        else:
            # Unknown token
            return b""

    def decode(self, tokens: List[int]) -> str:
        """Decode token IDs to string.

        Args:
            tokens: List of token IDs

        Returns:
            Decoded string (UTF-8)
        """
        byte_sequence = b""
        for token_id in tokens:
            byte_sequence += self._token_to_bytes(token_id)

        # Decode UTF-8 with error handling
        try:
            return byte_sequence.decode("utf-8")
        except UnicodeDecodeError:
            # Try to decode as much as possible
            return byte_sequence.decode("utf-8", errors="replace")


def build_xgrammar_tokenizer_info(
    merge_table_path: Union[str, Path],
    eos_token: bytes = b"</s>",
    add_prefix_space: bool = False,
) -> "xgrammar.TokenizerInfo":
    """Build XGrammar TokenizerInfo from BPE merge table.

    Args:
        merge_table_path: Path to BPE merge table JSON file
        eos_token: EOS token bytes (default: b"</s>")
        add_prefix_space: Whether tokenizer adds prefix space (default: False)

    Returns:
        xgrammar.TokenizerInfo for use with GrammarCompiler

    Raises:
        ImportError: If xgrammar is not installed
        FileNotFoundError: If merge table file doesn't exist
    """
    if xgrammar is None:
        raise ImportError(
            "xgrammar is required for constrained generation. "
            "Install with: pip install xgrammar"
        )

    # Build vocabulary from merge table
    encoded_vocab, eos_token_id, vocab_size = build_vocabulary_from_merges(merge_table_path)

    # Create metadata JSON
    # vocab_type: 0 = RAW, 1 = BYTE_FALLBACK, 2 = BYTE_LEVEL
    metadata = json.dumps({
        "vocab_type": 0,  # RAW - tokens are raw byte sequences
        "vocab_size": vocab_size,
        "add_prefix_space": add_prefix_space,
        "stop_token_ids": [eos_token_id],
    })

    # Create TokenizerInfo
    tokenizer_info = xgrammar.TokenizerInfo.from_vocab_and_metadata(
        encoded_vocab, metadata
    )

    return tokenizer_info


def compile_json_schema_grammar(
    tokenizer_info: "xgrammar.TokenizerInfo",
    schema: Union[str, dict],
    strict_mode: bool = True,
) -> "xgrammar.CompiledGrammar":
    """Compile a JSON schema into a grammar for constrained generation.

    Args:
        tokenizer_info: XGrammar TokenizerInfo
        schema: JSON schema (dict or JSON string)
        strict_mode: Whether to use strict JSON schema validation (default: True)

    Returns:
        Compiled grammar for use with GrammarMatcher
    """
    if xgrammar is None:
        raise ImportError("xgrammar is required")

    # Create grammar from schema
    grammar = xgrammar.Grammar.from_json_schema(schema, strict_mode=strict_mode)

    # Compile with tokenizer
    compiler = xgrammar.GrammarCompiler(tokenizer_info)
    compiled = compiler.compile_grammar(grammar)

    return compiled


# Schema paths for common schemas
SCHEMA_PATHS = {
    "kubernetes": "/home/josch/git/tct/schemas/popular/kubernetes.json",
    "eslintrc": "/home/josch/git/tct/schemas/popular/eslintrc.json",
    "tsconfig": "/home/josch/git/tct/schemas/popular/tsconfig.json",
}


def load_schema(schema_name: str) -> dict:
    """Load a predefined JSON schema.

    Args:
        schema_name: One of "kubernetes", "eslintrc", "tsconfig"

    Returns:
        Loaded JSON schema as dict
    """
    if schema_name not in SCHEMA_PATHS:
        raise ValueError(f"Unknown schema: {schema_name}. Available: {list(SCHEMA_PATHS.keys())}")

    schema_path = SCHEMA_PATHS[schema_name]
    with open(schema_path) as f:
        return json.load(f)
