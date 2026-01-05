"""
UTF8-BPE tokenizer utilities for nanochat.

Provides:
1. UTF8BPEDecoder: Standalone decoder that works WITHOUT xgrammar
2. build_xgrammar_tokenizer_info: XGrammar integration (requires xgrammar)
3. compile_json_schema_grammar: Schema-constrained generation (requires xgrammar)
4. compute_constrained_bpb: Compute BPB with XGrammar constraints (for evaluation)

Usage (decoding only - no xgrammar required):
    from nanochat.xgrammar_tokenizer import UTF8BPEDecoder

    decoder = UTF8BPEDecoder("bpe-merges/kubernetes-utf8-bpe.json")
    text = decoder.decode([123, 456, 789])  # Decode token IDs to string
    vocab_size = decoder.vocab_size()

Usage (constrained generation - requires xgrammar):
    from nanochat.xgrammar_tokenizer import (
        build_xgrammar_tokenizer_info,
        compile_json_schema_grammar,
    )

    tokenizer_info = build_xgrammar_tokenizer_info(
        merge_table_path="bpe-merges/kubernetes-utf8-bpe.json"
    )
    schema = {"type": "object", "properties": {...}}
    compiled = compile_json_schema_grammar(tokenizer_info, schema)

Usage (constrained BPB evaluation):
    from nanochat.xgrammar_tokenizer import compute_constrained_bpb

    results = compute_constrained_bpb(
        model=model,
        tokenizer_info=tokenizer_info,
        compiled_grammar=compiled_grammar,
        utf8_decoder=decoder,
        validation_tokens=val_tokens,
    )
    print(f"Raw BPB: {results['raw_bpb']:.4f}")
    print(f"Constrained BPB: {results['constrained_bpb']:.4f}")
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

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
        decoder = UTF8BPEDecoder("bpe-merges/kubernetes-utf8-bpe.json")
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


# =============================================================================
# Constrained BPB Evaluation
# =============================================================================

@dataclass
class ConstrainedBPBResult:
    """Results from constrained BPB evaluation."""
    raw_bpb: float              # BPB without grammar constraints
    constrained_bpb: float      # BPB with grammar constraints (renormalized)
    total_tokens: int           # Total tokens evaluated
    total_bytes: int            # Total bytes in decoded text
    num_sequences: int          # Number of sequences evaluated
    raw_loss: float             # Total raw cross-entropy loss (nats)
    constrained_loss: float     # Total constrained cross-entropy loss (nats)


def compute_constrained_bpb(
    model,
    tokenizer_info: "xgrammar.TokenizerInfo",
    compiled_grammar: "xgrammar.CompiledGrammar",
    utf8_decoder: UTF8BPEDecoder,
    validation_tokens: List[List[int]],
    device: str = "cuda",
    max_seq_len: Optional[int] = None,
    show_progress: bool = True,
) -> ConstrainedBPBResult:
    """Compute bits-per-byte with XGrammar constraints applied.

    For each token sequence:
    1. Get valid token mask from grammar state at each position
    2. Compute raw loss (unconstrained log prob)
    3. Compute constrained loss (log prob renormalized over valid tokens only)
    4. Track raw bytes for BPB normalization

    This allows fair comparison: BPE with XGrammar constraints vs TCT.
    The constrained BPB shows the effective compression when the model
    only needs to predict among grammar-valid tokens.

    Args:
        model: The GPT model (must have forward() returning logits)
        tokenizer_info: XGrammar TokenizerInfo for this vocabulary
        compiled_grammar: XGrammar CompiledGrammar for the schema
        utf8_decoder: UTF8BPEDecoder for decoding tokens to bytes
        validation_tokens: List of token sequences to evaluate
        device: Device to run on (default: "cuda")
        max_seq_len: Maximum sequence length to process (default: None = no limit)
        show_progress: Whether to show progress bar (default: True)

    Returns:
        ConstrainedBPBResult with raw and constrained BPB metrics
    """
    if xgrammar is None:
        raise ImportError("xgrammar is required for constrained BPB evaluation")

    try:
        import torch
        import torch.nn.functional as F
        from tqdm import tqdm
    except ImportError as e:
        raise ImportError(f"Required package not found: {e}")

    model.eval()

    total_raw_loss = 0.0
    total_constrained_loss = 0.0
    total_bytes = 0
    total_tokens = 0
    num_sequences = 0

    iterator = tqdm(validation_tokens, desc="Computing constrained BPB") if show_progress else validation_tokens

    for tokens in iterator:
        if len(tokens) < 2:
            continue  # Need at least 2 tokens for next-token prediction

        # Optionally truncate
        if max_seq_len is not None and len(tokens) > max_seq_len:
            tokens = tokens[:max_seq_len]

        # Decode to text for byte count
        text = utf8_decoder.decode(tokens)
        n_bytes = len(text.encode('utf-8'))
        if n_bytes == 0:
            continue

        total_bytes += n_bytes
        num_sequences += 1

        # Initialize grammar matcher for this sequence
        matcher = xgrammar.GrammarMatcher(compiled_grammar)
        bitmask = xgrammar.allocate_token_bitmask(1, tokenizer_info.vocab_size)

        # Get model logits for entire sequence at once
        input_ids = torch.tensor([tokens[:-1]], device=device)  # All but last
        with torch.no_grad():
            logits = model(input_ids)  # [1, T-1, V]
            logits = logits[0]  # [T-1, V]

        # Process each position
        for t in range(len(tokens) - 1):
            target_token = tokens[t + 1]
            pos_logits = logits[t]  # [V]

            # Raw loss (unconstrained)
            raw_log_probs = F.log_softmax(pos_logits, dim=-1)
            raw_loss = -raw_log_probs[target_token].item()
            total_raw_loss += raw_loss

            # Get valid token mask from grammar
            xgrammar.reset_token_bitmask(bitmask)
            matcher.fill_next_token_bitmask(bitmask)

            # Apply grammar mask for constrained loss
            constrained_logits = pos_logits.clone()
            bitmask_tensor = bitmask.to(device)
            xgrammar.apply_token_bitmask_inplace(constrained_logits.unsqueeze(0), bitmask_tensor)
            constrained_logits = constrained_logits.squeeze(0)

            # Constrained loss (renormalized over valid tokens)
            constrained_log_probs = F.log_softmax(constrained_logits, dim=-1)
            constrained_loss = -constrained_log_probs[target_token].item()

            # Handle case where target is not valid (shouldn't happen with correct grammar)
            if math.isinf(constrained_loss):
                # Target token not in valid set - use raw loss as fallback
                constrained_loss = raw_loss

            total_constrained_loss += constrained_loss
            total_tokens += 1

            # Advance grammar state
            try:
                matcher.accept_token(target_token)
            except Exception:
                # If grammar rejects token, reset matcher for next sequence
                break

    # Compute BPB
    if total_bytes == 0:
        return ConstrainedBPBResult(
            raw_bpb=float('inf'),
            constrained_bpb=float('inf'),
            total_tokens=0,
            total_bytes=0,
            num_sequences=0,
            raw_loss=0.0,
            constrained_loss=0.0,
        )

    raw_bpb = total_raw_loss / (math.log(2) * total_bytes)
    constrained_bpb = total_constrained_loss / (math.log(2) * total_bytes)

    return ConstrainedBPBResult(
        raw_bpb=raw_bpb,
        constrained_bpb=constrained_bpb,
        total_tokens=total_tokens,
        total_bytes=total_bytes,
        num_sequences=num_sequences,
        raw_loss=total_raw_loss,
        constrained_loss=total_constrained_loss,
    )


# =============================================================================
# TCT BPB Evaluation
# =============================================================================

@dataclass
class TCTBPBResult:
    """Results from TCT BPB evaluation."""
    bpb: float                  # Bits per byte
    total_tokens: int           # Total tokens evaluated
    total_bytes: int            # Total bytes in decoded text
    num_sequences: int          # Number of sequences evaluated
    total_loss: float           # Total cross-entropy loss (nats)


def get_tct_module(schema: str):
    """Get TCT tokenizer module for schema.

    Args:
        schema: One of "tsconfig", "eslintrc", "kubernetes"

    Returns:
        TCT module with encode/decode/decode_prefix/vocab_size functions
    """
    if schema == "tsconfig":
        import tct_tsconfig_base as tct
    elif schema == "eslintrc":
        import tct_eslintrc_bpe_500 as tct
    elif schema == "kubernetes":
        import tct_kubernetes_bpe_1k as tct
    else:
        raise ValueError(f"Unknown schema: {schema}. Available: tsconfig, eslintrc, kubernetes")
    return tct


def compute_tct_bpb(
    model,
    tct_module,
    validation_tokens: List[List[int]],
    device: str = "cuda",
    max_seq_len: Optional[int] = None,
    show_progress: bool = True,
) -> TCTBPBResult:
    """Compute bits-per-byte for TCT model.

    TCT is inherently schema-constrained (100% valid by construction),
    so we only compute raw BPB (no grammar masking needed).

    Args:
        model: The GPT model (must have forward() returning logits)
        tct_module: TCT tokenizer module (has decode function)
        validation_tokens: List of token sequences to evaluate
        device: Device to run on (default: "cuda")
        max_seq_len: Maximum sequence length to process (default: None = no limit)
        show_progress: Whether to show progress bar (default: True)

    Returns:
        TCTBPBResult with BPB metrics
    """
    try:
        import torch
        import torch.nn.functional as F
        from tqdm import tqdm
    except ImportError as e:
        raise ImportError(f"Required package not found: {e}")

    model.eval()

    total_loss = 0.0
    total_bytes = 0
    total_tokens = 0
    num_sequences = 0

    iterator = tqdm(validation_tokens, desc="Computing TCT BPB") if show_progress else validation_tokens

    for tokens in iterator:
        if len(tokens) < 2:
            continue  # Need at least 2 tokens for next-token prediction

        # Optionally truncate
        if max_seq_len is not None and len(tokens) > max_seq_len:
            tokens = tokens[:max_seq_len]

        # Decode to text for byte count
        try:
            json_out, consumed, surplus = tct_module.decode(tokens)
            n_bytes = len(json_out.encode('utf-8'))
        except Exception:
            continue

        if n_bytes == 0:
            continue

        total_bytes += n_bytes
        num_sequences += 1

        # Get model logits for entire sequence at once
        input_ids = torch.tensor([tokens[:-1]], device=device)  # All but last
        targets = torch.tensor([tokens[1:]], device=device)     # All but first

        with torch.no_grad():
            logits = model(input_ids)  # [1, T-1, V]

            # Compute cross-entropy loss
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                reduction='sum'
            )
            total_loss += loss.item()
            total_tokens += len(tokens) - 1

    # Compute BPB
    if total_bytes == 0:
        return TCTBPBResult(
            bpb=float('inf'),
            total_tokens=0,
            total_bytes=0,
            num_sequences=0,
            total_loss=0.0,
        )

    bpb = total_loss / (math.log(2) * total_bytes)

    return TCTBPBResult(
        bpb=bpb,
        total_tokens=total_tokens,
        total_bytes=total_bytes,
        num_sequences=num_sequences,
        total_loss=total_loss,
    )
