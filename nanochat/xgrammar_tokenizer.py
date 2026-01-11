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

from nanochat.json_position_classifier import JsonPositionClassifier, classify_token_bytes

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


# Schema directory (relative to this file's location)
def _get_schema_dir() -> Path:
    """Get the schemas directory relative to this file."""
    # Relative to this file: nanochat-tct/nanochat/ -> nanochat-tct/schemas/
    return Path(__file__).parent.parent / "schemas"


def load_schema(schema_name: str) -> dict:
    """Load a predefined JSON schema.

    Args:
        schema_name: One of "kubernetes", "eslintrc", "tsconfig"

    Returns:
        Loaded JSON schema as dict
    """
    schema_dir = _get_schema_dir()
    schema_path = schema_dir / f"{schema_name}.json"

    if not schema_path.exists():
        available = [p.stem for p in schema_dir.glob("*.json")]
        raise ValueError(f"Schema not found: {schema_path}. Available: {available}")

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
    # Syntax vs content analysis (optional)
    syntax_content: Optional["SyntaxContentResult"] = None


# JSON syntax characters (braces, brackets, colon, comma, quotes, whitespace)
SYNTAX_CHARS = set('{}[],:"\t\n\r ')


@dataclass
class SyntaxContentResult:
    """Results from syntax vs content loss analysis."""
    # Per-category token counts (character-based classification)
    syntax_tokens: int
    content_tokens: int
    mixed_tokens: int

    # Per-category loss (nats) - character-based
    syntax_loss: float
    content_loss: float
    mixed_loss: float

    # Per-category bits per token
    syntax_bpt: float
    content_bpt: float
    mixed_bpt: float

    # Loss percentages
    syntax_loss_pct: float
    content_loss_pct: float
    mixed_loss_pct: float

    # Content-only BPB for comparison with TCT
    content_bytes: int  # All content bytes in decoded text
    pure_content_token_bytes: int  # Bytes from pure content tokens only
    content_only_bpb: float  # Rigorous: content_loss / pure_content_token_bytes

    # Position-based semantic classification (syntax vs key vs value)
    # Syntax tokens: purely structural ({, }, [, ], :, ,, ", whitespace)
    # Key tokens: field names (schema-determined)
    # Value tokens: actual content (semantic - what TCT predicts)
    key_tokens: int = 0           # Tokens containing key content
    value_tokens: int = 0         # Tokens containing value content
    key_loss: float = 0.0         # Raw loss on key predictions
    value_loss: float = 0.0       # Raw loss on value predictions
    key_loss_pct: float = 0.0     # Percentage of raw loss on keys
    value_loss_pct: float = 0.0   # Percentage of raw loss on values
    value_bytes: int = 0          # Bytes from value tokens only
    # Constrained (XGrammar) losses by category
    constrained_key_loss: float = 0.0    # Constrained loss on key predictions
    constrained_value_loss: float = 0.0  # Constrained loss on value predictions
    # Loss per token metrics (for interpretability)
    syntax_loss_per_token: float = 0.0  # raw syntax_loss / syntax_tokens
    key_loss_per_token: float = 0.0     # raw key_loss / key_tokens
    value_loss_per_token: float = 0.0   # raw value_loss / value_tokens
    # Constrained loss per token metrics (for fair XGrammar comparison)
    constrained_syntax_loss_per_token: float = 0.0
    constrained_key_loss_per_token: float = 0.0
    constrained_value_loss_per_token: float = 0.0  # For fair comparison with TCT


def classify_token(utf8_decoder: UTF8BPEDecoder, token_id: int) -> str:
    """Classify a token as 'syntax', 'content', or 'mixed'.

    Args:
        utf8_decoder: UTF8BPEDecoder instance
        token_id: Token ID to classify

    Returns:
        'syntax': All bytes are JSON syntax/whitespace
        'content': No bytes are JSON syntax
        'mixed': Contains both syntax and content bytes
    """
    token_bytes = utf8_decoder._token_to_bytes(token_id)
    if not token_bytes:
        return 'syntax'  # EOS/empty tokens count as syntax

    try:
        text = token_bytes.decode('utf-8')
    except UnicodeDecodeError:
        return 'content'  # Non-UTF8 bytes are content

    has_syntax = any(c in SYNTAX_CHARS for c in text)
    has_content = any(c not in SYNTAX_CHARS for c in text)

    if has_syntax and has_content:
        return 'mixed'
    elif has_syntax:
        return 'syntax'
    else:
        return 'content'


def compute_constrained_bpb(
    model,
    tokenizer_info: "xgrammar.TokenizerInfo",
    compiled_grammar: "xgrammar.CompiledGrammar",
    utf8_decoder: UTF8BPEDecoder,
    validation_tokens: List[List[int]],
    device: str = "cuda",
    max_seq_len: Optional[int] = None,
    show_progress: bool = True,
    normalize_bytes: bool = False,
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
        normalize_bytes: If True, count bytes from minified JSON (for fair comparison)

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

    # Syntax vs content tracking (character-based)
    syntax_loss = 0.0
    content_loss = 0.0
    mixed_loss = 0.0
    syntax_tokens = 0
    content_tokens = 0
    mixed_tokens = 0
    total_content_bytes = 0  # All content bytes in decoded text
    pure_content_token_bytes = 0  # Bytes from pure content tokens only (for rigorous content-only BPB)

    # Position-based semantic tracking (key vs value)
    # Raw losses
    key_loss = 0.0
    value_loss = 0.0
    # Constrained losses (with XGrammar masking)
    constrained_key_loss = 0.0
    constrained_value_loss = 0.0
    # Token counts
    key_tokens = 0
    value_tokens = 0
    total_value_bytes = 0  # Bytes from value tokens only

    iterator = tqdm(validation_tokens, desc="Computing UTF8 loss") if show_progress else validation_tokens

    skipped_too_long = 0
    skipped_invalid_json = 0

    for tokens in iterator:
        if len(tokens) < 2:
            continue  # Need at least 2 tokens for next-token prediction

        # Skip sequences that exceed max_seq_len (don't truncate - it corrupts JSON)
        if max_seq_len is not None and len(tokens) > max_seq_len:
            skipped_too_long += 1
            continue

        # Check if first token is BOS (PAD token, which equals EOS in our vocab)
        # BOS is prepended during training but is NOT part of JSON content
        bos_token_id = utf8_decoder.eos_token_id()
        decode_tokens = tokens[1:] if tokens[0] == bos_token_id else tokens

        # Decode to text for byte count (without BOS token)
        text = utf8_decoder.decode(decode_tokens)

        # Normalize to minified JSON for fair comparison with TCT
        if normalize_bytes:
            try:
                import json
                import re
                parsed = json.loads(text)
                text = json.dumps(parsed, separators=(',', ':'), sort_keys=True)
                # Normalize ISO 8601 UTC timestamps:
                # 1. Strip microseconds: .NNNNNN -> empty (TCT doesn't preserve them)
                text = re.sub(r'(\d{2}:\d{2}:\d{2})\.\d+', r'\1', text)
                # 2. Normalize timezone: +00:00 -> Z
                text = re.sub(r'(\d{2}:\d{2}:\d{2})\+00:00', r'\1Z', text)
                text = re.sub(r'(\d{2}:\d{2}:\d{2})-00:00', r'\1Z', text)
            except json.JSONDecodeError:
                skipped_invalid_json += 1
                continue  # Skip sequences that aren't valid JSON

        n_bytes = len(text.encode('utf-8'))
        if n_bytes == 0:
            continue

        # Count content bytes (non-syntax characters) - computed before grammar check
        # but only added to totals after grammar check succeeds
        content_bytes_in_seq = sum(1 for c in text if c not in SYNTAX_CHARS)

        # Initialize grammar matcher for this sequence
        matcher = xgrammar.GrammarMatcher(compiled_grammar)
        bitmask = xgrammar.allocate_token_bitmask(1, tokenizer_info.vocab_size)

        # Get model logits for entire sequence at once
        input_ids = torch.tensor([tokens[:-1]], device=device)  # All but last
        with torch.no_grad():
            logits = model(input_ids)  # [1, T-1, V]
            logits = logits[0]  # [T-1, V]

        # Accept the first token to initialize grammar state
        # (the model predicts tokens[t+1] given tokens[0:t+1])
        # IMPORTANT: Skip accepting BOS - it's not part of JSON grammar
        # Grammar should start fresh, ready for first JSON token
        if tokens[0] != bos_token_id:
            try:
                matcher.accept_token(tokens[0])
            except Exception:
                continue  # Skip sequences that grammar can't parse

        # Only count bytes/sequences AFTER grammar check succeeds
        # This fixes the bug where bytes were counted for sequences that fail grammar parsing
        total_bytes += n_bytes
        num_sequences += 1
        total_content_bytes += content_bytes_in_seq

        # NEW: Create position classifier for semantic analysis (key vs value)
        position_classifier = JsonPositionClassifier(text)
        value_bytes_in_seq = position_classifier.get_stats()['value_bytes']

        # Precompute byte offsets for each token in decode_tokens
        # decode_tokens[i] starts at byte_offsets[i] in the decoded text
        # But we need to map from tokens[] to decode_tokens[]
        # tokens = [BOS, tok0, tok1, ...] and decode_tokens = [tok0, tok1, ...]
        # So tokens[t+1] corresponds to decode_tokens[t] when BOS is present
        has_bos = tokens[0] == bos_token_id
        byte_offsets = [0]
        for tok in decode_tokens:
            tok_bytes = utf8_decoder._token_to_bytes(tok)
            byte_offsets.append(byte_offsets[-1] + (len(tok_bytes) if tok_bytes else 0))

        # Process each position
        for t in range(len(tokens) - 1):
            target_token = tokens[t + 1]
            pos_logits = logits[t]  # [V]

            # Raw loss (unconstrained)
            raw_log_probs = F.log_softmax(pos_logits, dim=-1)
            raw_loss = -raw_log_probs[target_token].item()
            total_raw_loss += raw_loss

            # Classify token and track per-category loss (character-based)
            category = classify_token(utf8_decoder, target_token)
            if category == 'syntax':
                syntax_loss += raw_loss
                syntax_tokens += 1
            elif category == 'content':
                content_loss += raw_loss
                content_tokens += 1
                # Track bytes from pure content tokens for rigorous content-only BPB
                token_bytes = utf8_decoder._token_to_bytes(target_token)
                if token_bytes:
                    pure_content_token_bytes += len(token_bytes)
            else:  # mixed
                mixed_loss += raw_loss
                mixed_tokens += 1

            # Position-based semantic classification (key vs value)
            # target_token = tokens[t+1] corresponds to decode_tokens[t] when BOS present
            # We track both raw_loss and constrained_loss by category
            token_semantic_category = None  # Will be set if classification succeeds
            decode_idx = t if has_bos else t + 1
            if decode_idx < len(byte_offsets) - 1:
                byte_offset = byte_offsets[decode_idx]
                token_bytes = utf8_decoder._token_to_bytes(target_token)
                if token_bytes:
                    classification = classify_token_bytes(token_bytes, byte_offset, position_classifier)
                    token_semantic_category = classification['primary']
                    if token_semantic_category == 'key':
                        key_loss += raw_loss
                        key_tokens += 1
                    elif token_semantic_category == 'value':
                        value_loss += raw_loss
                        value_tokens += 1
                        total_value_bytes += len(token_bytes)

            # Get valid token mask from grammar (after accepting tokens[0:t+1])
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

            # Attribute constrained loss to semantic category (same as raw loss)
            if token_semantic_category == 'key':
                constrained_key_loss += constrained_loss
            elif token_semantic_category == 'value':
                constrained_value_loss += constrained_loss

            # Advance grammar state to include current target
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

    log2 = math.log(2)
    raw_bpb = total_raw_loss / (log2 * total_bytes)
    constrained_bpb = total_constrained_loss / (log2 * total_bytes)

    # Compute syntax/content analysis
    total_categorized_loss = syntax_loss + content_loss + mixed_loss
    if total_categorized_loss > 0:
        syntax_loss_pct = 100 * syntax_loss / total_categorized_loss
        content_loss_pct = 100 * content_loss / total_categorized_loss
        mixed_loss_pct = 100 * mixed_loss / total_categorized_loss
    else:
        syntax_loss_pct = content_loss_pct = mixed_loss_pct = 0.0

    # Bits per token for each category
    syntax_bpt = (syntax_loss / (log2 * syntax_tokens)) if syntax_tokens > 0 else 0.0
    content_bpt = (content_loss / (log2 * content_tokens)) if content_tokens > 0 else 0.0
    mixed_bpt = (mixed_loss / (log2 * mixed_tokens)) if mixed_tokens > 0 else 0.0

    # Content-only BPB (for fair comparison with TCT which has no syntax overhead)
    # Use pure_content_token_bytes for rigorous comparison: loss from pure content tokens / bytes of those tokens
    # This ensures numerator and denominator are consistent (both from same tokens)
    content_only_bpb = (content_loss / (log2 * pure_content_token_bytes)) if pure_content_token_bytes > 0 else 0.0

    # Compute key/value loss percentages and per-token metrics
    # Key loss percentage (of total loss)
    if total_raw_loss > 0:
        key_loss_pct = 100 * key_loss / total_raw_loss
        value_loss_pct = 100 * value_loss / total_raw_loss
    else:
        key_loss_pct = value_loss_pct = 0.0

    # Loss per token metrics (for interpretability)
    # Position-based syntax = tokens that are neither key nor value
    pos_syntax_tokens = total_tokens - key_tokens - value_tokens
    pos_syntax_loss = total_raw_loss - key_loss - value_loss
    syntax_loss_per_token = (pos_syntax_loss / pos_syntax_tokens) if pos_syntax_tokens > 0 else 0.0
    key_loss_per_token = (key_loss / key_tokens) if key_tokens > 0 else 0.0
    value_loss_per_token = (value_loss / value_tokens) if value_tokens > 0 else 0.0

    # Constrained loss per token metrics (for fair XGrammar vs TCT comparison)
    constrained_syntax_loss = total_constrained_loss - constrained_key_loss - constrained_value_loss
    constrained_syntax_loss_per_token = (constrained_syntax_loss / pos_syntax_tokens) if pos_syntax_tokens > 0 else 0.0
    constrained_key_loss_per_token = (constrained_key_loss / key_tokens) if key_tokens > 0 else 0.0
    constrained_value_loss_per_token = (constrained_value_loss / value_tokens) if value_tokens > 0 else 0.0

    syntax_content = SyntaxContentResult(
        syntax_tokens=syntax_tokens,
        content_tokens=content_tokens,
        mixed_tokens=mixed_tokens,
        syntax_loss=syntax_loss,
        content_loss=content_loss,
        mixed_loss=mixed_loss,
        syntax_bpt=syntax_bpt,
        content_bpt=content_bpt,
        mixed_bpt=mixed_bpt,
        syntax_loss_pct=syntax_loss_pct,
        content_loss_pct=content_loss_pct,
        mixed_loss_pct=mixed_loss_pct,
        content_bytes=total_content_bytes,
        pure_content_token_bytes=pure_content_token_bytes,
        content_only_bpb=content_only_bpb,
        # Position-based key/value metrics (raw)
        key_tokens=key_tokens,
        value_tokens=value_tokens,
        key_loss=key_loss,
        value_loss=value_loss,
        key_loss_pct=key_loss_pct,
        value_loss_pct=value_loss_pct,
        value_bytes=total_value_bytes,
        # Constrained losses by category
        constrained_key_loss=constrained_key_loss,
        constrained_value_loss=constrained_value_loss,
        # Raw loss per token
        syntax_loss_per_token=syntax_loss_per_token,
        key_loss_per_token=key_loss_per_token,
        value_loss_per_token=value_loss_per_token,
        # Constrained loss per token
        constrained_syntax_loss_per_token=constrained_syntax_loss_per_token,
        constrained_key_loss_per_token=constrained_key_loss_per_token,
        constrained_value_loss_per_token=constrained_value_loss_per_token,
    )

    # Report skipped sequences
    if skipped_too_long > 0 or skipped_invalid_json > 0:
        print(f"  Skipped: {skipped_too_long} too long, {skipped_invalid_json} invalid JSON")

    return ConstrainedBPBResult(
        raw_bpb=raw_bpb,
        constrained_bpb=constrained_bpb,
        total_tokens=total_tokens,
        total_bytes=total_bytes,
        num_sequences=num_sequences,
        raw_loss=total_raw_loss,
        constrained_loss=total_constrained_loss,
        syntax_content=syntax_content,
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
        import tct_tsconfig as tct
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
    normalize_bytes: bool = False,
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
        normalize_bytes: If True, count bytes from minified JSON with sorted keys
                        (for fair comparison - TCT already produces minified JSON
                        but may have different key ordering)

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

    iterator = tqdm(validation_tokens, desc="Computing TCT loss") if show_progress else validation_tokens

    skipped_too_long = 0
    skipped_decode_failed = 0

    for tokens in iterator:
        if len(tokens) < 2:
            continue  # Need at least 2 tokens for next-token prediction

        # Skip sequences that exceed max_seq_len (don't truncate - it corrupts decode)
        if max_seq_len is not None and len(tokens) > max_seq_len:
            skipped_too_long += 1
            continue

        # Check if first token is BOS (PAD token = vocab_size - 1)
        # BOS is prepended during training but is NOT part of JSON content
        pad_token_id = tct_module.vocab_size() - 1
        decode_tokens = tokens[1:] if tokens[0] == pad_token_id else tokens

        # Decode to text for byte count (without BOS token)
        try:
            json_out, consumed, surplus = tct_module.decode(decode_tokens)

            # Normalize to minified JSON with sorted keys for fair comparison
            if normalize_bytes:
                import json
                import re
                parsed = json.loads(json_out)
                json_out = json.dumps(parsed, separators=(',', ':'), sort_keys=True)
                # Normalize ISO 8601 UTC timestamps:
                # 1. Strip microseconds: .NNNNNN -> empty (TCT doesn't preserve them)
                json_out = re.sub(r'(\d{2}:\d{2}:\d{2})\.\d+', r'\1', json_out)
                # 2. Normalize timezone: +00:00 -> Z
                json_out = re.sub(r'(\d{2}:\d{2}:\d{2})\+00:00', r'\1Z', json_out)
                json_out = re.sub(r'(\d{2}:\d{2}:\d{2})-00:00', r'\1Z', json_out)

            n_bytes = len(json_out.encode('utf-8'))
        except Exception:
            skipped_decode_failed += 1
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

    # Report skipped sequences
    if skipped_too_long > 0 or skipped_decode_failed > 0:
        print(f"  Skipped: {skipped_too_long} too long, {skipped_decode_failed} decode failed")

    return TCTBPBResult(
        bpb=bpb,
        total_tokens=total_tokens,
        total_bytes=total_bytes,
        num_sequences=num_sequences,
        total_loss=total_loss,
    )
