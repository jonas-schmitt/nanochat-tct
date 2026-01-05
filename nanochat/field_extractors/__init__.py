"""Field extractors for schema-specific JSON config analysis.

Usage:
    from nanochat.field_extractors import get_extractor

    extractor = get_extractor("tsconfig")
    result = extractor.extract_from_samples(json_strings)
    print(result.field_distributions["compilerOptions.target"].counts)
"""

from typing import TYPE_CHECKING

from .base import (
    ExtractionResult,
    FieldDistribution,
    FieldExtractor,
    FieldValue,
)

if TYPE_CHECKING:
    pass

# Registry of available extractors
_EXTRACTORS = {
    "tsconfig": "nanochat.field_extractors.tsconfig",
    "eslintrc": "nanochat.field_extractors.eslintrc",
    "kubernetes": "nanochat.field_extractors.kubernetes",
}


def get_extractor(schema_name: str) -> FieldExtractor:
    """Get a field extractor for the given schema.

    Args:
        schema_name: Schema name ("tsconfig", "eslintrc", "kubernetes")

    Returns:
        FieldExtractor instance

    Raises:
        ValueError: If schema not supported
    """
    if schema_name not in _EXTRACTORS:
        available = list(_EXTRACTORS.keys())
        raise ValueError(f"Unknown schema: {schema_name}. Available: {available}")

    # Lazy import to avoid circular dependencies
    module_name = _EXTRACTORS[schema_name]
    import importlib
    module = importlib.import_module(module_name)
    return module.get_extractor()


def list_schemas() -> list:
    """List available schema names."""
    return list(_EXTRACTORS.keys())


__all__ = [
    "get_extractor",
    "list_schemas",
    "FieldExtractor",
    "FieldValue",
    "FieldDistribution",
    "ExtractionResult",
]
