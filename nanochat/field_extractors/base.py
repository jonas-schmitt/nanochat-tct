"""Base classes for schema-specific field extraction.

Field extractors parse JSON config files and extract semantic field values
for distribution comparison between TCT and BPE+XGrammar models.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union


@dataclass
class FieldValue:
    """A single extracted field value with metadata."""
    field_name: str           # Dotted path (e.g., "compilerOptions.target")
    value: Any                # The actual value (bool, str, int, list, etc.)
    value_type: str           # Type hint: "bool", "enum", "string", "number", "array"
    source_file: Optional[str] = None  # Optional source identifier


@dataclass
class FieldDistribution:
    """Distribution of values for a single field across many samples."""
    field_name: str
    value_type: str
    counts: Dict[str, int] = field(default_factory=dict)  # value -> count
    total: int = 0

    def add(self, value: Any) -> None:
        """Add a value to the distribution."""
        # Normalize value to string for counting
        key = self._normalize_value(value)
        self.counts[key] = self.counts.get(key, 0) + 1
        self.total += 1

    def _normalize_value(self, value: Any) -> str:
        """Normalize value to string key for counting."""
        if value is None:
            return "<None>"
        elif isinstance(value, bool):
            return str(value).lower()  # "true" or "false"
        elif isinstance(value, (int, float)):
            return str(value)
        elif isinstance(value, str):
            return value
        elif isinstance(value, list):
            return f"[{len(value)} items]"  # Just track array presence
        else:
            return str(value)

    def get_probability(self, value: Any) -> float:
        """Get probability of a specific value."""
        if self.total == 0:
            return 0.0
        key = self._normalize_value(value)
        return self.counts.get(key, 0) / self.total

    def mode(self) -> Optional[str]:
        """Get the most common value."""
        if not self.counts:
            return None
        return max(self.counts, key=self.counts.get)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "field_name": self.field_name,
            "value_type": self.value_type,
            "counts": self.counts,
            "total": self.total,
        }


@dataclass
class ExtractionResult:
    """Result of extracting fields from multiple samples."""
    schema_name: str
    num_samples: int
    num_valid: int
    num_failed: int
    field_distributions: Dict[str, FieldDistribution]
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "schema_name": self.schema_name,
            "num_samples": self.num_samples,
            "num_valid": self.num_valid,
            "num_failed": self.num_failed,
            "field_distributions": {
                k: v.to_dict() for k, v in self.field_distributions.items()
            },
            "errors": self.errors[:100],  # Limit errors in output
        }


class FieldExtractor(ABC):
    """Abstract base class for schema-specific field extractors."""

    @property
    @abstractmethod
    def schema_name(self) -> str:
        """Return the schema name (e.g., 'tsconfig', 'eslintrc', 'kubernetes')."""
        pass

    @property
    @abstractmethod
    def field_definitions(self) -> List[Dict[str, Any]]:
        """Return list of field definitions to extract.

        Each definition is a dict with:
        - name: str - Dotted path to field (e.g., "compilerOptions.strict")
        - type: str - Value type ("bool", "enum", "string", "number")
        - enum_values: List[str] - For enum fields, the valid values
        - description: str - Optional description
        """
        pass

    def extract_field(self, data: dict, field_path: str) -> Optional[Any]:
        """Extract a field value from a JSON object using dotted path.

        Args:
            data: Parsed JSON object
            field_path: Dotted path like "compilerOptions.target"

        Returns:
            Field value or None if not present
        """
        parts = field_path.split(".")
        current = data
        for part in parts:
            if not isinstance(current, dict):
                return None
            if part not in current:
                return None
            current = current[part]
        return current

    def extract_fields(self, data: dict) -> List[FieldValue]:
        """Extract all defined fields from a JSON object.

        Args:
            data: Parsed JSON object

        Returns:
            List of FieldValue objects for fields that exist
        """
        results = []
        for field_def in self.field_definitions:
            value = self.extract_field(data, field_def["name"])
            if value is not None:
                results.append(FieldValue(
                    field_name=field_def["name"],
                    value=value,
                    value_type=field_def["type"],
                ))
        return results

    def extract_from_samples(
        self,
        samples: List[Union[str, dict]],
        sample_type: str = "json",
    ) -> ExtractionResult:
        """Extract field distributions from multiple samples.

        Args:
            samples: List of JSON strings or parsed dicts
            sample_type: "json" (strings) or "dict" (parsed objects)

        Returns:
            ExtractionResult with field distributions
        """
        import json

        # Initialize distributions for all defined fields
        distributions = {}
        for field_def in self.field_definitions:
            distributions[field_def["name"]] = FieldDistribution(
                field_name=field_def["name"],
                value_type=field_def["type"],
            )

        num_valid = 0
        num_failed = 0
        errors = []

        for i, sample in enumerate(samples):
            try:
                # Parse if needed
                if sample_type == "json":
                    data = json.loads(sample) if isinstance(sample, str) else sample
                else:
                    data = sample

                # Extract fields
                field_values = self.extract_fields(data)

                # Update distributions
                for fv in field_values:
                    if fv.field_name in distributions:
                        distributions[fv.field_name].add(fv.value)

                num_valid += 1

            except Exception as e:
                num_failed += 1
                if len(errors) < 100:
                    errors.append(f"Sample {i}: {str(e)[:100]}")

        return ExtractionResult(
            schema_name=self.schema_name,
            num_samples=len(samples),
            num_valid=num_valid,
            num_failed=num_failed,
            field_distributions=distributions,
            errors=errors,
        )
