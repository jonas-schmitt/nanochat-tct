"""ESLint field extractor.

Extracts key fields from .eslintrc.json files for distribution comparison.
Based on EVALUATION_PLAN.md field definitions.
"""

from typing import Any, Dict, List, Optional

from .base import FieldExtractor, FieldValue


class ESLintExtractor(FieldExtractor):
    """Extractor for .eslintrc.json files."""

    @property
    def schema_name(self) -> str:
        return "eslintrc"

    @property
    def field_definitions(self) -> List[Dict[str, Any]]:
        """Return field definitions for ESLint.

        Key fields from EVALUATION_PLAN.md:
        - env.browser, env.node, env.es6 (boolean)
        - parserOptions.ecmaVersion (enum/number)
        - parserOptions.sourceType (enum: module, script)
        - Rule severity distribution
        """
        return [
            # Environment booleans
            {
                "name": "env.browser",
                "type": "bool",
                "description": "Browser global variables",
            },
            {
                "name": "env.node",
                "type": "bool",
                "description": "Node.js global variables",
            },
            {
                "name": "env.es6",
                "type": "bool",
                "description": "ES6 global variables",
            },
            {
                "name": "env.es2020",
                "type": "bool",
                "description": "ES2020 global variables",
            },
            {
                "name": "env.es2021",
                "type": "bool",
                "description": "ES2021 global variables",
            },
            {
                "name": "env.jest",
                "type": "bool",
                "description": "Jest global variables",
            },
            {
                "name": "env.mocha",
                "type": "bool",
                "description": "Mocha global variables",
            },
            # Parser options
            {
                "name": "parserOptions.ecmaVersion",
                "type": "enum",
                "enum_values": [
                    "3", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14",
                    "2015", "2016", "2017", "2018", "2019", "2020", "2021", "2022",
                    "latest",
                ],
                "description": "ECMAScript version",
            },
            {
                "name": "parserOptions.sourceType",
                "type": "enum",
                "enum_values": ["module", "script", "commonjs"],
                "description": "Source code type",
            },
            {
                "name": "parserOptions.ecmaFeatures.jsx",
                "type": "bool",
                "description": "Enable JSX parsing",
            },
            # Root/extends
            {
                "name": "root",
                "type": "bool",
                "description": "Stop looking for config in parent directories",
            },
            {
                "name": "extends",
                "type": "array",
                "description": "Config extensions",
            },
            {
                "name": "parser",
                "type": "string",
                "description": "Parser to use",
            },
        ]

    def extract_field(self, data: dict, field_path: str) -> Any:
        """Extract field with normalization for ecmaVersion."""
        value = super().extract_field(data, field_path)

        # Normalize ecmaVersion to string
        if field_path == "parserOptions.ecmaVersion" and value is not None:
            return str(value)

        return value

    def extract_rule_severity_distribution(self, data: dict) -> Dict[str, int]:
        """Extract distribution of rule severities (off/warn/error).

        ESLint rules can be:
        - 0 or "off"
        - 1 or "warn"
        - 2 or "error"
        - [severity, options...]

        Returns:
            Dict with keys 'off', 'warn', 'error' and counts
        """
        rules = data.get("rules", {})
        if not rules:
            return {}

        counts = {"off": 0, "warn": 0, "error": 0}

        for rule_name, config in rules.items():
            # Extract severity from various formats
            severity = None
            if isinstance(config, (int, str)):
                severity = config
            elif isinstance(config, list) and len(config) > 0:
                severity = config[0]

            # Normalize to string
            if severity in (0, "0", "off"):
                counts["off"] += 1
            elif severity in (1, "1", "warn"):
                counts["warn"] += 1
            elif severity in (2, "2", "error"):
                counts["error"] += 1

        return counts

    def extract_fields(self, data: dict) -> List[FieldValue]:
        """Extract all defined fields plus rule severity distribution."""
        results = super().extract_fields(data)

        # Add rule severity distribution as a special field
        severity_dist = self.extract_rule_severity_distribution(data)
        if severity_dist:
            # Report total rules and dominant severity
            total = sum(severity_dist.values())
            if total > 0:
                dominant = max(severity_dist, key=severity_dist.get)
                results.append(FieldValue(
                    field_name="rules.dominant_severity",
                    value=dominant,
                    value_type="enum",
                ))
                results.append(FieldValue(
                    field_name="rules.count",
                    value=total,
                    value_type="number",
                ))

        return results


def get_extractor() -> ESLintExtractor:
    """Factory function to get ESLint extractor."""
    return ESLintExtractor()
