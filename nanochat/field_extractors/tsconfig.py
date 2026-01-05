"""TSConfig field extractor.

Extracts key fields from tsconfig.json files for distribution comparison.
Based on EVALUATION_PLAN.md field definitions.
"""

from typing import Any, Dict, List

from .base import FieldExtractor


class TSConfigExtractor(FieldExtractor):
    """Extractor for tsconfig.json files."""

    @property
    def schema_name(self) -> str:
        return "tsconfig"

    @property
    def field_definitions(self) -> List[Dict[str, Any]]:
        """Return field definitions for TSConfig.

        Key fields from EVALUATION_PLAN.md:
        - Boolean flags: strict, esModuleInterop, skipLibCheck, forceConsistentCasingInFileNames
        - Enum fields: target, module, moduleResolution, jsx
        """
        return [
            # Boolean compiler options
            {
                "name": "compilerOptions.strict",
                "type": "bool",
                "description": "Enable all strict type checking options",
            },
            {
                "name": "compilerOptions.esModuleInterop",
                "type": "bool",
                "description": "Emit additional JS for ES module interop",
            },
            {
                "name": "compilerOptions.skipLibCheck",
                "type": "bool",
                "description": "Skip type checking of declaration files",
            },
            {
                "name": "compilerOptions.forceConsistentCasingInFileNames",
                "type": "bool",
                "description": "Disallow inconsistently-cased references to same file",
            },
            {
                "name": "compilerOptions.declaration",
                "type": "bool",
                "description": "Generate .d.ts declaration files",
            },
            {
                "name": "compilerOptions.sourceMap",
                "type": "bool",
                "description": "Generate source map files",
            },
            {
                "name": "compilerOptions.noImplicitAny",
                "type": "bool",
                "description": "Error on expressions with implied any type",
            },
            {
                "name": "compilerOptions.noEmit",
                "type": "bool",
                "description": "Do not emit outputs",
            },
            # Enum compiler options
            {
                "name": "compilerOptions.target",
                "type": "enum",
                "enum_values": [
                    "ES3", "ES5", "ES6", "ES2015", "ES2016", "ES2017",
                    "ES2018", "ES2019", "ES2020", "ES2021", "ES2022",
                    "ES2023", "ESNext",
                ],
                "description": "ECMAScript target version",
            },
            {
                "name": "compilerOptions.module",
                "type": "enum",
                "enum_values": [
                    "CommonJS", "AMD", "UMD", "System", "ES6", "ES2015",
                    "ES2020", "ES2022", "ESNext", "Node16", "NodeNext", "None",
                ],
                "description": "Module code generation",
            },
            {
                "name": "compilerOptions.moduleResolution",
                "type": "enum",
                "enum_values": [
                    "Classic", "Node", "Node10", "Node16", "NodeNext", "Bundler",
                ],
                "description": "Module resolution strategy",
            },
            {
                "name": "compilerOptions.jsx",
                "type": "enum",
                "enum_values": [
                    "preserve", "react", "react-jsx", "react-jsxdev", "react-native",
                ],
                "description": "JSX code generation",
            },
            {
                "name": "compilerOptions.lib",
                "type": "array",
                "description": "Library files to include",
            },
            # Additional useful fields
            {
                "name": "compilerOptions.outDir",
                "type": "string",
                "description": "Output directory",
            },
            {
                "name": "compilerOptions.rootDir",
                "type": "string",
                "description": "Root directory of source files",
            },
        ]

    def extract_field(self, data: dict, field_path: str) -> Any:
        """Extract field with case-insensitive enum matching.

        TypeScript compiler options are case-insensitive for enum values,
        so we normalize them for consistent counting.
        """
        value = super().extract_field(data, field_path)

        # Normalize enum values to canonical case
        if value is not None and isinstance(value, str):
            # Find the field definition
            for field_def in self.field_definitions:
                if field_def["name"] == field_path and field_def["type"] == "enum":
                    # Try to match to canonical enum value (case-insensitive)
                    enum_values = field_def.get("enum_values", [])
                    value_lower = value.lower()
                    for canonical in enum_values:
                        if canonical.lower() == value_lower:
                            return canonical
                    # Not a known enum value, return as-is
                    return value

        return value


def get_extractor() -> TSConfigExtractor:
    """Factory function to get TSConfig extractor."""
    return TSConfigExtractor()
