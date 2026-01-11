"""JSON Position Classifier for Semantic Loss Analysis.

Classifies each byte position in a JSON string as one of:
- 'syntax': Structural characters ({, }, [, ], :, ,, ", whitespace)
- 'key': Field name content (strings before : in objects)
- 'value': Actual value content (strings after :, numbers, booleans, null, array elements)

This enables separating UTF8 model loss into:
- Syntax loss: Easy predictions (deterministic from grammar)
- Key loss: Schema-determined predictions (field names)
- Value loss: Semantic predictions (actual content choices)

For fair comparison with TCT (which only predicts values), we compare:
- UTF8 value_loss vs TCT total_loss
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class JsonPositionClassifier:
    """Classify byte positions in JSON as syntax/key/value.

    Usage:
        classifier = JsonPositionClassifier('{"kind": "Pod", "count": 3}')
        category = classifier.get_category(2)  # 'key' (the 'k' in 'kind')
        category = classifier.get_category(10)  # 'value' (the 'P' in 'Pod')
    """

    json_str: str
    categories: List[str] = None  # Populated by __post_init__

    def __post_init__(self):
        """Classify all positions on initialization."""
        self.categories = ['syntax'] * len(self.json_str)
        self._classify()

    def _classify(self):
        """Walk JSON and mark each position as syntax/key/value."""
        i = 0
        stack = []  # Track context: 'object' or 'array'
        expecting_key = False
        expecting_value = False

        while i < len(self.json_str):
            c = self.json_str[i]

            if c == '{':
                stack.append('object')
                expecting_key = True
                expecting_value = False
                # '{' is syntax
                i += 1
            elif c == '[':
                stack.append('array')
                expecting_value = True  # Array elements are values
                expecting_key = False
                # '[' is syntax
                i += 1
            elif c == '}':
                if stack:
                    stack.pop()
                # Restore parent context expectations
                if stack:
                    if stack[-1] == 'object':
                        expecting_key = False  # After }, might be , or }
                        expecting_value = False
                    else:  # array
                        expecting_value = True
                # '}' is syntax
                i += 1
            elif c == ']':
                if stack:
                    stack.pop()
                # Restore parent context expectations
                if stack:
                    if stack[-1] == 'object':
                        expecting_key = False
                        expecting_value = False
                    else:  # array
                        expecting_value = True
                # ']' is syntax
                i += 1
            elif c == ':':
                expecting_key = False
                expecting_value = True
                # ':' is syntax
                i += 1
            elif c == ',':
                if stack and stack[-1] == 'object':
                    expecting_key = True
                    expecting_value = False
                else:  # array or top-level
                    expecting_value = True
                    expecting_key = False
                # ',' is syntax
                i += 1
            elif c == '"':
                # Parse string - find end quote
                start_quote = i
                i += 1  # Skip opening quote
                string_start = i

                while i < len(self.json_str):
                    if self.json_str[i] == '\\' and i + 1 < len(self.json_str):
                        # Skip escaped character
                        i += 2
                    elif self.json_str[i] == '"':
                        break
                    else:
                        i += 1

                string_end = i

                # Classify string content (not the quotes)
                category = 'key' if expecting_key else 'value'
                for j in range(string_start, string_end):
                    self.categories[j] = category

                # Quotes are syntax
                # self.categories[start_quote] = 'syntax'  # Already default
                # self.categories[i] = 'syntax' if i < len(self.json_str) else None

                # After a string
                if expecting_key:
                    expecting_key = False
                    # Expect : next
                else:
                    expecting_value = False
                    # Expect , or } or ] next

                i += 1  # Skip closing quote

            elif c in ' \t\n\r':
                # Whitespace is syntax
                i += 1
            else:
                # Non-string value: number, boolean, null
                # Find extent of this token
                token_start = i
                while i < len(self.json_str) and self.json_str[i] not in ' \t\n\r,]}':
                    i += 1
                token_end = i

                # Classify as value if we're expecting a value
                if expecting_value:
                    for j in range(token_start, token_end):
                        self.categories[j] = 'value'

                expecting_value = False

    def get_category(self, byte_offset: int) -> str:
        """Return category for a byte position.

        Args:
            byte_offset: Index into the JSON string

        Returns:
            'syntax', 'key', or 'value'
        """
        if 0 <= byte_offset < len(self.categories):
            return self.categories[byte_offset]
        return 'syntax'

    def get_stats(self) -> dict:
        """Return statistics about the classification."""
        syntax_count = self.categories.count('syntax')
        key_count = self.categories.count('key')
        value_count = self.categories.count('value')
        total = len(self.categories)

        return {
            'total_bytes': total,
            'syntax_bytes': syntax_count,
            'key_bytes': key_count,
            'value_bytes': value_count,
            'syntax_pct': 100 * syntax_count / total if total > 0 else 0,
            'key_pct': 100 * key_count / total if total > 0 else 0,
            'value_pct': 100 * value_count / total if total > 0 else 0,
        }

    def visualize(self) -> str:
        """Return a visualization of the classification.

        Useful for debugging. Shows the JSON with category markers:
        S = syntax, K = key, V = value
        """
        markers = []
        for cat in self.categories:
            if cat == 'syntax':
                markers.append('S')
            elif cat == 'key':
                markers.append('K')
            else:  # value
                markers.append('V')

        return f"{self.json_str}\n{''.join(markers)}"


def classify_token_bytes(
    token_bytes: bytes,
    byte_offset: int,
    classifier: JsonPositionClassifier,
) -> dict:
    """Classify a token's bytes and return category breakdown.

    For mixed tokens (containing multiple categories), we prioritize semantic content:
    - If ANY byte is 'value', classify as 'value' (semantic content)
    - Else if ANY byte is 'key', classify as 'key' (schema-determined)
    - Else classify as 'syntax'

    This ensures that tokens like '"nginx"' (quotes + value) count as value predictions,
    which is fair for comparing with TCT where the model only predicts values.

    Args:
        token_bytes: The bytes this token represents
        byte_offset: Starting byte offset in the JSON string
        classifier: Pre-computed JsonPositionClassifier

    Returns:
        dict with:
        - 'primary': Category for loss attribution ('syntax', 'key', or 'value')
        - 'syntax_fraction': Fraction of bytes that are syntax
        - 'key_fraction': Fraction of bytes that are keys
        - 'value_fraction': Fraction of bytes that are values
    """
    if not token_bytes:
        return {
            'primary': 'syntax',
            'syntax_fraction': 1.0,
            'key_fraction': 0.0,
            'value_fraction': 0.0,
        }

    syntax_count = 0
    key_count = 0
    value_count = 0

    for i, _ in enumerate(token_bytes):
        cat = classifier.get_category(byte_offset + i)
        if cat == 'syntax':
            syntax_count += 1
        elif cat == 'key':
            key_count += 1
        else:
            value_count += 1

    total = len(token_bytes)

    # Determine primary category - prioritize semantic content
    # If ANY byte is value content, count as value (be generous to semantic content)
    # This handles mixed tokens like '"nginx"' which contain both quotes and value
    if value_count > 0:
        primary = 'value'
    elif key_count > 0:
        primary = 'key'
    else:
        primary = 'syntax'

    return {
        'primary': primary,
        'syntax_fraction': syntax_count / total,
        'key_fraction': key_count / total,
        'value_fraction': value_count / total,
    }


# Test cases for verification
if __name__ == '__main__':
    # Test 1: Simple object
    json1 = '{"kind": "Pod", "count": 3}'
    c1 = JsonPositionClassifier(json1)
    print("Test 1:", json1)
    print(c1.visualize())
    print(c1.get_stats())
    print()

    # Test 2: Nested object
    json2 = '{"spec": {"replicas": 3}}'
    c2 = JsonPositionClassifier(json2)
    print("Test 2:", json2)
    print(c2.visualize())
    print(c2.get_stats())
    print()

    # Test 3: Array
    json3 = '{"items": [1, 2, "three"]}'
    c3 = JsonPositionClassifier(json3)
    print("Test 3:", json3)
    print(c3.visualize())
    print(c3.get_stats())
    print()

    # Test 4: Boolean and null
    json4 = '{"enabled": true, "data": null}'
    c4 = JsonPositionClassifier(json4)
    print("Test 4:", json4)
    print(c4.visualize())
    print(c4.get_stats())
    print()

    # Test 5: Escaped string
    json5 = '{"path": "C:\\\\Users\\\\test"}'
    c5 = JsonPositionClassifier(json5)
    print("Test 5:", json5)
    print(c5.visualize())
    print(c5.get_stats())
