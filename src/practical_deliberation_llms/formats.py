from __future__ import annotations

from typing import Dict, List


# Canonical field name used for label-scoring JSON objects across the
# project. Both the structured_outputs grammar and the parsing helpers
# (regexes, JSON extractors) should derive from this constant to keep
# the contract in sync.
LABEL_FIELD_NAME = "label"


def make_label_json_schema(labels: List[str]) -> Dict[str, object]:
    """Return the JSON schema used for label objects.

    This schema is plugged into the structured_outputs grammar and is
    also assumed by parsing utilities when they extract the label
    value from JSON-like strings.
    """

    return {
        "type": "object",
        "properties": {
            LABEL_FIELD_NAME: {
                "type": "string",
                "enum": labels,
            },
        },
        "required": [LABEL_FIELD_NAME],
        "additionalProperties": False,
    }
