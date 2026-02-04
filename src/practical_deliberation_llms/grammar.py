"""Structured output grammar helpers for constrained label scoring.

This module centralizes construction of the XGrammar / structured_outputs
specifications used with vLLM's OpenAI-compatible chat API.

Currently we expose a single helper:

- ``make_structured_label_grammar(reasoning, labels)``: forces the model to
  emit ``<think>{reasoning}</think>{"label": "<LABEL>"}`` where ``<LABEL>``
  must be one of the provided ``labels``.

The reasoning segment is enforced as a ``const_string``; there is no relaxed
``any_text`` variant in this module.
"""

from __future__ import annotations

from typing import Dict, List

import json


def make_structured_label_grammar(
    reasoning: str, labels: List[str]
) -> Dict[str, object]:
    """Return ``extra_body`` payload for structured label scoring.

    The returned dict is intended to be passed as ``extra_body`` to
    ``OpenAI.chat.completions.create``. It constrains the model output to the
    exact format::

        <think>{reasoning}</think>{"label": "<LABEL>"}

    where ``<LABEL>`` is restricted to the provided ``labels`` list.

    For compatibility with vLLM's `StructuredOutputsParams`, the
    ``structural_tag`` value is JSON-encoded as a string.
    """

    structural_tag_config = {
        "type": "structural_tag",
        "format": {
            "type": "sequence",
            "elements": [
                {
                    "type": "tag",
                    "begin": "<think>",
                    "content": {
                        "type": "const_string",
                        "value": reasoning,
                    },
                    "end": "</think>",
                },
                {
                    "type": "json_schema",
                    "json_schema": {
                        "type": "object",
                        "properties": {
                            "label": {
                                "type": "string",
                                "enum": labels,
                            },
                        },
                        "required": ["label"],
                        "additionalProperties": False,
                    },
                },
            ],
        },
    }

    return {"structured_outputs": {"structural_tag": json.dumps(structural_tag_config)}}
