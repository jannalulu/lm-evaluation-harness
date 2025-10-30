import json
from difflib import SequenceMatcher
from typing import Any, Dict, List

import pandas as pd
from huggingface_hub import hf_hub_download


# Token bin to n_chars range mapping and max context length
# These ranges are derived from the actual token counts in the dataset
# Format: (min_chars, max_chars, max_tokens_in_bin)
TOKEN_BIN_RANGES = [
    (0, 40000, 8192),           # 4k-8k tokens
    (35000, 82000, 16384),      # 8k-16k tokens
    (73000, 162000, 32768),     # 16k-32k tokens
    (154000, 320000, 65536),    # 32k-64k tokens
    (315000, 632000, 131072),   # 64k-128k tokens
    (630000, 1260000, 262144),  # 128k-256k tokens
    (1280000, 2510000, 524288), # 256k-512k tokens
    (2560000, 5110000, 1048576),# 512k-1M tokens
]


def assign_context_length(n_chars: int) -> int:
    """
    Assign context_length metadata based on n_chars value.

    Args:
        n_chars: Character count from the dataset

    Returns:
        Maximum token count for the bin (e.g., 8192, 16384, etc.)
    """
    for min_chars, max_chars, max_tokens in TOKEN_BIN_RANGES:
        if min_chars < n_chars <= max_chars:
            return max_tokens
    # Default to highest bin if outside ranges
    return 1048576


def load_mrcr_dataset(needle_count: int) -> List[Dict[str, Any]]:
    """
    Load MRCR dataset for a specific needle count with metadata.

    Each sample gets a 'context_length' metadata field indicating the maximum
    token count for its bin (e.g., 8192 for 4k-8k bin, 16384 for 8k-16k bin).

    Users can filter by context length using:
        --metadata context_length=8192
        --metadata context_length=16384,32768,65536

    Args:
        needle_count: Number of needles (2, 4, or 8)

    Returns:
        List of dataset samples as dictionaries with metadata
    """
    filename = f"{needle_count}needle.parquet"
    path = hf_hub_download(
        repo_id="openai/mrcr", filename=filename, repo_type="dataset"
    )
    df = pd.read_parquet(path)

    # Add context_length metadata to each sample
    df["context_length"] = df["n_chars"].apply(assign_context_length)

    return df.to_dict("records")


# Dataset loading functions for each needle count
def load_mrcr_2needle(**kwargs):
    """Load 2-needle MRCR dataset (all context lengths)."""
    return load_mrcr_dataset(2)


def load_mrcr_4needle(**kwargs):
    """Load 4-needle MRCR dataset (all context lengths)."""
    return load_mrcr_dataset(4)


def load_mrcr_8needle(**kwargs):
    """Load 8-needle MRCR dataset (all context lengths)."""
    return load_mrcr_dataset(8)


def doc_to_text(doc: Dict[str, Any]) -> str:
    """
    Convert document to text format for the model.

    The prompt field contains a JSON-encoded list of chat messages.
    This formats the entire conversation for the model.

    Args:
        doc: Document dictionary with 'prompt' field containing JSON messages

    Returns:
        The full formatted conversation as text
    """
    messages = json.loads(doc["prompt"])
    # Format all messages into a conversation
    formatted_parts = []
    for msg in messages:
        role = msg["role"].capitalize()
        content = msg["content"]
        formatted_parts.append(f"{role}: {content}")

    return "\n\n".join(formatted_parts)


def process_results(doc: Dict[str, Any], results: List[str]) -> Dict[str, float]:
    """
    Process results for a single document using MRCR scoring logic.

    Grading:
    1. If response doesn't start with random_string_to_prepend, score = 0
    2. Otherwise, remove prefix from both response and answer
    3. Use SequenceMatcher.ratio() to compute similarity (0.0 to 1.0)

    Args:
        doc: Document dictionary with answer and random_string_to_prepend
        results: List with model response

    Returns:
        Dictionary with mrcr_score
    """
    if not results:
        return {"mrcr_score": 0.0}

    response = results[0] if isinstance(results, list) else results
    answer = doc["answer"]
    random_string = doc["random_string_to_prepend"]

    # Grade according to official MRCR logic
    if not response.startswith(random_string):
        score = 0.0
    else:
        # Remove prefix from both
        response_clean = response.removeprefix(random_string)
        answer_clean = answer.removeprefix(random_string)
        # Compute sequence similarity
        score = float(SequenceMatcher(None, response_clean, answer_clean).ratio())

    return {"mrcr_score": score}
