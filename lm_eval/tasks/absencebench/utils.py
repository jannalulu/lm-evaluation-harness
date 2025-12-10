"""
Utility functions for AbsenceBench tasks.

AbsenceBench evaluates LLMs' ability to identify conspicuously missing information
from long inputs. Unlike needle-in-a-haystack tests, models must identify what has
been intentionally omitted rather than finding irrelevant inserted content.

The benchmark uses Micro F1 scoring: 2*TP / (2*TP + FP + FN)
"""


def normalize_text(text: str) -> str:
    """Normalize text for comparison by converting to lowercase and removing extra whitespace."""
    return " ".join(text.lower().strip().split())


def calculate_micro_f1(
    model_response: str,
    omitted_items: list[str],
    original_items: list[str],
    omitted_indices: list[int],
) -> float:
    """
    Calculate Micro F1 score for absence detection.

    Args:
        model_response: The model's generated response
        omitted_items: List of items that were actually omitted
        original_items: List of all items in the original context
        omitted_indices: Indices of omitted items in the original context

    Returns:
        Micro F1 score (float between 0 and 1)
    """
    # Normalize model response
    response_lower = model_response.lower()

    # Track true positives, false positives, false negatives
    tp = 0
    fp = 0
    fn = 0

    # Check which omitted items the model correctly identified
    identified_omissions = set()
    for idx, item in enumerate(omitted_items):
        normalized_item = normalize_text(item)
        # Check if this omitted item appears in the model's response
        if normalized_item in response_lower or item.lower() in response_lower:
            tp += 1
            identified_omissions.add(idx)

    # Count false negatives (omitted items the model missed)
    fn = len(omitted_items) - tp

    # Check for false positives (non-omitted items the model incorrectly identified)
    omitted_set = set(omitted_indices)
    for idx, item in enumerate(original_items):
        # Skip items that were actually omitted
        if idx in omitted_set:
            continue

        normalized_item = normalize_text(item)
        # If a non-omitted item appears in the response, it's a false positive
        if normalized_item in response_lower or item.lower() in response_lower:
            fp += 1

    # Calculate Micro F1
    denominator = 2 * tp + fp + fn
    if denominator == 0:
        # If nothing was omitted and model found nothing, that's perfect
        return 1.0 if len(omitted_items) == 0 else 0.0

    micro_f1 = (2 * tp) / denominator
    return micro_f1


def process_results_poetry(doc: dict, results: list[str]) -> dict[str, float]:
    """
    Process results for poetry domain.

    Poetry contexts are split by lines. The model should identify which lines
    are missing from the modified context.
    """
    model_response = results[0]

    # Split original context into lines
    original_lines = doc["original_context"].split("\n")

    # Get omitted context and indices
    omitted_lines = doc["omitted_context"]
    omitted_indices = doc["omitted_index"]

    # Calculate Micro F1
    micro_f1 = calculate_micro_f1(
        model_response=model_response,
        omitted_items=omitted_lines,
        original_items=original_lines,
        omitted_indices=omitted_indices,
    )

    return {"micro_f1": micro_f1}


def process_results_numerical(doc: dict, results: list[str]) -> dict[str, float]:
    """
    Process results for numerical domain.

    Numerical contexts contain sequences of numbers, one per line.
    The model should identify which numbers are missing.
    """
    model_response = results[0]

    # Split original context into lines (each line is a number)
    original_numbers = doc["original_context"].split("\n")

    # Get omitted context and indices
    omitted_numbers = doc["omitted_context"]
    omitted_indices = doc["omitted_index"]

    # Calculate Micro F1
    micro_f1 = calculate_micro_f1(
        model_response=model_response,
        omitted_items=omitted_numbers,
        original_items=original_numbers,
        omitted_indices=omitted_indices,
    )

    return {"micro_f1": micro_f1}


def process_results_code(doc: dict, results: list[str]) -> dict[str, float]:
    """
    Process results for GitHub PR/code domain.

    Code contexts are git diffs. The model should identify which changed lines
    (additions/deletions) are missing from the modified diff.
    """
    model_response = results[0]

    # Split original diff into lines
    original_lines = doc["original_context"].split("\n")

    # Get omitted context and indices
    omitted_lines = doc["omitted_context"]
    omitted_indices = doc["omitted_index"]

    # Calculate Micro F1
    micro_f1 = calculate_micro_f1(
        model_response=model_response,
        omitted_items=omitted_lines,
        original_items=original_lines,
        omitted_indices=omitted_indices,
    )

    return {"micro_f1": micro_f1}
