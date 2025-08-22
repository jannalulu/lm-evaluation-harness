import logging
import re
from functools import cache
from typing import TYPE_CHECKING, Dict, Union

import datasets
from transformers import AutoTokenizer


if TYPE_CHECKING:
    import transformers


eval_logger = logging.getLogger(__name__)

DEFAULT_SEQ_LENGTHS = [
    0,
    1024,
    2048,
    4096,
    8192,
    16384,
    32768,
    65536,
    131072,
    262144,
    524288,
    1048576,
]

CONTEXT_LENGTH_MAPPING = {
    "0k": 0,
    "1k": 1024,
    "2k": 2048,
    "4k": 4096,
    "8k": 8192,
    "16k": 16384,
    "32k": 32768,
    "64k": 65536,
    "128k": 131072,
    "256k": 262144,
    "512k": 524288,
    "1M": 1048576,
}


@cache
def get_tokenizer(
    tokenizer=None, pretrained=None, **kwargs
) -> Union["transformers.PreTrainedTokenizer", "transformers.PreTrainedTokenizerFast"]:
    pretrained = tokenizer or pretrained
    assert pretrained, "No tokenizer or pretrained provided."
    eval_logger.info(f"Using tokenizer {pretrained} for babilong tasks.")
    return AutoTokenizer.from_pretrained(pretrained, trust_remote_code=True)


def postprocess_pred(prediction: list[str]) -> list[str]:
    res = []
    for predict_str in prediction:
        predict_str = predict_str.strip()

        # Remove all non-printable characters
        np_pattern = re.compile(r"[\x00-\x1f]")
        predict_str = np_pattern.sub("\n", predict_str).strip()
        res.append(predict_str)

    return res


def load_dataset(
    dataset_path: str, config_name: str = "0k", qa_split: str = "qa1"
) -> Dict[str, datasets.Dataset]:
    """
    Load babilong dataset for a specific configuration and qa split.

    Args:
        dataset_path: HuggingFace dataset path (e.g., "RMT-team/babilong")
        config_name: Context length configuration (e.g., "0k", "1k", "2k", etc.)
        qa_split: QA task to load (e.g., "qa1", "qa2", etc.)

    Returns:
        Dictionary mapping to test dataset
    """
    try:
        # Load specific config and split
        ds = datasets.load_dataset(dataset_path, name=config_name, split=qa_split)

        # Add context length metadata to each sample
        context_tokens = CONTEXT_LENGTH_MAPPING.get(config_name, 0)
        all_samples = []
        for sample in ds:
            sample_with_meta = dict(sample)
            sample_with_meta["max_length"] = context_tokens
            sample_with_meta["config_name"] = config_name
            sample_with_meta["qa_task"] = qa_split
            all_samples.append(sample_with_meta)

        eval_logger.info(
            f"Loaded {len(ds)} samples from {dataset_path} config={config_name} split={qa_split}"
        )

        return {
            "test": datasets.Dataset.from_list(all_samples, split=datasets.Split.TEST)
        }

    except Exception as e:
        eval_logger.error(
            f"Failed to load {dataset_path} config={config_name} split={qa_split}: {e}"
        )
        raise


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    """Add context length metadata to each document."""
    def _process_doc(doc: dict) -> dict:
        # Debug: Print available fields
        print(f"Fields in doc: {list(doc.keys())}")
        print(f"Doc content: {doc}")
        
        # The config name should be available in the dataset info
        config_name = getattr(dataset, 'config_name', '4k')
        context_tokens = CONTEXT_LENGTH_MAPPING.get(config_name, 4096)
        
        out_doc = dict(doc)
        out_doc["max_length"] = context_tokens
        out_doc["config_name"] = config_name
        
        print(f"Output doc fields: {list(out_doc.keys())}")
        return out_doc
    
    return dataset.map(_process_doc)


def load_babilong_1k(**kwargs):
    """Load babilong-1k-samples dataset with all tasks (qa1-qa20) across available context lengths."""
    dataset_path = "RMT-team/babilong-1k-samples"
    # 1k dataset only has up to 128k context lengths
    max_context_lengths = [ctx for ctx in DEFAULT_SEQ_LENGTHS if ctx <= 131072]
    qa_splits = kwargs.get("qa_splits", [f"qa{i}" for i in range(1, 21)])
    return load_dataset(dataset_path, qa_splits, max_context_lengths)


def process_results(doc: dict, results: list[str]) -> dict[str, float]:
    pred = postprocess_pred(results)
    target = doc.get("target", "").strip()
    
    # String match
    score = 1.0 if target.lower() in pred[0].lower() else 0.0
    
    return {"acc": score}
