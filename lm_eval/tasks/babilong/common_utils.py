import logging
import re
from functools import cache
from typing import TYPE_CHECKING, Union

import datasets
from transformers import AutoTokenizer


if TYPE_CHECKING:
    import transformers


eval_logger = logging.getLogger(__name__)


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


def load_dataset(**kwargs):
    # Get config name from metadata, default to "0k"
    config_name = kwargs.get("config_name", "0k")

    eval_logger.info(
        f"Loading babilong dataset: config={config_name}"
    )

    # Load all qa splits and return as dict - task system will pick the right one
    qa_splits = [f"qa{i}" for i in range(1, 21)]
    dataset_dict = {}
    
    for qa_split in qa_splits:
        try:
            dataset = datasets.load_dataset(
                "RMT-team/babilong", name=config_name, split=qa_split
            )
            dataset_dict[qa_split] = dataset
        except Exception as e:
            eval_logger.warning(f"Failed to load {qa_split}: {e}")
            
    return dataset_dict


def process_results(doc: dict, results: list[str]) -> dict[str, float]:
    pred = postprocess_pred(results)
    target = doc.get("target", "").strip()

    # String match
    score = 1.0 if target.lower() in pred[0].lower() else 0.0

    return {"acc": score}
