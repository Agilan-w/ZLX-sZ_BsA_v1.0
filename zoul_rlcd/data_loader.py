# Copyright 2026 ZLX-CDE V2.5
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Real-world dataset loading and dynamic collation for the ZLX-CDE V2.5 Decision Engine.

Supports loading datasets from local files (.csv, .jsonl, .json, .parquet) or
from the Hugging Face Hub, with:

- Automated string-to-integer label encoding producing id2label and label2id maps
- Dynamic batch padding via transformers.DataCollatorWithPadding to minimize
  VRAM footprint on 8 GB GPUs
- Clean PyTorch DataLoader generation for train and validation splits
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional, Union

import torch
from datasets import ClassLabel, Dataset, DatasetDict, load_dataset, load_from_disk
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


def load_raw_dataset(
    data_path: Union[str, Path],
    dataset_config: Optional[str] = None,
) -> Union[Dataset, DatasetDict]:
    """Load a raw dataset from local disk or the Hugging Face Hub.

    Parameters
    ----------
    data_path : str | Path
        Path to a local file (.csv, .jsonl, .json, .parquet), directory, or
        a Hugging Face Hub dataset repository name (e.g., 'imdb', 'ag_news').
    dataset_config : str | None
        Optional sub-configuration name for Hugging Face Hub datasets.

    Returns
    -------
    Dataset | DatasetDict
        Loaded dataset or dataset dictionary.

    Raises
    ------
    FileNotFoundError
        If data_path points to a non-existent local path with file extension.
    ValueError
        If an unsupported file extension is provided.
    """
    path_obj = Path(data_path)

    # Local file detection
    if path_obj.is_file():
        suffix = path_obj.suffix.lower()
        logger.info("Loading local dataset file '%s' (type: %s).", path_obj, suffix)
        if suffix == ".csv":
            return load_dataset("csv", data_files=str(path_obj))
        elif suffix in (".jsonl", ".json"):
            return load_dataset("json", data_files=str(path_obj))
        elif suffix == ".parquet":
            return load_dataset("parquet", data_files=str(path_obj))
        else:
            raise ValueError(
                f"Unsupported file format '{suffix}' for {data_path}. "
                "Supported formats: .csv, .jsonl, .json, .parquet."
            )

    # Local directory detection (e.g., save_to_disk format)
    if path_obj.is_dir():
        try:
            logger.info("Attempting load_from_disk on directory '%s'.", path_obj)
            return load_from_disk(str(path_obj))
        except Exception:
            logger.info("Falling back to load_dataset for directory '%s'.", path_obj)
            return load_dataset(str(path_obj))

    # Hugging Face Hub repository
    logger.info("Loading dataset '%s' from Hugging Face Hub...", data_path)
    if dataset_config:
        return load_dataset(str(data_path), dataset_config)
    return load_dataset(str(data_path))


def prepare_dataset_splits(
    raw_dataset: Union[Dataset, DatasetDict],
    val_split: float = 0.2,
    seed: int = 42,
) -> tuple[Dataset, Dataset]:
    """Extract or partition raw dataset into train and validation splits.

    Parameters
    ----------
    raw_dataset : Dataset | DatasetDict
        Input dataset or dictionary of splits.
    val_split : float
        Fraction of data reserved for validation if dataset is unpartitioned.
    seed : int
        Random seed for reproducible train/val partitioning.

    Returns
    -------
    tuple[Dataset, Dataset]
        (train_dataset, validation_dataset)
    """
    if isinstance(raw_dataset, DatasetDict):
        if "train" in raw_dataset:
            train_ds = raw_dataset["train"]
            if "validation" in raw_dataset:
                val_ds = raw_dataset["validation"]
            elif "val" in raw_dataset:
                val_ds = raw_dataset["val"]
            elif "test" in raw_dataset:
                val_ds = raw_dataset["test"]
            else:
                logger.info(
                    "No validation split found in DatasetDict; partitioning 'train' with val_split=%.2f.",
                    val_split,
                )
                split = train_ds.train_test_split(test_size=val_split, seed=seed)
                train_ds = split["train"]
                val_ds = split["test"]
        else:
            first_split_name = next(iter(raw_dataset.keys()))
            logger.warning(
                "No 'train' split found in DatasetDict. Using split '%s' and partitioning.",
                first_split_name,
            )
            split = raw_dataset[first_split_name].train_test_split(
                test_size=val_split, seed=seed
            )
            train_ds = split["train"]
            val_ds = split["test"]
    else:
        logger.info(
            "Partitioning un-split dataset (%d rows) with val_split=%.2f.",
            len(raw_dataset),
            val_split,
        )
        split = raw_dataset.train_test_split(test_size=val_split, seed=seed)
        train_ds = split["train"]
        val_ds = split["test"]

    return train_ds, val_ds


def extract_label_mappings(
    train_ds: Dataset,
    val_ds: Dataset,
    label_column: str,
) -> tuple[dict[int, str], dict[str, int]]:
    """Automatically discover labels and construct id2label and label2id mappings.

    Parameters
    ----------
    train_ds : Dataset
        Training dataset split.
    val_ds : Dataset
        Validation dataset split.
    label_column : str
        Name of the column containing target class labels.

    Returns
    -------
    tuple[dict[int, str], dict[str, int]]
        id2label and label2id mappings.

    Raises
    ------
    KeyError
        If label_column is not present in the dataset.
    """
    if label_column not in train_ds.column_names:
        raise KeyError(
            f"Label column '{label_column}' not found. "
            f"Available columns: {train_ds.column_names}"
        )

    # 1. Check if dataset features define ClassLabel
    feature = train_ds.features.get(label_column)
    if isinstance(feature, ClassLabel):
        names = feature.names
        id2label = {i: str(name) for i, name in enumerate(names)}
        label2id = {str(name): i for i, name in enumerate(names)}
        logger.info(
            "Extracted %d classes from ClassLabel feature '%s': %s",
            len(id2label),
            label_column,
            id2label,
        )
        return id2label, label2id

    # 2. Extract unique labels from data
    train_unique = set(train_ds[label_column])
    val_unique = set(val_ds[label_column])
    all_unique = train_unique | val_unique

    # Convert to string and sort deterministically
    sorted_labels = sorted(list(all_unique), key=lambda x: str(x))

    id2label = {i: str(lbl) for i, lbl in enumerate(sorted_labels)}
    label2id = {str(lbl): i for i, lbl in enumerate(sorted_labels)}

    logger.info(
        "Discovered %d unique classes in column '%s': %s",
        len(id2label),
        label_column,
        id2label,
    )
    return id2label, label2id


def tokenize_and_encode(
    train_ds: Dataset,
    val_ds: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    text_column: str,
    label_column: str,
    label2id: dict[str, int],
    max_length: int = 512,
) -> tuple[Dataset, Dataset]:
    """Tokenize text column and map labels to integer target IDs.

    Note that dynamic padding is deferred to DataCollatorWithPadding to
    save GPU memory.

    Parameters
    ----------
    train_ds : Dataset
        Training dataset.
    val_ds : Dataset
        Validation dataset.
    tokenizer : PreTrainedTokenizerBase
        Tokenizer for encoding text inputs.
    text_column : str
        Column name containing input text.
    label_column : str
        Column name containing target labels.
    label2id : dict[str, int]
        Mapping from label string to integer index.
    max_length : int
        Maximum sequence length for truncation.

    Returns
    -------
    tuple[Dataset, Dataset]
        Processed (train_ds, val_ds) containing 'input_ids', 'attention_mask',
        and 'labels'.
    """
    if text_column not in train_ds.column_names:
        raise KeyError(
            f"Text column '{text_column}' not found. "
            f"Available columns: {train_ds.column_names}"
        )

    def preprocess_batch(batch: dict[str, list[Any]]) -> dict[str, Any]:
        # Clean text inputs (handle None or non-string values)
        raw_texts = [
            str(t) if t is not None else "" for t in batch[text_column]
        ]

        tokenized = tokenizer(
            raw_texts,
            truncation=True,
            max_length=max_length,
            # No padding here -- DataCollatorWithPadding handles per-batch dynamic padding
        )

        # Encode labels as integer IDs
        tokenized["labels"] = [
            label2id[str(lbl)] for lbl in batch[label_column]
        ]
        return tokenized

    cols_to_remove_train = [
        col for col in train_ds.column_names
        if col not in ("input_ids", "attention_mask", "labels")
    ]
    cols_to_remove_val = [
        col for col in val_ds.column_names
        if col not in ("input_ids", "attention_mask", "labels")
    ]

    logger.info("Tokenizing training split (%d samples)...", len(train_ds))
    tokenized_train = train_ds.map(
        preprocess_batch,
        batched=True,
        remove_columns=cols_to_remove_train,
        desc="Tokenizing train set",
    )

    logger.info("Tokenizing validation split (%d samples)...", len(val_ds))
    tokenized_val = val_ds.map(
        preprocess_batch,
        batched=True,
        remove_columns=cols_to_remove_val,
        desc="Tokenizing val set",
    )

    return tokenized_train, tokenized_val


def load_decision_dataset(
    data_path: Union[str, Path],
    tokenizer: PreTrainedTokenizerBase,
    text_column: str = "text",
    label_column: str = "label",
    batch_size: int = 4,
    max_length: int = 512,
    val_split: float = 0.2,
    seed: int = 42,
    dataset_config: Optional[str] = None,
    pad_to_multiple_of: Optional[int] = 8,
) -> tuple[DataLoader, DataLoader, dict[int, str], dict[str, int], int]:
    """End-to-end dataset loader with dynamic padding and label extraction.

    Parameters
    ----------
    data_path : str | Path
        Path to local file (.csv, .jsonl, .json, .parquet) or HF Hub dataset.
    tokenizer : PreTrainedTokenizerBase
        Tokenizer for model input encoding.
    text_column : str
        Column containing input text.
    label_column : str
        Column containing target labels.
    batch_size : int
        Micro-batch size for training.
    max_length : int
        Maximum sequence length threshold.
    val_split : float
        Validation split fraction if unpartitioned.
    seed : int
        Random seed for shuffling and splitting.
    dataset_config : str | None
        Optional sub-configuration name for HF Hub dataset.
    pad_to_multiple_of : int | None
        Pad batch length to a multiple of this value (default: 8, optimal for
        NVIDIA Tensor Cores).

    Returns
    -------
    tuple[DataLoader, DataLoader, dict[int, str], dict[str, int], int]
        (train_loader, val_loader, id2label, label2id, num_classes)
    """
    # Ensure tokenizer has a pad token configured
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info("Set tokenizer.pad_token to eos_token ('%s').", tokenizer.eos_token)
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            logger.info("Added special '[PAD]' token to tokenizer.")

    # 1. Load raw dataset
    raw_dataset = load_raw_dataset(data_path, dataset_config=dataset_config)

    # 2. Partition into train and val splits
    train_ds, val_ds = prepare_dataset_splits(
        raw_dataset, val_split=val_split, seed=seed
    )

    # 3. Discover and build label mappings
    id2label, label2id = extract_label_mappings(
        train_ds=train_ds, val_ds=val_ds, label_column=label_column
    )
    num_classes = len(id2label)

    # 4. Tokenize and encode
    tokenized_train, tokenized_val = tokenize_and_encode(
        train_ds=train_ds,
        val_ds=val_ds,
        tokenizer=tokenizer,
        text_column=text_column,
        label_column=label_column,
        label2id=label2id,
        max_length=max_length,
    )

    # 5. Construct DataCollatorWithPadding for dynamic padding
    collate_fn = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding=True,
        pad_to_multiple_of=pad_to_multiple_of,
        return_tensors="pt",
    )

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        tokenized_train,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=len(tokenized_train) >= batch_size,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        tokenized_val,
        batch_size=batch_size * 2,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=False,
        pin_memory=pin_memory,
    )

    logger.info(
        "DataLoaders ready -- Train: %d samples (%d batches) | "
        "Val: %d samples (%d batches) | Classes: %d",
        len(tokenized_train),
        len(train_loader),
        len(tokenized_val),
        len(val_loader),
        num_classes,
    )

    return train_loader, val_loader, id2label, label2id, num_classes
