from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np


NTU_PATTERN = re.compile(
    r"S(?P<setup>\d{3})"
    r"C(?P<camera>\d{3})"
    r"P(?P<subject>\d{3})"
    r"R(?P<repetition>\d{3})"
    r"A(?P<action>\d{3})",
    re.IGNORECASE,
)

# Official NTU RGB+D 60 cross-subject training subjects.
TRAIN_SUBJECTS = {
    1, 2, 4, 5, 8, 9, 13, 14, 15, 16,
    17, 18, 19, 25, 27, 28, 31, 34, 35, 38,
}

NUM_CLASSES = 49
SOURCE_SHAPE = (64, 18, 4, 2)
MODEL_SHAPE = (4, 64, 18, 1)


def parse_arguments():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--features-root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    return parser.parse_args()


def parse_filename(path):

    match = NTU_PATTERN.search(path.stem)

    if match is None:
        raise ValueError(
            f"Invalid NTU filename: {path.name}"
        )

    return {
        key: int(value)
        for key, value in match.groupdict().items()
    }


def validate_file(path):

    data = np.load(
        path,
        mmap_mode="r",
    )

    if data.shape != SOURCE_SHAPE:
        raise ValueError(
            f"Wrong shape {data.shape}: {path}"
        )

    if data.dtype != np.float32:
        raise ValueError(
            f"Wrong dtype {data.dtype}: {path}"
        )

    if not np.isfinite(data).all():
        raise ValueError(
            f"NaN or infinity found: {path}"
        )


def collect_records(features_root):

    records = []

    for action in range(1, NUM_CLASSES + 1):

        action_folder = (
            features_root
            / f"A{action:03d}"
        )

        if not action_folder.is_dir():
            raise FileNotFoundError(
                f"Missing folder: {action_folder}"
            )

        files = sorted(
            action_folder.glob("*.npy")
        )

        if len(files) == 0:
            raise RuntimeError(
                f"No files found in {action_folder}"
            )

        for path in files:

            metadata = parse_filename(path)

            if metadata["action"] != action:
                raise ValueError(
                    f"Action mismatch: {path}"
                )

            validate_file(path)

            records.append(
                {
                    "path": path,
                    "name": path.stem,
                    "subject": metadata["subject"],
                    "label": action - 1,
                }
            )

    return records


def load_sample(path):

    # Source: T,V,C,M = 64,18,4,2
    source = np.load(path)

    # Keep only person 1.
    source = source[..., :1]

    # Convert to C,T,V,M.
    sample = source.transpose(
        2,
        0,
        1,
        3,
    )

    sample = np.ascontiguousarray(
        sample,
        dtype=np.float32,
    )

    if sample.shape != MODEL_SHAPE:
        raise RuntimeError(
            f"Converted shape {sample.shape}: {path}"
        )

    return sample


def build_split(records, split_name):

    number_of_samples = len(records)

    x = np.empty(
        (
            number_of_samples,
            4,
            64,
            18,
            1,
        ),
        dtype=np.float32,
    )

    labels = np.empty(
        number_of_samples,
        dtype=np.int64,
    )

    names = np.empty(
        number_of_samples,
        dtype="<U64",
    )

    for index, record in enumerate(records):

        x[index] = load_sample(
            record["path"]
        )

        labels[index] = record["label"]
        names[index] = record["name"]

        if (
            (index + 1) % 1000 == 0
            or
            (index + 1) == number_of_samples
        ):

            print(
                f"{split_name}: "
                f"{index + 1}/{number_of_samples}",
                flush=True,
            )

    y = np.eye(
        NUM_CLASSES,
        dtype=np.float32,
    )[labels]

    return x, y, names, labels


def main():

    args = parse_arguments()

    features_root = (
        args.features_root.resolve()
    )

    output_path = (
        args.output.resolve()
    )

    if not features_root.is_dir():
        raise FileNotFoundError(
            f"Feature folder not found:\n"
            f"{features_root}"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Checking feature files...")

    records = collect_records(
        features_root
    )

    train_records = [
        record
        for record in records
        if record["subject"] in TRAIN_SUBJECTS
    ]

    test_records = [
        record
        for record in records
        if record["subject"] not in TRAIN_SUBJECTS
    ]

    print()
    print("Total samples:", len(records))
    print("Training samples:", len(train_records))
    print("Testing samples:", len(test_records))

    print("\nBuilding training data...")

    (
        x_train,
        y_train,
        train_names,
        train_labels,
    ) = build_split(
        train_records,
        "train",
    )

    print("\nBuilding testing data...")

    (
        x_test,
        y_test,
        test_names,
        test_labels,
    ) = build_split(
        test_records,
        "test",
    )

    expected_labels = set(
        range(NUM_CLASSES)
    )

    if set(np.unique(train_labels)) != expected_labels:
        raise RuntimeError(
            "Training split does not contain "
            "all 49 classes."
        )

    if set(np.unique(test_labels)) != expected_labels:
        raise RuntimeError(
            "Testing split does not contain "
            "all 49 classes."
        )

    np.savez(
        output_path,

        x_train=x_train,
        y_train=y_train,

        x_test=x_test,
        y_test=y_test,

        train_names=train_names,
        test_names=test_names,
    )

    print()
    print("Dataset saved:", output_path)

    print(
        "x_train:",
        x_train.shape,
        x_train.dtype,
    )

    print(
        "y_train:",
        y_train.shape,
        y_train.dtype,
    )

    print(
        "x_test:",
        x_test.shape,
        x_test.dtype,
    )

    print(
        "y_test:",
        y_test.shape,
        y_test.dtype,
    )

    print(
        "Note: source setup S013 is absent, "
        "but the official subject split was preserved."
    )


if __name__ == "__main__":
    main()