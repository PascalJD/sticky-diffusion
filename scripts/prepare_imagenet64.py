"""Build a TFDS-compatible cache for downsampled ImageNet 64x64.

The official TFDS builder (`downsampled_imagenet/64x64`) fetches
`train_64x64.tar` / `valid_64x64.tar` from image-net.org, whose URLs are
long dead. This script consumes the alternative Chrabaszcz pickle format
distributed as `Imagenet64_train_part{1,2}.zip` + `Imagenet64_val.zip`
from image-net.org's downloads page, and materializes a TFDS cache at
the same path that `tfds.load("downsampled_imagenet/64x64")` would use.

Expected input layout (after unzipping):
  {manual_dir}/train_data_batch_1 ... train_data_batch_10
  {manual_dir}/val_data

Output layout (matches TFDS generator conventions):
  {data_dir}/sticky_imagenet64/2.0.0/
    sticky_imagenet64-train.tfrecord-00000-of-NNN ...
    sticky_imagenet64-validation.tfrecord-00000-of-MMM ...
    dataset_info.json
    features.json

Then set the Hydra override:
  experiment.dataset.tfds_name=sticky_imagenet64
and point `experiment.dataset.data_dir` to `{data_dir}`.
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

DATASET_NAME = "sticky_imagenet64"
VERSION = "2.0.0"
IMAGE_SHAPE = (64, 64, 3)
TRAIN_BATCHES = [f"train_data_batch_{i}" for i in range(1, 11)]
VAL_BATCHES = ["val_data"]


class StickyImagenet64(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version(VERSION)

    def _info(self) -> tfds.core.DatasetInfo:
        return tfds.core.DatasetInfo(
            builder=self,
            features=tfds.features.FeaturesDict(
                {
                    "image": tfds.features.Image(shape=IMAGE_SHAPE, dtype=np.uint8),
                    "label": tfds.features.ClassLabel(num_classes=1000),
                }
            ),
            supervised_keys=("image", "label"),
            description="Chrabaszcz downsampled ImageNet 64x64 (local pickle import).",
        )

    def _split_generators(self, dl_manager):
        manual = Path(dl_manager.manual_dir)
        missing = [b for b in TRAIN_BATCHES + VAL_BATCHES if not (manual / b).exists()]
        if missing:
            raise FileNotFoundError(
                f"Missing pickle batches in {manual}: {missing}\n"
                f"Unzip Imagenet64_train_part1.zip, Imagenet64_train_part2.zip, "
                f"Imagenet64_val.zip into this directory first."
            )
        return {
            "train": self._generate_examples(
                [manual / b for b in TRAIN_BATCHES], split_name="train"
            ),
            "validation": self._generate_examples(
                [manual / b for b in VAL_BATCHES], split_name="validation"
            ),
        }

    def _generate_examples(self, paths, split_name):
        counter = 0
        for path in paths:
            with open(path, "rb") as f:
                batch = pickle.load(f)
            data = batch["data"]
            labels = batch["labels"]
            n = data.shape[0]
            assert data.shape[1] == 3 * 64 * 64, (
                f"Unexpected pickle shape {data.shape} in {path}"
            )
            images = data.reshape(n, 3, 64, 64).transpose(0, 2, 3, 1).astype(np.uint8)
            for i in range(n):
                key = f"{split_name}_{counter}"
                counter += 1
                yield key, {
                    "image": images[i],
                    "label": int(labels[i]) - 1,
                }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data-dir",
        required=True,
        help="TFDS data root (e.g. $RCAC_SCRATCH/sticky-diffusion/data/tfds).",
    )
    ap.add_argument(
        "--manual-dir",
        required=True,
        help="Directory containing the unzipped train_data_batch_* and val_data pickles.",
    )
    args = ap.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    manual_dir = os.path.abspath(args.manual_dir)
    os.makedirs(data_dir, exist_ok=True)

    print(f"data_dir={data_dir}")
    print(f"manual_dir={manual_dir}")

    # Suppress TF chatter.
    tf.get_logger().setLevel("ERROR")

    builder = StickyImagenet64(data_dir=data_dir)
    builder.download_and_prepare(
        download_config=tfds.download.DownloadConfig(manual_dir=manual_dir),
    )
    print(f"Built {DATASET_NAME}/{VERSION} at {builder.data_dir}")


if __name__ == "__main__":
    sys.exit(main())
