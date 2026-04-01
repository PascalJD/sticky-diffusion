from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np

from sticky.data.cifar10_discrete import make_cifar10_iterator
from sticky.data.tfds_discrete_image import make_tfds_discrete_image_iterator
from sticky.tasks.cifar10_discrete import CIFAR10DiscreteTask
from sticky.tasks.cifar10_sjd import CIFAR10SJDTask
from sticky.tasks import tfds_discrete_image as tfds_task_mod
from sticky.data import cifar10_discrete as cifar10_data_mod
from sticky.data import tfds_discrete_image as tfds_data_mod


@dataclass
class _TFDSStub:
    dataset: object
    load_calls: list[dict[str, object]] = field(default_factory=list)

    def load(self, dataset_name, *, split, data_dir, shuffle_files, as_supervised):
        self.load_calls.append(
            {
                "dataset_name": dataset_name,
                "split": split,
                "data_dir": data_dir,
                "shuffle_files": shuffle_files,
                "as_supervised": as_supervised,
            }
        )
        return self.dataset

    def as_numpy(self, dataset):
        return dataset.as_numpy_iterator()


class _FakeDataset:
    def __init__(self, examples):
        self.examples = list(examples)
        first = self.examples[0] if self.examples else {}
        self.element_spec = {key: object() for key in first}
        self.last_shuffle_kwargs = None

    def shuffle(self, buffer_size, seed, reshuffle_each_iteration):
        self.last_shuffle_kwargs = {
            "buffer_size": buffer_size,
            "seed": seed,
            "reshuffle_each_iteration": reshuffle_each_iteration,
        }
        return self

    def repeat(self):
        return self

    def map(self, fn, num_parallel_calls=None):
        return _FakeDataset([fn(ex) for ex in self.examples])

    def batch(self, batch_size, drop_remainder):
        batches = []
        for start in range(0, len(self.examples), batch_size):
            chunk = self.examples[start : start + batch_size]
            if drop_remainder and len(chunk) < batch_size:
                break
            batch = {
                key: np.stack([np.asarray(ex[key]) for ex in chunk], axis=0)
                for key in chunk[0]
            }
            batches.append(batch)
        return _FakeDataset(batches)

    def prefetch(self, buffer_size):
        return self

    def as_numpy_iterator(self):
        return iter(self.examples)


class _FakeTF:
    uint8 = np.uint8
    int32 = np.int32
    data = SimpleNamespace(
        AUTOTUNE=object(),
        experimental=SimpleNamespace(cardinality=lambda ds: SimpleNamespace(numpy=lambda: len(ds.examples))),
    )

    @staticmethod
    def cast(value, dtype):
        return np.asarray(value, dtype=dtype)


def _make_dataset(*, images: np.ndarray, labels: np.ndarray | None = None):
    examples = []
    for idx in range(images.shape[0]):
        example = {"image": images[idx]}
        if labels is not None:
            example["label"] = labels[idx]
        examples.append(example)
    return _FakeDataset(examples)


def test_tfds_loader_supports_labeled_examples(monkeypatch):
    images = np.arange(4 * 32 * 32 * 3, dtype=np.uint8).reshape(4, 32, 32, 3)
    labels = np.array([1, 3, 5, 7], dtype=np.int64)
    tfds_stub = _TFDSStub(_make_dataset(images=images, labels=labels))
    monkeypatch.setattr(tfds_data_mod, "_require_tf", lambda: _FakeTF())
    monkeypatch.setattr(tfds_data_mod, "_require_tfds", lambda: tfds_stub)

    iterator = make_tfds_discrete_image_iterator(
        dataset_name="fake_labeled",
        split="train",
        batch_size=2,
        shuffle=False,
        repeat=False,
        drop_remainder=True,
    )
    batch = next(iterator)

    assert tfds_stub.load_calls == [
        {
            "dataset_name": "fake_labeled",
            "split": "train",
            "data_dir": None,
            "shuffle_files": False,
            "as_supervised": False,
        }
    ]
    assert set(batch.keys()) == {"image", "label"}
    assert batch["image"].shape == (2, 32, 32, 3)
    assert batch["image"].dtype == np.uint8
    assert batch["label"].shape == (2,)
    assert batch["label"].dtype == np.int32
    np.testing.assert_array_equal(batch["image"], images[:2])
    np.testing.assert_array_equal(batch["label"], labels[:2].astype(np.int32))


def test_tfds_loader_supports_unlabeled_examples(monkeypatch):
    images = np.arange(3 * 32 * 32 * 3, dtype=np.uint8).reshape(3, 32, 32, 3)
    tfds_stub = _TFDSStub(_make_dataset(images=images))
    monkeypatch.setattr(tfds_data_mod, "_require_tf", lambda: _FakeTF())
    monkeypatch.setattr(tfds_data_mod, "_require_tfds", lambda: tfds_stub)

    iterator = make_tfds_discrete_image_iterator(
        dataset_name="fake_unlabeled",
        split="train",
        batch_size=2,
        shuffle=False,
        repeat=False,
        drop_remainder=True,
        dummy_label_value=-9,
    )
    batch = next(iterator)

    assert batch["image"].shape == (2, 32, 32, 3)
    assert batch["image"].dtype == np.uint8
    assert batch["label"].shape == (2,)
    assert batch["label"].dtype == np.int32
    np.testing.assert_array_equal(batch["image"], images[:2])
    np.testing.assert_array_equal(batch["label"], np.full((2,), -9, dtype=np.int32))


def test_tfds_loader_caps_shuffle_buffer_for_large_datasets(monkeypatch):
    images = np.zeros((120_000, 4, 4, 3), dtype=np.uint8)
    labels = np.zeros((120_000,), dtype=np.int32)
    dataset = _make_dataset(images=images, labels=labels)
    tfds_stub = _TFDSStub(dataset)
    monkeypatch.setattr(tfds_data_mod, "_require_tf", lambda: _FakeTF())
    monkeypatch.setattr(tfds_data_mod, "_require_tfds", lambda: tfds_stub)

    iterator = make_tfds_discrete_image_iterator(
        dataset_name="fake_large",
        split="train",
        batch_size=4,
        shuffle=True,
        repeat=False,
        drop_remainder=True,
    )
    next(iterator)

    assert dataset.last_shuffle_kwargs is not None
    assert dataset.last_shuffle_kwargs["buffer_size"] == 50_000


def test_tfds_loader_accepts_explicit_shuffle_buffer_size(monkeypatch):
    images = np.zeros((32, 4, 4, 3), dtype=np.uint8)
    labels = np.zeros((32,), dtype=np.int32)
    dataset = _make_dataset(images=images, labels=labels)
    tfds_stub = _TFDSStub(dataset)
    monkeypatch.setattr(tfds_data_mod, "_require_tf", lambda: _FakeTF())
    monkeypatch.setattr(tfds_data_mod, "_require_tfds", lambda: tfds_stub)

    iterator = make_tfds_discrete_image_iterator(
        dataset_name="fake_buffer_override",
        split="train",
        batch_size=4,
        shuffle=True,
        repeat=False,
        drop_remainder=True,
        shuffle_buffer_size=1234,
    )
    next(iterator)

    assert dataset.last_shuffle_kwargs is not None
    assert dataset.last_shuffle_kwargs["buffer_size"] == 1234


def test_tfds_loader_propagates_64x64_shape(monkeypatch):
    images = np.arange(2 * 64 * 64 * 3, dtype=np.uint8).reshape(2, 64, 64, 3)
    labels = np.array([0, 1], dtype=np.int32)
    tfds_stub = _TFDSStub(_make_dataset(images=images, labels=labels))
    monkeypatch.setattr(tfds_data_mod, "_require_tf", lambda: _FakeTF())
    monkeypatch.setattr(tfds_data_mod, "_require_tfds", lambda: tfds_stub)

    iterator = make_tfds_discrete_image_iterator(
        dataset_name="fake_64",
        split="validation",
        batch_size=2,
        shuffle=False,
        repeat=False,
        drop_remainder=True,
    )
    batch = next(iterator)

    assert batch["image"].shape == (2, 64, 64, 3)
    assert batch["label"].shape == (2,)


def test_make_cifar10_iterator_is_compatibility_wrapper(monkeypatch):
    expected_batch = {
        "image": np.zeros((2, 32, 32, 3), dtype=np.uint8),
        "label": np.zeros((2,), dtype=np.int32),
    }
    calls: list[dict[str, object]] = []

    def fake_make_tfds_discrete_image_iterator(**kwargs):
        calls.append(kwargs)
        return iter([expected_batch])

    monkeypatch.setattr(
        cifar10_data_mod,
        "make_tfds_discrete_image_iterator",
        fake_make_tfds_discrete_image_iterator,
    )

    batch = next(
        make_cifar10_iterator(
            split="train",
            batch_size=2,
            seed=11,
            data_dir="/tmp/cifar10",
            shuffle=False,
            repeat=False,
            drop_remainder=False,
            augment=True,
            augment_prob=0.2,
            augment_rotate=False,
            augment_hflip=True,
        )
    )

    assert batch is expected_batch
    assert calls == [
        {
            "dataset_name": "cifar10",
            "split": "train",
            "batch_size": 2,
            "seed": 11,
            "data_dir": "/tmp/cifar10",
            "shuffle": False,
            "repeat": False,
            "drop_remainder": False,
            "augment": True,
            "augment_prob": 0.2,
            "augment_rotate": False,
            "augment_hflip": True,
            "include_label": "auto",
            "dummy_label_value": -1,
        }
    ]


def test_cifar10_tasks_keep_public_names_and_splits(monkeypatch):
    calls: list[dict[str, object]] = []

    def fake_make_tfds_discrete_image_iterator(**kwargs):
        calls.append(kwargs)
        batch_size = int(kwargs["batch_size"])
        height, width, channels = (32, 32, 3)
        return iter(
            [
                {
                    "image": np.zeros((batch_size, height, width, channels), dtype=np.uint8),
                    "label": np.full((batch_size,), -1, dtype=np.int32),
                }
            ]
        )

    monkeypatch.setattr(
        tfds_task_mod,
        "make_tfds_discrete_image_iterator",
        fake_make_tfds_discrete_image_iterator,
    )

    discrete_task = CIFAR10DiscreteTask(
        task_name="md4_cifar10",
        data_dir="/tmp/cifar10",
        batch_size=4,
        eval_batch_size=3,
        augment_enabled=False,
    )
    sjd_task = CIFAR10SJDTask(
        data_dir="/tmp/cifar10",
        batch_size=5,
        eval_batch_size=2,
        data_shape=(32, 32, 3),
        vocab_size=256,
        num_classes=-1,
        beta=lambda t: t,
        augment_enabled=False,
    )

    train_iter, eval_iter = discrete_task.make_dataloaders(seed=13)
    next(train_iter)
    next(eval_iter)

    sjd_train_iter, sjd_eval_iter = sjd_task.make_dataloaders(seed=17)
    next(sjd_train_iter)
    next(sjd_eval_iter)

    assert discrete_task.spec.name == "md4_cifar10"
    assert discrete_task.spec.data_shape == (32, 32, 3)
    assert sjd_task.spec.name == "sjd_cifar10"
    assert sjd_task.spec.data_shape == (32, 32, 3)

    assert calls == [
        {
            "dataset_name": "cifar10",
            "split": "train",
            "batch_size": 4,
            "seed": 13,
            "data_dir": "/tmp/cifar10",
            "shuffle": True,
            "repeat": True,
            "drop_remainder": True,
            "augment": False,
            "augment_prob": 0.15,
            "augment_rotate": True,
            "augment_hflip": True,
            "include_label": "auto",
            "dummy_label_value": -1,
            "shuffle_buffer_size": None,
        },
        {
            "dataset_name": "cifar10",
            "split": "test",
            "batch_size": 3,
            "seed": 14,
            "data_dir": "/tmp/cifar10",
            "shuffle": False,
            "repeat": False,
            "drop_remainder": False,
            "augment": False,
            "augment_prob": 0.15,
            "augment_rotate": True,
            "augment_hflip": True,
            "include_label": "auto",
            "dummy_label_value": -1,
            "shuffle_buffer_size": None,
        },
        {
            "dataset_name": "cifar10",
            "split": "train",
            "batch_size": 5,
            "seed": 17,
            "data_dir": "/tmp/cifar10",
            "shuffle": True,
            "repeat": True,
            "drop_remainder": True,
            "augment": False,
            "augment_prob": 0.15,
            "augment_rotate": True,
            "augment_hflip": True,
            "include_label": "auto",
            "dummy_label_value": -1,
            "shuffle_buffer_size": None,
        },
        {
            "dataset_name": "cifar10",
            "split": "test",
            "batch_size": 2,
            "seed": 18,
            "data_dir": "/tmp/cifar10",
            "shuffle": False,
            "repeat": False,
            "drop_remainder": False,
            "augment": False,
            "augment_prob": 0.15,
            "augment_rotate": True,
            "augment_hflip": True,
            "include_label": "auto",
            "dummy_label_value": -1,
            "shuffle_buffer_size": None,
        },
    ]
