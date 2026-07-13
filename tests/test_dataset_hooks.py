from pathlib import Path

import numpy as np
import pandas as pd

from src.datasets.dataset import MultiSourceDataset
from src.datasets.sources.base import DataSource


class ArraySource(DataSource):
    @property
    def input_dim(self) -> int:
        return 1

    def get_sample(self, context: dict) -> np.ndarray:
        return np.asarray(context["data"])


def test_dataset_applies_metadata_filter_and_patch_transform(tmp_path: Path) -> None:
    np.save(tmp_path / "keep.npy", np.array([[1]], dtype=np.float32))
    np.save(tmp_path / "drop.npy", np.array([[2]], dtype=np.float32))
    pd.DataFrame(
        [
            {"filename": "keep.npy", "hex_id": 16},
            {"filename": "drop.npy", "hex_id": 17},
        ]
    ).to_csv(tmp_path / "test.csv", index=False)

    dataset = MultiSourceDataset(
        csv_name="test.csv",
        root_dir=str(tmp_path),
        sources={"array": ArraySource()},
        metadata_filter=lambda metadata: metadata.loc[metadata["hex_id"] == 16],
        patch_transform=lambda data, _patch_info: np.asarray(data) + 3,
    )

    assert len(dataset) == 1
    assert dataset[0]["array"].item() == 4
