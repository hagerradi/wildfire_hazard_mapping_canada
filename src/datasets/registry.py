import os

import pandas as pd


class Registry:
    """
    Manages a collection of file paths and metadata stored in the split CSV.

    This class handles:
        - loading metadata
        - filtering records based on mask quality (valid_ratio)
        - packages and provides context (dict) for various DataSource objects
    """

    def __init__(self, csv_name: str, root_dir: str, filename_col: str = "filename", valid_mask_threshold: float = 0.01):
        """
        Args:
            csv_name (str): Name of the metadata file.
            root_dir (str): Directory with all the .npy files.
            filename_col (str): Column name in CSV containing the filenames.
            valid_mask_threshold (float): The threshold for how much valid data should be present in a data sample.
        """
        self.root_dir = root_dir
        self.csv_name = csv_name
        self.metadata_path = os.path.join(self.root_dir, self.csv_name)
        self.valid_mask_threshold = valid_mask_threshold
        self.filename_col = filename_col
        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(f"Metadata not found at: {self.metadata_path}")
        metadata_df = pd.read_csv(self.metadata_path)
        # Filter by valid_ratio if the column exists
        if "valid_ratio" in metadata_df.columns:
            metadata_df = metadata_df[metadata_df["valid_ratio"] > self.valid_mask_threshold].copy()
        if filename_col not in metadata_df.columns:
            raise KeyError(f"Column '{filename_col}' not found in {csv_name}")
        self.metadata = metadata_df
        self.records = self.metadata.to_dict("records")  # Convert to list of dicts for O(1) access performance

    def get_context(self, idx: int):
        """
        Retrieves metadata and the absolute file path for a specific record.

        Args:
            idx (int): The index of the record in the filtered metadata.

        Returns:
            context (dict): A dictionary containing all CSV columns for the item, plus
            'idx' and the absolute 'file_path'.
        """
        row = self.records[idx]
        context = row.copy()
        context["idx"] = idx
        context["file_path"] = os.path.join(self.root_dir, row[self.filename_col])
        return context

    def __len__(self):
        return len(self.metadata)


if __name__ == "__main__":
    registry = Registry(
        csv_name="train_indices.csv",
        root_dir="yan_bp3/data_samples_approach_1/",
    )
    context = registry.get_context(0)
    print(context)
