 class DataIndex:
    """TODO add meaningful description"""
    def __init__(
        self, 
        csv_name: str, 
        root_dir: str,
        filename_col: str = "filename",
        valid_mask_threshold: float = 0.01
        ):
        self.root_dir = root_dir
        self.metadata = pd.read_csv(os.path.join(self.root_dir, csv_name))
        self.valid_mask_threshold = valid_mask_threshold
        self.filename_col = filename_col
        if "valid_ratio" in self.metadata.columns:
            self.metadata = self.metadata[self.metadata["valid_ratio"] > self.valid_mask_threshold]
        self.records = self.metadata.to_dict('records')
        
    
    def get_context(self, idx: int):
        """
        TODO: add meaningful comment
        """
        row = self.records[idx]
        context = row.copy()
        context['file_path'] = os.path.join(self.root_dir, row[self.filename_col])
        return context

    def __len__(self):
        return len(self.metadata)

indices = DataIndex(
    csv_name = "train_indices.csv",
    root_dir="/home/mila/o/oseaj/scratch/nrcan_data/data_samples_approach_1_weather"
    )
