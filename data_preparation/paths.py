from pathlib import Path


class Paths:
    def __init__(self, hex_id: str, root_dir: str | Path = ".") -> None:
        self.base_dir = Path(root_dir) / f"hex{hex_id}"
        self.spatial_dir = self.base_dir / "spatial"
        self.tabular_dir = self.base_dir / "tabular"
        self.hex_id = hex_id

    def ignition_prob_dir(self) -> Path:
        return self.spatial_dir / "ignition_grids"

    def mask_grid_actual(self, hex_id: int | str) -> Path:
        return self.spatial_dir / "mask_grids" / f"hex{hex_id}_actual.shp"

    def mask_grid_buffer(self, hex_id: int | str) -> Path:
        return self.spatial_dir / "mask_grids" / f"hex{hex_id}_buffer.shp"

    def fire_zone_grid(self, hex_id: int | str) -> Path:
        return self.spatial_dir / f"hex{hex_id}_firezones.tif"

    def fuel_grid(self, hex_id: int | str) -> Path:
        return self.spatial_dir / f"hex{hex_id}_fbp.tif"

    def elevation_grid(self, hex_id: int | str) -> Path:
        return self.spatial_dir / f"hex{hex_id}_dem.tif"

    def fuel_table(self, hex_id: int | str | None = None) -> Path:
        if not hex_id:
            hex_id = self.hex_id
        return self.tabular_dir / f"hex{hex_id}_FuelTypes.csv"

    def weather_table(self, hex_id: int | str) -> Path:
        return self.tabular_dir / f"hex{hex_id}_DailyWeather.csv"

    def output_burn_prob(self) -> Path:
        return self.base_dir / "results" / "burnP3Plus_OutputBurnProbability" / "burnProbability-sn2.tif"

    def output_fire_intensity(self) -> Path:
        return self.base_dir / "results" / "burnP3Plus_OutputFireIntensitySummaryMap" / "fbpSummary-FireIntensity-Average.tif"

    def output_ros(self) -> Path:
        return self.base_dir / "results" / "burnP3Plus_OutputRateOfSpreadSummaryMap" / "fbpSummary-RateOfSpread-Average.tif"
