import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from data_preparation.generate_season_cause_output import FireCountRasterizer
from data_preparation.grid_loader.utils import NODATA, load_fire_shapefiles
from data_preparation.hexel_loader import load_features_per_hexel
from data_preparation.utils import find_hex_ids


def save_split_hexel_windows(
    valid_window: np.ndarray, out_dir: str, win_id: int, season: str, cause: str, hex_id: str, format: str = "npy"
) -> str:
    """Save a hexel window"""
    filename = f"numpy_files/hex_{hex_id}_{str(win_id)}_{season}_{cause}.{format}"
    np.save(os.path.join(out_dir, filename), valid_window) if format == "npy" else np.savez_compressed(
        os.path.join(out_dir, filename), arr=valid_window
    )
    return filename


def get_split_hexel_window(
    season_cause_stacked_feats: np.ndarray,
    season_cause_mask: np.ndarray,
    season_cause_mapping: dict | None,
    out_dir: str,
    root_dir: str,
    hex_id: str,
    win_h: int = 128,
    win_w: int = 128,
    overlap_ratio: float = 0.2,
):
    """
    Split the hexel using sliding windows for inp to the model
    Inputs:
        season_cause_stacked_feats(np.ndarray): Stacked array of all the features of the shape (num_season_cause, H, W, num_feats)
        season_cause_mask(np.ndarray): A bool array where True means to ignore the pixel (num_season_cause, H,W)
    """
    print("============Splitting the hexel==================")
    shp_paths = load_fire_shapefiles(os.path.join(root_dir, "hex" + str(hex_id)))
    rasterizer = FireCountRasterizer(shp_paths, None)
    total_unique_iters = rasterizer.get_num_unique_iters(season=None, cause=None)
    num_season_cause, H, W, _ = season_cause_stacked_feats.shape
    stride_h = max(1, int(win_h * (1 - overlap_ratio)))  # n_rows = (H-win_h)//stride_h + 1
    stride_w = max(1, int(win_w * (1 - overlap_ratio)))

    # Adding padding for the edges
    pad_h = stride_h - (H - win_h) % stride_h if (H - win_h) % stride_h != 0 else 0
    pad_w = stride_w - (W - win_w) % stride_w if (W - win_w) % stride_w != 0 else 0
    season_cause_stacked_feats_padded = np.pad(
        season_cause_stacked_feats, ((0, 0), (0, pad_h), (0, pad_w), (0, 0)), mode="constant", constant_values=NODATA
    )
    season_cause_mask_padded = np.pad(
        season_cause_mask, ((0, 0), (0, pad_h), (0, pad_w)), mode="constant", constant_values=1.0
    )  # Mask should be 1 where nan
    # season_cause_stacked_feats_padded[:,:,:,-1][season_cause_mask_padded] = 0.0 #For outputs, mask means 0 probability
    _, H_pad, W_pad, _ = season_cause_stacked_feats_padded.shape

    window_area = win_h * win_w
    num_total_windows, num_valid_windows = 0.0, 0.0
    valid_coords = []
    for i in range(num_season_cause):
        stacked_feats = season_cause_stacked_feats_padded[i]
        mask = season_cause_mask_padded[i]
        if season_cause_mapping is None:
            season, cause = "all", "all"
            season_cause_unique_iters = None
        else:
            season, cause = season_cause_mapping[i]
            season_cause_unique_iters = rasterizer.get_num_unique_iters(season=season, cause=cause)
        for row in range(0, H_pad - win_h + 1, stride_h):
            for col in range(0, W_pad - win_w + 1, stride_w):
                num_total_windows += 1
                # Extract the mask patch
                mask_window = mask[row : row + win_h, col : col + win_w]
                # Count True values in the mask
                true_count = np.count_nonzero(~mask_window)
                valid_ratio = true_count / window_area

                num_valid_windows += 1
                window_data = stacked_feats[row : row + win_h, col : col + win_w, :]
                filename = save_split_hexel_windows(window_data, out_dir, int(num_valid_windows), season, cause, hex_id)
                valid_coords.append(
                    [
                        str(Path(filename)),
                        season,
                        cause,
                        hex_id,
                        num_valid_windows,
                        row,
                        col,
                        valid_ratio,
                        total_unique_iters,
                        season_cause_unique_iters,
                    ]
                )
    df_coords = pd.DataFrame(valid_coords)
    df_coords.columns = [
        "filename",
        "season",
        "cause",
        "hex_id",
        "window_id",
        "row",
        "col",
        "valid_ratio",
        "total_unique_iters",
        "season_cause_unique_iters",
    ]
    df_coords.to_csv(os.path.join(out_dir, f"meta_hex_{hex_id}.csv"), index=False)
    print(f"=====Hexel data Saved at {out_dir} ========")


def generate_data_samples(
    root_dir: str, modelling_approach: int, output_type: str = "count", win_h: int = 128, win_w: int = 128, overlap_ratio: float = 0.2
):
    out_dir = os.path.join(root_dir, f"data_samples_approach_{modelling_approach}")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "numpy_files"), exist_ok=True)
    hex_ids = find_hex_ids(root_dir)
    for hex_id in hex_ids:
        if hex_id == "52":
            print("======Skipping hex=======", hex_id)
            continue
        print(f"======Working on Hex ID: {hex_id}==========")
        stacked_feats, mask, season_cause_mapping = load_features_per_hexel(
            root_dir=root_dir,
            hex_id=hex_id,
            feature_channel_map_path=os.path.join(out_dir, f"feature_channel_map_{modelling_approach}.json"),
            modelling_approach=modelling_approach,
            output_type=output_type,
        )
        get_split_hexel_window(
            season_cause_stacked_feats=stacked_feats,
            season_cause_mask=mask,
            season_cause_mapping=season_cause_mapping,
            out_dir=out_dir,
            root_dir=root_dir,
            hex_id=hex_id,
            win_h=win_h,
            win_w=win_w,
            overlap_ratio=overlap_ratio,
        )
        print(f"======Processed Hex ID: {hex_id}==========")


def main():
    parser = argparse.ArgumentParser(description="Generate data samples from each hexel")

    parser.add_argument("--root_dir", type=str, help="data root directory", required=True)
    parser.add_argument("--modelling_approach", type=int, help="Either 1 or 2", default=2)
    parser.add_argument("--output_type", type=str, help="output type as prob or count", default="count")
    parser.add_argument("--win_h", type=int, help="Height of the window", default=128)
    parser.add_argument("--win_w", type=int, help="Height of the window", default=128)
    parser.add_argument("--overlap_ratio", type=float, help="Overlap ratio between windows", default=0.2)

    args = parser.parse_args()

    generate_data_samples(
        root_dir=args.root_dir,
        modelling_approach=args.modelling_approach,
        win_h=args.win_h,
        win_w=args.win_w,
        overlap_ratio=args.overlap_ratio,
        output_type=args.output_type,
    )


if __name__ == "__main__":
    main()
