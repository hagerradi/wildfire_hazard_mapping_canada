from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


def run_r_script(
    r_script_path: str | Path,
    output_dir: str | Path,
) -> None:
    """
    Run the R script and pass the output directory as its first argument.

    Equivalent command:
        Rscript compute_vector_values_national.R <output_dir>
    """
    r_script_path = Path(r_script_path).resolve()
    output_dir = Path(output_dir).resolve()

    if not r_script_path.is_file():
        raise FileNotFoundError(f"R script not found: {r_script_path}")

    rscript_executable = shutil.which("Rscript")

    if rscript_executable is None:
        raise RuntimeError("Rscript was not found on PATH. " "Make sure R is installed and Rscript is accessible.")

    output_dir.mkdir(parents=True, exist_ok=True)

    command = [
        rscript_executable,
        str(r_script_path),
        str(output_dir),
    ]

    print("Running command:")
    print(" ".join(command))

    try:
        subprocess.run(
            command,
            check=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"R script failed with exit code {exc.returncode}") from exc

    print(f"R script completed successfully.")
    print(f"Outputs saved under: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the FBP curve generation R script.")

    parser.add_argument(
        "--r-script",
        type=Path,
        default=Path("data_preparation/tabular/fuel_features/" "compute_vector_values_national.R"),
        help="Path to the R script.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Folder where the R script should save its outputs.",
    )

    args = parser.parse_args()

    run_r_script(
        r_script_path=args.r_script,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
