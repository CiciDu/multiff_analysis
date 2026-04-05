import sys
from pathlib import Path
import os


for p in [Path.cwd()] + list(Path.cwd().parents):
    if p.name == "Multifirefly-Project":
        os.chdir(p)
        sys.path.insert(0, str(p / "multiff_analysis/multiff_code/methods"))
        sys.path.insert(0, str(p / 'multiff_analysis'))
        break


from neural_data_analysis.neural_analysis_tools.encoding_tools.encoding_helpers import (
    encoding_design_utils,
)
from jobs.encoding.shared_encoding_script import run_encoding_main


def main():
    encoding_design_utils.bootstrap_repo_path()

    from neural_data_analysis.neural_analysis_tools.encoding_tools.encoding_by_topics import (
        encode_pn_pipeline,
    )

    return run_encoding_main(
        lambda raw_data_folder_path, bin_width, **runner_kwargs: encode_pn_pipeline.PNEncodingRunner(
            raw_data_folder_path=raw_data_folder_path,
            bin_width=bin_width,
            **runner_kwargs,
        )
    )


if __name__ == "__main__":
    main()
