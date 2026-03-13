import os
import sys
from pathlib import Path

# -------------------------------------------------------
# Repo path bootstrap
# -------------------------------------------------------
for p in [Path.cwd()] + list(Path.cwd().parents):
    if p.name == 'Multifirefly-Project':
        os.chdir(p)
        sys.path.insert(0, str(p / 'multiff_analysis/multiff_code/methods'))
        sys.path.insert(0, str(p / 'multiff_analysis'))
        break

from neural_data_analysis.neural_analysis_tools.decoding_tools.decoding_by_topics import (
    decode_pn_pipeline,
)
from jobs.decoding.shared_decoding_script import run_decoding_main


def main():
    return run_decoding_main(
        decode_pn_pipeline.PNDecodingRunner,
        run_kwargs={'cv_decoding_verbosity': 2},
    )


if __name__ == '__main__':
    main()
