from pathlib import Path
import os
import sys

# -------------------------
# Infer project root
# -------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[3]

if not (PROJECT_ROOT / 'multiff_analysis').exists():
    raise RuntimeError(
        f'Could not infer project root from {__file__}'
    )

# -------------------------
# Set working directory
# -------------------------
os.chdir(PROJECT_ROOT)
print(f'[INFO] Changed working directory → {PROJECT_ROOT}')


OUT_DIR = PROJECT_ROOT / 'multiff_analysis' / 'jobs' / 'data_configs'
OUT_DIR.mkdir(parents=True, exist_ok=True)

RAW_DATA_DIR_NAME = 'all_monkey_data/raw_monkey_data'
MONKEY_NAMES = ['monkey_Bruno', 'monkey_Schro']


# -------------------------
# Import canonical utils
# -------------------------
sys.path.insert(
    0,
    str(PROJECT_ROOT / 'multiff_analysis' / 'multiff_code' / 'methods'),
)

from data_wrangling import combine_info_utils


# -------------------------
# Helpers
# -------------------------
def generate_raw_data_paths_for_monkey(
    project_root,
    raw_data_dir_name,
    monkey_name,
    out_file,
):
    sessions_df = combine_info_utils.make_sessions_df_for_one_monkey(
        raw_data_dir_name,
        monkey_name,
    )

    paths = []
    for _, row in sessions_df.iterrows():
        path = (
            project_root
            / raw_data_dir_name
            / row['monkey_name']
            / row['data_name']
        )
        paths.append(path.resolve())

    out_file.write_text('\n'.join(str(p) for p in paths) + '\n')

    print(f'[OK] {monkey_name}: wrote {len(paths)} paths → {out_file}')


# -------------------------
# Main
# -------------------------
def main():
    for monkey_name in MONKEY_NAMES:
        out_file = OUT_DIR / f'raw_data_paths_{monkey_name}.txt'

        generate_raw_data_paths_for_monkey(
            PROJECT_ROOT,
            RAW_DATA_DIR_NAME,
            monkey_name,
            out_file,
        )


if __name__ == '__main__':
    main()
