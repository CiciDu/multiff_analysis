#!/usr/bin/env python

import argparse
import time
import os

import numpy as np

from neural_data_analysis.neural_analysis_tools.neural_axes import cebra_analyzer
from neural_data_analysis.loaders import load_processed_neural_data  # adjust if needed


def main():
    parser = argparse.ArgumentParser(description='Run CEBRA embedding')
    parser.add_argument('--session_id', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--output_dim', type=int, default=3)
    parser.add_argument('--max_iterations', type=int, default=8000)
    parser.add_argument('--bin_width_ms', type=float, default=10.0)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f'Loading data for session {args.session_id}...')
    pn, stop_label_df, columns = load_processed_neural_data(
        session_id=args.session_id
    )

    analyzer = cebra_analyzer.CEBRAAnalyzer(
        spikes_df=pn.spikes_df,
        behavior_df=stop_label_df,
        event_col='time',
        behavior_cols=columns,
        bin_width_ms=args.bin_width_ms,
    )

    print('Starting CEBRA fit...')
    t0 = time.time()

    embedding, model = analyzer.fit_cebra(
        output_dim=args.output_dim,
        conditional='behavior',
        max_iterations=args.max_iterations,
        device=args.device,
        verbose=True,
    )

    elapsed = time.time() - t0
    print(f'CEBRA finished in {elapsed / 60:.2f} minutes')
    print(f'Iterations used: {getattr(analyzer, "n_iterations_", None)}')

    out_path = os.path.join(
        args.output_dir,
        f'cebra_embedding_{args.session_id}'
    )

    analyzer.save_embedding(out_path)

    print(f'Saved embedding to {out_path}.npz')


if __name__ == '__main__':
    main()
