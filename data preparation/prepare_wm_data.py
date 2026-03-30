"""Script to prepare WM (Working Memory) data for FunctionalMamba.
Combines individual subject raw fMRI timeseries and creates labels file.
Uses raw timeseries data (_TS.npy) for capturing dynamic brain connectivity.
Supports both LR and RL phase encoding directions.
"""

import numpy as np
import pandas as pd
import os
import argparse
from glob import glob
from tqdm import tqdm
from scipy.ndimage import zoom

def prepare_wm_data(phase_encoding='LR'):
    # Use functional connectivity (FC) data instead of raw timeseries
    pe = phase_encoding.upper()  # LR or RL
    pe_lower = phase_encoding.lower()
    
    print(f"\nPreparing WM_{pe} data from functional connectivity matrices...")
    
    # Paths
    data_dir = f'dataset/WM_{pe}'
    label_csv = 'dataset/hcp_fluid_cognition_label.csv'
    
    # Output paths
    output_data = f'dataset/WM_{pe}/wm_{pe_lower}_fc_resized.npy'
    output_labels = f'dataset/WM_{pe}/wm_{pe_lower}_labels.npy'
    
    # Load labels CSV
    df = pd.read_csv(label_csv)
    print(f"Labels CSV has {len(df)} subjects")
    
    # Get all FC files (.npy)
    fc_files = sorted(glob(os.path.join(data_dir, f'Power_WM_{pe}_*.npy')))
    # Filter out time series files (_TS), keep only FC files
    fc_files = [f for f in fc_files if not f.endswith('_TS.npy')]
    print(f"Found {len(fc_files)} FC files")
    
    # Extract subject IDs from filenames
    subject_ids = []
    for f in fc_files:
        # Extract ID from filename like Power_WM_LR_100206.npy
        basename = os.path.basename(f)
        subject_id = int(basename.replace(f'Power_WM_{pe}_', '').replace('.npy', ''))
        subject_ids.append(subject_id)
    
    # Match with labels
    df_matched = df[df['Subject'].isin(subject_ids)].copy()
    print(f"Matched {len(df_matched)} subjects with labels")
    
    # Prepare data and labels
    data_list = []
    labels_list = []

    for subject_id in tqdm(subject_ids, desc="Loading FC data"):
        if subject_id not in df_matched['Subject'].values:
            continue

        # Get labels for this subject first
        row = df_matched[df_matched['Subject'] == subject_id].iloc[0]

        # Get fluid cognition - SKIP subjects with missing values
        fluid_cog = row['CogFluidComp_AgeAdj']
        if pd.isna(fluid_cog) or fluid_cog == 0:
            continue  # Skip subjects with missing fluid cognition

        # Load FC data
        fc_file = os.path.join(data_dir, f'Power_WM_{pe}_{subject_id}.npy')
        fc_data = np.load(fc_file)  # Shape: (264, 264)

        # Resize FC matrix to 128x128
        zoom_factor = 128 / fc_data.shape[0]
        resized_fc = zoom(fc_data, (zoom_factor, zoom_factor), order=1)  # (128, 128)

        data_list.append(resized_fc)

        # Encode labels: sex (0=F, 1=M), age (encoded as midpoint), fluid cognition
        sex = 1 if row['Gender'] == 'M' else 0

        # Convert age range to numeric (use midpoint)
        age_map = {'22-25': 23.5, '26-30': 28, '31-35': 33, '36+': 38}
        age = age_map.get(row['Age'], 30)  # default 30 if unknown

        labels_list.append([sex, age, fluid_cog])

    # Convert to numpy arrays
    data_array = np.array(data_list)  # (N, 128, 128)
    data_array = data_array[:, np.newaxis, :, :]  # (N, 1, 128, 128)
    labels_array = np.array(labels_list)

    print(f"\nFinal data shape: {data_array.shape}")
    print(f"Final labels shape: {labels_array.shape}")
    print(f"Labels: [sex, age, fluid_cognition]")

    # Save
    np.save(output_data, data_array)
    np.save(output_labels, labels_array)

    print(f"\nSaved resized FC data to: {output_data}")
    print(f"Saved labels to: {output_labels}")

    # Print statistics
    print(f"\nFluid Cognition Statistics:")
    fluid_scores = labels_array[:, 2]
    print(f"  Mean: {fluid_scores.mean():.2f}")
    print(f"  Std: {fluid_scores.std():.2f}")
    print(f"  Min: {fluid_scores.min():.2f}")
    print(f"  Max: {fluid_scores.max():.2f}")

    print(f"\nFC Data Statistics:")
    print(f"  Mean: {data_array.mean():.4f}")
    print(f"  Std: {data_array.std():.4f}")
    print(f"  Min: {data_array.min():.4f}")
    print(f"  Max: {data_array.max():.4f}")

    return data_array, labels_array

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare WM (Working Memory) task data')
    parser.add_argument('--phase', type=str, default='LR', choices=['LR', 'RL'],
                        help='Phase encoding direction: LR or RL (default: LR)')
    args = parser.parse_args()
    prepare_wm_data(phase_encoding=args.phase)
