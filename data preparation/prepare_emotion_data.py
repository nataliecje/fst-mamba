"""Script to prepare EMOTION data for FunctionalMamba.
Combines individual subject FC matrices and creates labels file.
Resizes 264x264 to 128x128 for model compatibility.
Supports both LR and RL phase encoding directions.
"""

import numpy as np
import pandas as pd
import os
import argparse
from glob import glob
from tqdm import tqdm
from scipy.ndimage import zoom

def prepare_emotion_data(phase_encoding='LR'):
    # Target size: 128 (compatible: 128/patch_size(2)=64, 64%window_size(4)=0)
    target_size = 128
    pe = phase_encoding.upper()  # LR or RL
    pe_lower = phase_encoding.lower()
    
    print(f"\nPreparing EMOTION_{pe} data...")
    
    # Paths
    data_dir = f'dataset/EMOTION_{pe}'
    label_csv = 'dataset/hcp_fluid_cognition_label.csv'
    
    # Output paths
    output_data = f'dataset/EMOTION_{pe}/emotion_{pe_lower}_fc_combined.npy'
    output_labels = f'dataset/EMOTION_{pe}/emotion_{pe_lower}_labels.npy'
    
    # Load labels CSV
    df = pd.read_csv(label_csv)
    print(f"Labels CSV has {len(df)} subjects")
    
    # Get all FC files (non-timeseries, 264x264 matrices)
    fc_files = sorted(glob(os.path.join(data_dir, f'Power_EMOTION_{pe}_*.npy')))
    fc_files = [f for f in fc_files if '_TS.npy' not in f]  # Exclude timeseries files
    print(f"Found {len(fc_files)} FC matrix files")
    
    # Extract subject IDs from filenames
    subject_ids = []
    for f in fc_files:
        # Extract ID from filename like Power_EMOTION_LR_100206.npy or Power_EMOTION_RL_100206.npy
        basename = os.path.basename(f)
        subject_id = int(basename.replace(f'Power_EMOTION_{pe}_', '').replace('.npy', ''))
        subject_ids.append(subject_id)
    
    # Match with labels
    df_matched = df[df['Subject'].isin(subject_ids)].copy()
    print(f"Matched {len(df_matched)} subjects with labels")
    
    # Create subject to index mapping from matched dataframe
    subject_to_idx = {row['Subject']: idx for idx, row in df_matched.iterrows()}
    
    # Prepare data and labels
    data_list = []
    labels_list = []
    matched_subjects = []
    
    for subject_id in tqdm(subject_ids, desc="Loading FC matrices (264->128)"):
        if subject_id not in df_matched['Subject'].values:
            continue
        
        # Load FC matrix and resize from 264x264 to 128x128
        fc_file = os.path.join(data_dir, f'Power_EMOTION_{pe}_{subject_id}.npy')
        fc_matrix = np.load(fc_file)  # Shape: (264, 264)
        
        # Resize to 128x128 using bilinear interpolation
        zoom_factor = target_size / fc_matrix.shape[0]
        fc_matrix = zoom(fc_matrix, zoom_factor, order=1)
        
        # Get labels for this subject
        row = df_matched[df_matched['Subject'] == subject_id].iloc[0]
        
        # Get fluid cognition - SKIP subjects with missing values
        fluid_cog = row['CogFluidComp_AgeAdj']
        if pd.isna(fluid_cog) or fluid_cog == 0:
            continue  # Skip subjects with missing fluid cognition
        
        # Encode labels: sex (0=F, 1=M), age (encoded as midpoint), fluid cognition
        sex = 1 if row['Gender'] == 'M' else 0
        
        # Convert age range to numeric (use midpoint)
        age_map = {'22-25': 23.5, '26-30': 28, '31-35': 33, '36+': 38}
        age = age_map.get(row['Age'], 30)  # default 30 if unknown
        
        data_list.append(fc_matrix)
        labels_list.append([sex, age, fluid_cog])
        matched_subjects.append(subject_id)
    
    # Stack into arrays
    # Data shape: (N_subjects, 1, 64, 64) - adding time dimension of 1 for static FC
    data_array = np.stack(data_list, axis=0)[:, np.newaxis, :, :]
    labels_array = np.array(labels_list)
    
    print(f"\nFinal data shape: {data_array.shape}")  # (N, 1, 128, 128)
    print(f"Final labels shape: {labels_array.shape}")  # (N, 3) -> [sex, age, fluid_cog]
    print(f"Sex distribution: Male={int(np.sum(labels_array[:, 0]))}, Female={len(labels_array) - int(np.sum(labels_array[:, 0]))}")
    print(f"Fluid cognition range: [{labels_array[:, 2].min():.1f}, {labels_array[:, 2].max():.1f}]")
    print(f"Subjects with valid fluid cognition: {len(labels_array)}")
    
    # Save
    np.save(output_data, data_array.astype(np.float32))
    np.save(output_labels, labels_array.astype(np.float32))
    
    print(f"\nSaved combined data to: {output_data}")
    print(f"Saved labels to: {output_labels}")
    print(f"\nLabels format: [sex (0=F, 1=M), age, fluid_cognition]")
    
    return data_array, labels_array

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare EMOTION task data')
    parser.add_argument('--phase', type=str, default='LR', choices=['LR', 'RL'],
                        help='Phase encoding direction: LR or RL (default: LR)')
    args = parser.parse_args()
    prepare_emotion_data(phase_encoding=args.phase)
