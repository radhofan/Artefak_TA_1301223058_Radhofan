import pandas as pd
import numpy as np
import argparse
import os
from aif360.datasets import StandardDataset
from aif360.algorithms.preprocessing import DisparateImpactRemover


def aif360_fairness_preprocessing(input_csv, output_csv, protected_attribute='gender_binary', 
                                   label_name='rating', threshold=4, repair_level=1.0):
    """
    Apply AIF360 Disparate Impact Remover to mitigate bias.
    This modifies feature values to be fair WITHOUT adding or removing columns.
    
    Args:
        input_csv: Path to input CSV file
        output_csv: Path to output CSV file (can be same as input)
        protected_attribute: Column name for protected attribute (default: 'gender_binary')
        label_name: Column name for the label (default: 'rating')
        threshold: Threshold for binary labels (default: 4)
        repair_level: Repair level (0.0 = no repair, 1.0 = full repair)
    """
    
    print("=" * 80)
    print("AIF360 Disparate Impact Remover Preprocessing")
    print("=" * 80)
    
    # Load the data
    df = pd.read_csv(input_csv)
    print(f"Loaded data from: {input_csv}")
    print(f"Total samples: {len(df)}")
    print(f"Original columns: {list(df.columns)}")
    
    # Store original columns and dtypes
    original_columns = df.columns.tolist()
    original_dtypes = df.dtypes.to_dict()
    
    # Store non-numeric columns to restore later
    non_numeric_cols = {}
    for col in df.columns:
        if df[col].dtype == 'object' or col == 'gender':
            non_numeric_cols[col] = df[col].copy()
    
    # Create binary label if needed
    had_binary_label = 'binary_label' in df.columns
    if not had_binary_label:
        df['binary_label'] = (df[label_name] >= threshold).astype(int)
    
    # Prepare numeric-only dataframe for AIF360
    dataset_df = df.copy()
    
    # Drop non-numeric columns except protected attribute and label
    cols_to_keep = [col for col in dataset_df.columns 
                    if col in [protected_attribute, 'binary_label'] or 
                    dataset_df[col].dtype in ['int64', 'float64']]
    
    dataset_df = dataset_df[cols_to_keep]
    
    print(f"\nNumeric columns for AIF360: {list(dataset_df.columns)}")
    
    # Create AIF360 StandardDataset
    aif_dataset = StandardDataset(
        dataset_df,
        label_name='binary_label',
        favorable_classes=[1],
        protected_attribute_names=[protected_attribute],
        privileged_classes=[[1]]
    )
    
    print(f"\nProtected attribute: {protected_attribute}")
    print(f"Label: binary_label (threshold={threshold})")
    print(f"Repair level: {repair_level}")
    
    # Apply Disparate Impact Remover
    DIR = DisparateImpactRemover(repair_level=repair_level, sensitive_attribute=protected_attribute)
    
    dataset_transf = DIR.fit_transform(aif_dataset)
    
    print("\nAfter Disparate Impact Removal:")
    print(f"  Dataset size: {len(dataset_transf.labels)} (unchanged)")
    
    # Convert back to dataframe
    df_transf = dataset_transf.convert_to_dataframe()[0]
    
    # Restore non-numeric columns
    for col, values in non_numeric_cols.items():
        df_transf[col] = values.values
    
    # Remove binary_label if it wasn't in original
    if not had_binary_label and 'binary_label' in df_transf.columns:
        df_transf = df_transf.drop(columns=['binary_label'])
    
    # Ensure column order matches original
    df_transf = df_transf[original_columns]
    
    # Restore original dtypes where possible
    for col in df_transf.columns:
        if col in original_dtypes and col not in non_numeric_cols:
            try:
                df_transf[col] = df_transf[col].astype(original_dtypes[col])
            except:
                pass
    
    print(f"\nFinal columns: {list(df_transf.columns)}")
    print(f"Columns match original: {list(df_transf.columns) == original_columns}")
    
    # Save the transformed data
    df_transf.to_csv(output_csv, index=False)
    print(f"\nFair data saved to: {output_csv}")
    print("=" * 80)
    
    return df_transf


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Apply AIF360 Disparate Impact Remover')
    parser.add_argument('--dataset', type=str, default='ml-100k',
                        help='Dataset name (e.g., ml-100k, ml-200k)')
    parser.add_argument('--protected_attr', type=str, default='gender_binary',
                        help='Protected attribute column name')
    parser.add_argument('--label', type=str, default='rating',
                        help='Label column name')
    parser.add_argument('--threshold', type=int, default=4,
                        help='Threshold for binary labels')
    parser.add_argument('--repair_level', type=float, default=1.0,
                        help='Repair level (0.0-1.0, higher = more fairness correction)')
    
    args = parser.parse_args()
    
    # Construct file paths
    input_csv = f'data/{args.dataset}/{args.dataset}.csv'
    output_csv = f'data/{args.dataset}/{args.dataset}-fair.csv'  # Same file (overwrite)
    
    print(f"\n### AIF360 Fairness Preprocessing for {args.dataset} ###\n")
    
    if not os.path.exists(input_csv):
        print(f"Error: Input file not found: {input_csv}")
        print("Please run the data loading script first.")
        exit(1)
    
    # Apply fairness preprocessing
    fair_data = aif360_fairness_preprocessing(
        input_csv=input_csv,
        output_csv=output_csv,
        protected_attribute=args.protected_attr,
        label_name=args.label,
        threshold=args.threshold,
        repair_level=args.repair_level
    )
    
    print("\n### Preprocessing Complete ###\n")