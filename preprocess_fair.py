import pandas as pd
import numpy as np
import argparse
import os
from aif360.datasets import StandardDataset
from aif360.algorithms.preprocessing import DisparateImpactRemover
import shutil
import random

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

def aif360_fairness_preprocessing(input_csv, output_csv, protected_attribute='gender_binary', 
                                   label_name='rating', threshold=4, repair_level=1.0):
    
    print("=" * 80)
    print("AIF360 Disparate Impact Remover Preprocessing")
    print("=" * 80)
    
    df = pd.read_csv(input_csv)
    print(f"Loaded data from: {input_csv}")
    print(f"Total samples: {len(df)}")
    print(f"Original columns: {list(df.columns)}")
    
    original_columns = df.columns.tolist()
    original_dtypes = df.dtypes.to_dict()
    
    id_columns = {}
    id_columns['user_id'] = df['user_id'].copy()
    id_columns['item_id'] = df['item_id'].copy()
    
    non_numeric_cols = {}
    for col in df.columns:
        if df[col].dtype == 'object' or col == 'gender':
            non_numeric_cols[col] = df[col].copy()
    
    had_binary_label = 'binary_label' in df.columns
    if not had_binary_label:
        df['binary_label'] = (df[label_name] >= threshold).astype(int)
    
    dataset_df = df.copy()
    
    cols_to_keep = [col for col in dataset_df.columns 
                    if col in [protected_attribute, 'binary_label'] or 
                    (dataset_df[col].dtype in ['int64', 'float64'] and col not in ['user_id', 'item_id'])]
    
    dataset_df = dataset_df[cols_to_keep]
    
    print(f"\nNumeric columns for AIF360 (excluding user_id, item_id): {list(dataset_df.columns)}")
    
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
    
    DIR = DisparateImpactRemover(repair_level=repair_level, sensitive_attribute=protected_attribute)
    
    dataset_transf = DIR.fit_transform(aif_dataset)
    
    print("\nAfter Disparate Impact Removal:")
    print(f"  Dataset size: {len(dataset_transf.labels)} (unchanged)")
    
    df_transf = dataset_transf.convert_to_dataframe()[0]
    
    df_transf['user_id'] = id_columns['user_id'].values
    df_transf['item_id'] = id_columns['item_id'].values
    
    for col, values in non_numeric_cols.items():
        df_transf[col] = values.values
    
    if not had_binary_label and 'binary_label' in df_transf.columns:
        df_transf = df_transf.drop(columns=['binary_label'])
    
    df_transf = df_transf[original_columns]
    
    for col in df_transf.columns:
        if col in original_dtypes and col not in non_numeric_cols and col not in ['user_id', 'item_id']:
            try:
                df_transf[col] = df_transf[col].astype(original_dtypes[col])
            except:
                pass
    
    print(f"\nFinal columns: {list(df_transf.columns)}")
    print(f"Columns match original: {list(df_transf.columns) == original_columns}")
    print(f"user_id range: [{df_transf['user_id'].min()}, {df_transf['user_id'].max()}]")
    print(f"item_id range: [{df_transf['item_id'].min()}, {df_transf['item_id'].max()}]")
    
    df_transf.to_csv(output_csv, index=False)
    print(f"\nFair data saved to: {output_csv}")
    print("=" * 80)
    
    return df_transf

def no_fairness_preprocessing(input_csv, output_csv):
    
    print("=" * 80)
    print("NO FAIRNESS PREPROCESSING (Ablation Study)")
    print("=" * 80)
    
    shutil.copy(input_csv, output_csv)
    
    df = pd.read_csv(output_csv)
    print(f"Copied data from: {input_csv}")
    print(f"Saved to: {output_csv}")
    print(f"Total samples: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print("\nNO fairness corrections applied (control group for ablation)")
    print("=" * 80)
    
    return df


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
    
    NO_FAIRNESS = False
    
    input_csv = f'data/{args.dataset}/{args.dataset}.csv'
    output_csv = f'data/{args.dataset}/{args.dataset}-fair.csv'
    
    if not os.path.exists(input_csv):
        print(f"Error: Input file not found: {input_csv}")
        print("Please run the data loading script first.")
        exit(1)
    
    # if NO_FAIRNESS:
    #     print(f"\n### NO Fairness Preprocessing (Ablation Study) for {args.dataset} ###\n")
    #     print("=" * 80)
    #     print("NO FAIRNESS PREPROCESSING - ABLATION STUDY")
    #     print("=" * 80)
    #     df = pd.read_csv(input_csv)
    #     df_sample = df.sample(n=10000, random_state=42)
    #     df_sample.to_csv(output_csv, index=False)
    #     print(f"Loaded data from: {input_csv}")
    #     print(f"Sampled 10000 rows (same as fairness preprocessing)")
    #     print(f"Saved to: {output_csv}")
    #     print(f"Total samples: {len(df_sample)}")
    #     print("NO fairness corrections applied")
    #     print("=" * 80)
    # else:
    #     print(f"\n### AIF360 Fairness Preprocessing for {args.dataset} ###\n")
    #     fair_data = aif360_fairness_preprocessing(
    #         input_csv=input_csv,
    #         output_csv=output_csv,
    #         protected_attribute=args.protected_attr,
    #         label_name=args.label,
    #         threshold=args.threshold,
    #         repair_level=args.repair_level
    #     )
    
    df_full = pd.read_csv(input_csv)
    np.random.seed(42)
    indices = np.arange(len(df_full))
    np.random.shuffle(indices)
    test_size = int(len(df_full) * 0.2)
    val_size = int(len(df_full) * 0.1)
    train_indices = indices[test_size + val_size:]
    df_train = df_full.iloc[train_indices]
    train_csv = f'data/{args.dataset}/{args.dataset}-train.csv'
    df_train.to_csv(train_csv, index=False)

    if NO_FAIRNESS:
        print(f"\n### NO Fairness Preprocessing (Ablation Study) for {args.dataset} ###\n")
        print("=" * 80)
        print("NO FAIRNESS PREPROCESSING - ABLATION STUDY")
        print("=" * 80)
        df_sample = df_train.sample(n=10000, random_state=42)
        df_sample.to_csv(output_csv, index=False)
        print(f"Loaded data from: {train_csv}")
        print(f"Sampled 10000 rows (same as fairness preprocessing)")
        print(f"Saved to: {output_csv}")
        print(f"Total samples: {len(df_sample)}")
        print("NO fairness corrections applied")
        print("=" * 80)
    else:
        print(f"\n### AIF360 Fairness Preprocessing for {args.dataset} ###\n")
        fair_data = aif360_fairness_preprocessing(
            input_csv=train_csv,
            output_csv=output_csv,
            protected_attribute=args.protected_attr,
            label_name=args.label,
            threshold=args.threshold,
            repair_level=args.repair_level
        )
    
    print("\n### Preprocessing Complete ###\n")