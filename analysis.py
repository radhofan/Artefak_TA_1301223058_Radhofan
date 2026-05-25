import pandas as pd
import os

def extract_paper_statistics(data_path='data/ml-100k'):
    print("=" * 60)
    print("PAPER DATASET STATISTICS EXTRACTOR")
    print("=" * 60)
    
    # 1. Load Data
    ratings = pd.read_csv(
        os.path.join(data_path, 'u.data'),
        sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'], engine='python'
    )
    users = pd.read_csv(
        os.path.join(data_path, 'u.user'),
        sep='|', names=['user_id', 'age', 'gender', 'occupation', 'zip_code'], engine='python'
    )
    
    # 2. Merge Data
    data = pd.merge(ratings, users, on='user_id', how='left')
    data['gender_binary'] = (data['gender'] == 'M').astype(int)
    
    features = ['user_id', 'item_id', 'rating', 'timestamp', 'gender', 'gender_binary', 'age', 'occupation']
    gan_data = data[features]
    
    # --- METRIC 1: Features ---
    print(f"\n[1] FEATURE COUNT")
    print(f"Total features kept: {len(features)}")
    print(f"Feature list: {', '.join(features)}")
    
    # --- METRIC 2: Demographic Distribution (Users) ---
    total_unique_users = users['user_id'].nunique()
    male_users = len(users[users['gender'] == 'M'])
    female_users = len(users[users['gender'] == 'F'])
    
    print(f"\n[2] DEMOGRAPHIC DISTRIBUTION (UNIQUE USERS)")
    print(f"Total Users: {total_unique_users}")
    print(f"Male: {male_users} ({(male_users/total_unique_users)*100:.2f}%)")
    print(f"Female: {female_users} ({(female_users/total_unique_users)*100:.2f}%)")

    # --- METRIC 3: Demographic Distribution (Interactions) ---
    total_interactions = len(gan_data)
    male_interactions = len(gan_data[gan_data['gender'] == 'M'])
    female_interactions = len(gan_data[gan_data['gender'] == 'F'])
    
    print(f"\n[3] DEMOGRAPHIC DISTRIBUTION (INTERACTIONS)")
    print(f"Total Interactions: {total_interactions}")
    print(f"Male Interactions: {male_interactions} ({(male_interactions/total_interactions)*100:.2f}%)")
    print(f"Female Interactions: {female_interactions} ({(female_interactions/total_interactions)*100:.2f}%)")
    
    # --- METRIC 4: Data Splitting Sizes ---
    test_ratio = 0.20
    val_ratio = 0.10
    train_ratio = 1.0 - test_ratio - val_ratio
    
    test_size = int(total_interactions * test_ratio)
    val_size = int(total_interactions * val_ratio)
    train_size = total_interactions - test_size - val_size
    
    print(f"\n[4] DATA SPLITTING (70/10/20)")
    print(f"Training Data (70%): {train_size} samples")
    print(f"Validation Data (10%): {val_size} samples")
    print(f"Testing Data (20%): {test_size} samples")
    print("=" * 60)

if __name__ == '__main__':
    try:
        extract_paper_statistics()
    except FileNotFoundError:
        print("Error: Could not find data/ml-100k. Make sure you run this from the project root.")