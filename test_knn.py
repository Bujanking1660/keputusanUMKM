import pandas as pd
import numpy as np

# Load training data
train_df = pd.read_csv('data/data.csv')

# Test data (from the user's image)
test_data = [
    {"Nama_UKM": "Umk Pamurbaya", "Lama_Usaha": 4, "Jumlah_Pekerja": 15, "Omzet": 4, "Jumlah_Aset": 6},
    {"Nama_UKM": "Umk Bandeng", "Lama_Usaha": 3, "Jumlah_Pekerja": 28, "Omzet": 4, "Jumlah_Aset": 10},
    {"Nama_UKM": "Bukid Graffer", "Lama_Usaha": 2, "Jumlah_Pekerja": 12, "Omzet": 1, "Jumlah_Aset": 3}
]

features = ["Lama_Usaha", "Jumlah_Pekerja", "Omzet", "Jumlah_Aset"]

# Let's perform kNN for each test instance with k=3 and k=5
for q in test_data:
    q_name = q["Nama_UKM"]
    q_val = np.array([q[f] for f in features])
    
    print(f"\n### Analisis kNN untuk: {q_name} {q_val}")
    
    df = train_df.copy()
    
    # Calculate squared distance
    dists_sq = []
    for idx, row in df.iterrows():
        x_val = row[features].values.astype(float)
        d_sq = np.sum((x_val - q_val) ** 2)
        dists_sq.append(d_sq)
        
    df['Square_Distance'] = dists_sq
    df['Distance'] = np.sqrt(dists_sq)
    
    # Sort by distance
    df = df.sort_values(by='Distance').reset_index(drop=True)
    df['Jarak_Terkecil_Rank'] = df.index + 1
    
    # For k=3
    df['Is_NN_k3'] = df['Jarak_Terkecil_Rank'] <= 3
    # For k=5
    df['Is_NN_k5'] = df['Jarak_Terkecil_Rank'] <= 5
    
    # Determine decision result
    for k in [3, 5]:
        nn_labels = df.head(k)['Hasil_Keputusan'].tolist()
        # Find majority vote
        from collections import Counter
        votes = Counter(nn_labels)
        # Find most common label, in case of tie, sort alphabetically or by distance?
        # Let's see the votes
        most_common = votes.most_common()
        if len(most_common) > 1 and most_common[0][1] == most_common[1][1]:
            # Tie: could choose based on closer average distance, or alphabet
            # Let's see what the labels are
            best_label = most_common[0][0]
        else:
            best_label = most_common[0][0]
        print(f"Keputusan (k={k}): {best_label} (votes: {dict(votes)})")
        
    # Display the table
    print(df[['Nama_UKM', 'Lama_Usaha', 'Jumlah_Pekerja', 'Omzet', 'Jumlah_Aset', 'Square_Distance', 'Distance', 'Jarak_Terkecil_Rank', 'Hasil_Keputusan', 'Is_NN_k3', 'Is_NN_k5']].to_string())
