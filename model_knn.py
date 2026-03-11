import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.manifold import TSNE
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import model_selection
import pickle
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.neighbors import KNeighborsClassifier
import statistics
import seaborn as sns
import joblib
from sklearn.model_selection import cross_val_score
from sklearn.compose import ColumnTransformer
from matplotlib.colors import ListedColormap
import warnings
warnings.filterwarnings('ignore')

# Meload dataset
df = pd.read_csv('dataset_magang.csv')
df

# Untuk mengetahui info dari datasset
df.info()

# =============================
# Data Cleaning untuk Dataset Baru
# =============================

# Buat copy dataframe untuk menjaga data original
df_cleaned = df.copy()

print("=" * 60)
print("DATA SEBELUM CLEANING")
print("=" * 60)
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print("\n5 baris pertama data original:")
print(df.head().to_string())

# Hapus kolom 'Nama' dan 'Instansi'
if 'Nama' in df_cleaned.columns:
    df_cleaned = df_cleaned.drop(columns=['Nama'])
    print(f"\n✅ Kolom 'Nama' dihapus")

if 'Instansi' in df_cleaned.columns:
    df_cleaned = df_cleaned.drop(columns=['Instansi'])
    print(f"✅ Kolom 'Instansi' dihapus")

# Ubah nama kolom untuk konsistensi
df_cleaned = df_cleaned.rename(columns={
    'Jurusan': 'jurusan',
    'Divisi': 'divisi',
    'Tanggal Mulai': 'tanggal_mulai',
    'Tanggal Akhir': 'tanggal_akhir',
    'Mapel1': 'mapel1',
    'Mapel2': 'mapel2',
    'Skill Teknis': 'skill_teknis',
    'Sertifikasi': 'sertifikasi',
    'Proyek': 'proyek'
})

# Ubah tipe tanggal
df_cleaned['tanggal_mulai'] = pd.to_datetime(df_cleaned['tanggal_mulai'])
df_cleaned['tanggal_akhir'] = pd.to_datetime(df_cleaned['tanggal_akhir'])

# Tambah fitur durasi
df_cleaned['durasi_hari'] = (df_cleaned['tanggal_akhir'] - df_cleaned['tanggal_mulai']).dt.days

# Normalisasi teks untuk semua fitur kategorikal
categorical_columns = ['jurusan', 'divisi', 'mapel1', 'mapel2', 'skill_teknis', 'sertifikasi', 'proyek']
for col in categorical_columns:
    df_cleaned[col] = df_cleaned[col].astype(str).str.strip().str.lower()

# =============================
# VISUALISASI HASIL CLEANING
# =============================

print("\n" + "=" * 60)
print("HASIL DATA CLEANING")
print("=" * 60)
print(f"Shape: {df_cleaned.shape}")
print(f"Columns: {list(df_cleaned.columns)}")

# Tampilkan sample data sebelum dan sesudah cleaning
print("\n📋 PERBANDINGAN SEBELUM & SESUDAH CLEANING")
print("-" * 50)

# Buat dataframe comparision
comparison_sample = pd.concat([
    df.head(5).reset_index(drop=True),
    df_cleaned.head(5).reset_index(drop=True)
], axis=1, keys=['Sebelum Cleaning', 'Sesudah Cleaning'])

print(comparison_sample.to_string())

# =============================
# INFORMASI DATA CLEANING
# =============================

print("\n" + "=" * 60)
print("INFORMASI DATA CLEANING")
print("=" * 60)

# 1. Info missing values
print("🔍 MISSING VALUES:")
missing_info = df_cleaned.isnull().sum()
print(missing_info[missing_info > 0] if missing_info.sum() > 0 else "Tidak ada missing values")

# 2. Info duplikat
duplicate_count = df_cleaned.duplicated().sum()
print(f"\n🔍 DATA DUPLIKAT: {duplicate_count}")

# 3. Info data types
print("\n🔍 TIPE DATA:")
print(df_cleaned.dtypes)

# 4. Statistik durasi_hari
print("\n📊 STATISTIK DURASI MAGANG:")
print(df_cleaned['durasi_hari'].describe())

# =============================
# VISUALISASI DISTRIBUSI SETELAH CLEANING
# =============================

fig, axes = plt.subplots(2, 3, figsize=(20, 12))

# 1. Distribusi Jurusan (Top 10)
jurusan_counts = df_cleaned['jurusan'].value_counts().head(10)
axes[0,0].barh(jurusan_counts.index, jurusan_counts.values)
axes[0,0].set_title('Top 10 Jurusan')
axes[0,0].set_xlabel('Jumlah')

# 2. Distribusi Divisi
divisi_counts = df_cleaned['divisi'].value_counts()
axes[0,1].pie(divisi_counts.values, labels=divisi_counts.index, autopct='%1.1f%%')
axes[0,1].set_title('Distribusi Divisi')

# 3. Distribusi Durasi Magang
axes[0,2].hist(df_cleaned['durasi_hari'], bins=20, edgecolor='black', alpha=0.7)
axes[0,2].set_title('Distribusi Durasi Magang (Hari)')
axes[0,2].set_xlabel('Durasi (Hari)')
axes[0,2].set_ylabel('Frekuensi')

# 4. Distribusi Mapel1 (Top 10)
mapel1_counts = df_cleaned['mapel1'].value_counts().head(10)
axes[1,0].barh(mapel1_counts.index, mapel1_counts.values)
axes[1,0].set_title('Top 10 Mapel1')
axes[1,0].set_xlabel('Jumlah')

# 5. Distribusi Skill Teknis (Top 10)
skill_counts = df_cleaned['skill_teknis'].value_counts().head(10)
axes[1,1].barh(skill_counts.index, skill_counts.values)
axes[1,1].set_title('Top 10 Skill Teknis')
axes[1,1].set_xlabel('Jumlah')

# 6. Distribusi Sertifikasi (Top 10)
sertifikasi_counts = df_cleaned['sertifikasi'].value_counts().head(10)
axes[1,2].barh(sertifikasi_counts.index, sertifikasi_counts.values)
axes[1,2].set_title('Top 10 Sertifikasi')
axes[1,2].set_xlabel('Jumlah')

plt.tight_layout()
plt.show()

# Hapus duplikat dulu
df_unique = df_cleaned.drop_duplicates()

# Ambil X dan y dari data unik
X = df_unique.drop("divisi", axis=1)
y = df_unique["divisi"]

# Split data train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42)

print(f"Jumlah data unik: {len(df_unique)}")
print(f"Data Train: {len(X_train)} ({len(X_train)/len(df_unique)*100:.1f}%)")
print(f"Data Test : {len(X_test)} ({len(X_test)/len(df_unique)*100:.1f}%)")

# =============================
# Data Preprocessing untuk Dataset Baru
# =============================

# Define features (X) dan target (y)
# Menggunakan fitur-fitur baru: jurusan, mapel1, mapel2, skill_teknis, sertifikasi, proyek, durasi_hari
X = df_unique[['jurusan', 'mapel1', 'mapel2', 'skill_teknis', 'sertifikasi', 'proyek', 'durasi_hari']]
y = df_unique['divisi']

# Encode target variable (divisi)
le_divisi = LabelEncoder()
y_encoded = le_divisi.fit_transform(y)

# Identifikasi fitur kategorikal dan numerik
categorical_features = ['jurusan', 'mapel1', 'mapel2', 'skill_teknis', 'sertifikasi', 'proyek']
numerical_features = ['durasi_hari']

# Create a column transformer for preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('num', StandardScaler(), numerical_features)
    ],
    remainder='passthrough'
)

# Apply preprocessing
X_processed = preprocessor.fit_transform(X)

# Convert the processed data to a dense array sebelum membuat DataFrame
X_processed_dense = X_processed.toarray()

# Dapatkan nama fitur setelah one-hot encoding
cat_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
all_feature_names = list(cat_feature_names) + numerical_features

X_processed_df = pd.DataFrame(X_processed_dense, columns=all_feature_names)

print("Data setelah preprocessing:")
print(X_processed_df.head().to_string())
print(f"\nShape setelah preprocessing: {X_processed_df.shape}")

# =============================
# Train-Test Split (70% training, 30% testing)
# =============================
X_train, X_test, y_train, y_test = train_test_split(
    X_processed_df, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
)

print(f"\n\nData Train: {X_train.shape[0]} ({X_train.shape[0]/len(X_processed_df)*100:.1f}%)")
print(f"Data Test : {X_test.shape[0]} ({X_test.shape[0]/len(X_processed_df)*100:.1f}%)")

# Buat DataFrames untuk data training dan testing (for display/inspection, use arrays for model fitting)
X_train_df = pd.DataFrame(X_train, columns=X_processed_df.columns)
X_test_df = pd.DataFrame(X_test, columns=X_processed_df.columns)


print("\nSample Data Training after Preprocessing:")
print(pd.DataFrame(X_train, columns=X_processed_df.columns).head().to_string(index=False)) # Use X_train array for display

print("\nSample Data Testing after Preprocessing:")
print(pd.DataFrame(X_test, columns=X_processed_df.columns).head().to_string(index=False)) # Use X_test array for display


# =============================
# Visualisasi Proporsi Data
# =============================

plt.figure(figsize=(8, 6))
sizes = [len(X_train), len(X_test)] # Use X_train and X_test arrays
labels = ['Data Latih (70%)', 'Data Uji (30%)']
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
plt.title("Proporsi Data Latih & Data Uji")
plt.show()

# =============================
# Pemilihan k Terbaik dengan Cross-Validation
# =============================
k_range = range(1, 21)  # Uji k dari 1 sampai 20
cv_scores = []
cv_scores_std = []  # Untuk standar deviasi

print("=" * 50)
print("PEMILIHAN k TERBAIK DENGAN CROSS-VALIDATION")
print("=" * 50)

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=5, scoring='accuracy')
    cv_scores.append(scores.mean())
    cv_scores_std.append(scores.std())
    print(f"k = {k:2d} | Akurasi: {scores.mean():.4f} ± {scores.std():.4f}")

# Ambil k terbaik
best_k = k_range[np.argmax(cv_scores)]
best_score = max(cv_scores)

print("\n" + "=" * 50)
print(f"✅ K TERBAIK: {best_k} dengan akurasi rata-rata: {best_score:.4f}")
print("=" * 50)

# Visualisasi pemilihan k
plt.figure(figsize=(12, 6))
plt.plot(k_range, cv_scores, marker='o', linestyle='-', color='b', linewidth=2, markersize=8, label='Akurasi Rata-rata')
plt.fill_between(k_range,
                 np.array(cv_scores) - np.array(cv_scores_std),
                 np.array(cv_scores) + np.array(cv_scores_std),
                 alpha=0.2, color='b', label='±1 Std Dev')

plt.axvline(x=best_k, color='r', linestyle='--', linewidth=2, label=f'k Terbaik = {best_k}')
plt.xlabel('Nilai k', fontsize=12)
plt.ylabel('Akurasi Rata-rata (CV)', fontsize=12)
plt.title('Pemilihan k Terbaik dengan 5-Fold Cross-Validation\n', fontsize=14, fontweight='bold')
plt.xticks(k_range)
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

# =============================
# Model Training dengan KNN (Menggunakan Data Preprocessing)
# =============================

# Use the preprocessed and split data from previous steps
# X_train_df, X_test_df, y_train, y_test are already defined and contain numerical data
# from cell QlY8WAs37dgi

# Inisialisasi dan latih model KNN dengan k terbaik
# Using k=5 as determined as the best k in cell sO6w6F6n7uGE
knn_full = KNeighborsClassifier(n_neighbors=best_k) # Use the best_k found in sO6w6F6n7uGE
knn_full.fit(X_train_df, y_train) # Fit with the preprocessed training data

# Evaluate the model on the test set
test_accuracy = knn_full.score(X_test_df, y_test)

print(f"Akurasi model KNN pada data test (k={best_k}): {test_accuracy:.4f}")

# You can add more evaluation metrics here if needed, e.g., classification_report, confusion_matrix
# from sklearn.metrics import classification_report, confusion_matrix
# y_pred = knn_full.predict(X_test_df)
# print("\nClassification Report:")
# print(classification_report(y_test, y_pred, target_names=le_divisi.classes_)) # Assuming le_divisi is available

# The code for plotting decision regions with only two features is removed
# as it was causing issues and is not directly related to training the main model

# =============================
# Modeling K-NN
# =============================
k = 4  # jumlah tetangga terdekat
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)

# =============================
# Modeling K-NN dengan k terbaik (k=4)
# =============================

# Gunakan k terbaik dari cross-validation
best_k = 4  # Dari hasil cross-validation

# Inisialisasi dan latih model KNN dengan k terbaik
knn_best = KNeighborsClassifier(n_neighbors=best_k)
knn_best.fit(X_train, y_train)

# Prediksi pada data training dan testing
y_train_pred = knn_best.predict(X_train)
y_test_pred = knn_best.predict(X_test)

# Hitung akurasi
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print("=" * 60)
print("HASIL MODEL KNN DENGAN k TERBAIK (k=4)")
print("=" * 60)
print(f"Akurasi Training: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
print(f"Akurasi Testing:  {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print(f"Selisih: {abs(train_accuracy - test_accuracy):.4f}")

# =============================
# Evaluasi Model
# =============================

# Confusion Matrix
cm = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le_divisi.classes_,
            yticklabels=le_divisi.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title(f'Confusion Matrix - KNN (k={best_k})')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# Classification Report
print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)
print(classification_report(y_test, y_test_pred,
                           target_names=le_divisi.classes_))


report = classification_report(y_test, y_test_pred,
                               target_names=le_divisi.classes_,
                               output_dict=True)
df_report = pd.DataFrame(report).transpose()

plt.figure(figsize=(8,5))
df_report['recall'][:-3].plot(kind='bar', color='skyblue')
plt.title("Recall per Divisi")
plt.ylabel("Recall")
plt.ylim(0,1)
plt.show()