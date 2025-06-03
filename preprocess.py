# Impor library yang diperlukan
import pandas as pd  # Buat manipulasi data tabel
import numpy as np  # Buat operasi matematika
from sklearn.impute import SimpleImputer  # Buat nanganin data yang kosong

def load_and_preprocess_data(uploaded_file):
    """Ngolah data yang diupload tanpa nyimpen ke file"""
    # Baca data dari file yang diupload
    df = pd.read_csv(uploaded_file)
    
    # Ubah tipe data ke numeric biar bisa diproses
    numeric_cols = df.columns.drop('is_safe')  # Ambil semua kolom kecuali target
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')  # Ubah ke angka, kalo gagal jadi NaN
    df['is_safe'] = pd.to_numeric(df['is_safe'], errors='coerce')  # Target juga diubah ke numeric
    
    # Nanganin data yang kosong (missing values)
    imputer = SimpleImputer(strategy='median')  # Pakai median buat ngisi data kosong
    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])  # Terapkan ke semua kolom numerik
    
    # Nanganin data yang ekstrim (outlier) pake metode IQR
    for col in numeric_cols:
        q1 = df[col].quantile(0.25)  # Hitung kuartil bawah
        q3 = df[col].quantile(0.75)  # Hitung kuartil atas
        iqr = q3 - q1  # Hitung jarak interkuartil
        lower_bound = q1 - 1.5 * iqr  # Batas bawah
        upper_bound = q3 + 1.5 * iqr  # Batas atas
        df[col] = np.clip(df[col], lower_bound, upper_bound)  # Potong data yang di luar batas
    
    # Normalisasi fitur biar skalanya sama (0-1)
    for col in numeric_cols:
        df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())  # Rumus normalisasi min-max
    
    # Buang baris yang targetnya null (ga valid)
    df = df.dropna(subset=['is_safe'])
    
    # Pastikan targetnya integer (0 atau 1)
    df['is_safe'] = df['is_safe'].astype(int)
    
    return df  # Kembalikan data yang udah bersih