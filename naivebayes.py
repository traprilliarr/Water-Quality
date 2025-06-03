import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight

class GaussianNaiveBayes:
    def __init__(self, use_class_weight=True):
        """Inisialisasi model Naive Bayes Gaussian"""
        self.classes_ = None      # Daftar kelas unik [0, 1] untuk klasifikasi biner
        self.class_priors_ = None # Probabilitas prior P(y=c) untuk tiap kelas
        self.means_ = None        # Rata-rata (μ) tiap fitur per kelas: {kelas: [μ1, μ2, ...]}
        self.stds_ = None         # Standar deviasi (σ) tiap fitur per kelas
        self.use_class_weight = use_class_weight
        self.class_weights_ = None  # Bobot kelas untuk menangani data tidak seimbang

    def fit(self, X, y):
        """Melatih model dengan data X dan label y"""
        self.classes_ = np.unique(y)  # Ambil kelas unik
        
        # Hitung bobot kelas untuk data tidak seimbang
        # Rumus bobot: jumlah_sampel / (jumlah_kelas * jumlah_kelas_c)
        if self.use_class_weight:
            weights = compute_class_weight('balanced', classes=self.classes_, y=y)
            self.class_weights_ = dict(zip(self.classes_, weights))
        else:
            self.class_weights_ = {c: 1 for c in self.classes_}  # Bobot seragam
            
        # Hitung probabilitas prior P(y=c) = jumlah sampel kelas c / total sampel
        self.class_priors_ = y.value_counts(normalize=True).sort_index()
        
        # Hitung mean (μ) dan std dev (σ) untuk tiap fitur di tiap kelas
        self.means_ = {}
        self.stds_ = {}
        
        for c in self.classes_:
            X_c = X[y == c]  # Ambil data untuk kelas c saja
            self.means_[c] = X_c.mean(axis=0)  # Hitung rata-rata tiap kolom
            self.stds_[c] = X_c.std(axis=0)    # Hitung std dev tiap kolom
            
            # Hindari pembagian dengan nol (ganti σ=0 dengan nilai sangat kecil)
            self.stds_[c] = np.where(self.stds_[c] == 0, 1e-10, self.stds_[c])

    def _calculate_likelihood(self, x, mean, std):
        """
        Menghitung likelihood menggunakan distribusi normal (Gaussian)
        Rumus PDF Gaussian:
        P(x|μ,σ) = (1/√(2πσ²)) * e^(-(x-μ)²/(2σ²))
        """
        exponent = np.exp(-((x - mean) ** 2 / (2 * std ** 2)))
        return (1 / (np.sqrt(2 * np.pi) * std)) * exponent

    def predict_proba(self, X):
        """Menghitung probabilitas kelas untuk data X"""
        probas = []  # Untuk menyimpan hasil probabilitas
        
        # Konversi ke DataFrame jika belum
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        
        for _, sample in X_df.iterrows():
            posteriors = []  # Untuk menyimpan posterior tiap kelas
            
            for c in self.classes_:
                # Hitung log posterior = log(P(y=c)) + log(P(x1|y=c)) + log(P(x2|y=c)) + ...
                prior = np.log(self.class_priors_[c] * self.class_weights_[c])
                
                # Hitung likelihood dengan log untuk stabilitas numerik
                likelihood = np.sum(
                    np.log(self._calculate_likelihood(sample, self.means_[c], self.stds_[c]))
                )
                posteriors.append(prior + likelihood)
            
            # Konversi ke probabilitas dengan softmax:
            # P(y=c|x) = e^posterior_c / sum(e^posterior_all_classes)
            probas.append(np.exp(posteriors) / np.sum(np.exp(posteriors)))
        
        return np.array(probas)

    def predict(self, X, threshold=0.5):
        """
        Memprediksi kelas berdasarkan threshold
        Default threshold 0.5 (50%) untuk klasifikasi biner
        """
        probas = self.predict_proba(X)
        # Kelas 1 jika probabilitas >= threshold, else kelas 0
        return (probas[:, 1] >= threshold).astype(int)