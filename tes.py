import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight

# 1. Load Data & Preprocessing
df = pd.read_csv('waterQuality1.csv')

# Konversi kolom object ke numeric (jika ada)
df['ammonia'] = pd.to_numeric(df['ammonia'], errors='coerce')
df['is_safe'] = pd.to_numeric(df['is_safe'], errors='coerce')
df = df.dropna()  # Hapus baris dengan NA

# Pisahkan fitur (X) dan target (y)
X = df.drop('is_safe', axis=1)
y = df['is_safe']

# 2. Split Data (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Normalisasi Data (StandardScaler)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Hitung Class Weights untuk Imbalance Handling
class_weights = class_weight.compute_class_weight(
    'balanced', classes=np.unique(y_train), y=y_train
)
class_weights_dict = {0: class_weights[0], 1: class_weights[1]}
print("\nClass Weights:", class_weights_dict)

# 5. Implementasi Gaussian Naive Bayes dengan Class Weight
class GaussianNaiveBayes:
    def __init__(self, class_weights=None):
        self.classes_ = None
        self.class_priors_ = None
        self.means_ = None
        self.stds_ = None
        self.class_weights_ = class_weights or {0: 1, 1: 1}  # Default: no weighting
    
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.class_priors_ = y.value_counts(normalize=True).sort_index()
        
        # Hitung mean dan std untuk tiap fitur per kelas
        self.means_ = {}
        self.stds_ = {}
        
        for c in self.classes_:
            X_c = X[y == c]
            self.means_[c] = X_c.mean(axis=0)
            self.stds_[c] = X_c.std(axis=0)
            # Hindari std = 0 dengan menambahkan nilai kecil
            self.stds_[c] = np.where(self.stds_[c] == 0, 1e-10, self.stds_[c])
    
    def _calculate_likelihood(self, x, mean, std):
        exponent = np.exp(-((x - mean) ** 2 / (2 * std ** 2)))
        return (1 / (np.sqrt(2 * np.pi) * std)) * exponent
    
    def predict_proba(self, X):
        probas = []
        for _, sample in pd.DataFrame(X).iterrows():
            posteriors = []
            for c in self.classes_:
                prior = np.log(self.class_priors_[c] * self.class_weights_[c])
                likelihood = np.sum(np.log(self._calculate_likelihood(sample, self.means_[c], self.stds_[c])))
                posteriors.append(prior + likelihood)
            # Konversi ke probabilitas dengan softmax
            probas.append(np.exp(posteriors) / np.sum(np.exp(posteriors)))
        return np.array(probas)
    
    def predict(self, X, threshold=0.5):
        probas = self.predict_proba(X)
        return (probas[:, 1] >= threshold).astype(int)

# 6. Train Model Naive Bayes
gnb = GaussianNaiveBayes(class_weights=class_weights_dict)
gnb.fit(X_train_scaled, y_train)

# 7. Prediksi dengan Threshold Default (0.5)
y_pred_nb = gnb.predict(X_test_scaled)
print("\n=== Hasil Naive Bayes (Threshold Default 0.5) ===")
print(classification_report(y_test, y_pred_nb))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_nb))

# 8. Dapatkan Probabilitas Prediksi untuk Data Test
y_proba = gnb.predict_proba(X_test_scaled)[:, 1]

# 9. Implementasi PSO untuk Optimasi Threshold
class PSO:
    def __init__(self, n_particles, w, max_iter):
        self.n_particles = n_particles
        self.w = w  # Inertia weight
        self.c1 = 1.5  # Cognitive coefficient (fixed)
        self.c2 = 1.5  # Social coefficient (fixed)
        self.max_iter = max_iter
    
    def optimize(self, y_true, y_proba):
        # Fungsi objektif: memaksimalkan F1-score
        def objective_func(threshold):
            y_pred = (y_proba >= threshold).astype(int)
            return -f1_score(y_true, y_pred)  # Minimalkan negative F1-score
        
        # Inisialisasi partikel (threshold antara 0 dan 1)
        particles = np.random.uniform(0, 1, self.n_particles)
        velocities = np.zeros(self.n_particles)
        personal_best_positions = particles.copy()
        personal_best_scores = np.array([objective_func(p) for p in particles])
        
        global_best_position = personal_best_positions[np.argmin(personal_best_scores)]
        global_best_score = np.min(personal_best_scores)
        
        # Iterasi PSO
        for _ in range(self.max_iter):
            for i in range(self.n_particles):
                # Update velocity
                r1, r2 = np.random.rand(2)
                velocities[i] = (self.w * velocities[i] +
                                self.c1 * r1 * (personal_best_positions[i] - particles[i]) +
                                self.c2 * r2 * (global_best_position - particles[i]))
                
                # Update position
                particles[i] = np.clip(particles[i] + velocities[i], 0, 1)
                
                # Evaluasi
                current_score = objective_func(particles[i])
                
                # Update personal best
                if current_score < personal_best_scores[i]:
                    personal_best_scores[i] = current_score
                    personal_best_positions[i] = particles[i]
            
            # Update global best
            if np.min(personal_best_scores) < global_best_score:
                global_best_position = personal_best_positions[np.argmin(personal_best_scores)]
                global_best_score = np.min(personal_best_scores)
        
        return global_best_position, -global_best_score  # Return threshold dan F1-score

# 10. Input Parameter PSO dari User (kecuali c1 dan c2 yang sudah fixed 1.5)
print("\n=== Masukkan Parameter PSO ===")
n_particles = int(input("Jumlah partikel (populasi): "))
w = float(input("Inertia weight (w): "))
max_iter = int(input("Jumlah iterasi: "))

pso = PSO(n_particles=n_particles, w=w, max_iter=max_iter)
optimal_threshold, optimal_f1 = pso.optimize(y_test, y_proba)

print(f"\nThreshold Optimal PSO: {optimal_threshold:.4f}")
print(f"F1-Score Optimal: {optimal_f1:.4f}")

# 11. Prediksi dengan Threshold PSO
y_pred_pso = (y_proba >= optimal_threshold).astype(int)
print("\n=== Hasil Naive Bayes + PSO ===")
print(classification_report(y_test, y_pred_pso))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_pso))

# 12. Perbandingan Hasil
print("\n=== Perbandingan Hasil ===")
print("1. Naive Bayes (Threshold 0.5):")
print(f"   - F1-Score: {f1_score(y_test, y_pred_nb):.4f}")
print("2. Naive Bayes + PSO:")
print(f"   - F1-Score: {optimal_f1:.4f}")
print(f"   - Threshold Optimal: {optimal_threshold:.4f}")