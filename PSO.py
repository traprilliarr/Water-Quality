import numpy as np
from sklearn.metrics import f1_score

class PSO:
    def __init__(self, n_particles=30, w=0.7, c1=1.5, c2=1.5, max_iter=50):
        """Inisialisasi algoritma PSO
        
        Parameter:
        - n_particles: Jumlah partikel dalam swarm
        - w: Inertia weight (bobot kecepatan sebelumnya)
        - c1: Cognitive coefficient (pengaruh best pribadi)
        - c2: Social coefficient (pengaruh best global)
        - max_iter: Jumlah maksimum iterasi
        """
        self.n_particles = n_particles
        self.w = w  # Inertia weight
        self.c1 = c1  # Koefisien kognitif
        self.c2 = c2  # Koefisien sosial
        self.max_iter = max_iter  # Maksimum iterasi
    
    def optimize(self, y_true, y_proba):
        """Fungsi optimasi untuk mencari threshold terbaik
        
        Parameter:
        - y_true: Label sebenarnya (ground truth)
        - y_proba: Probabilitas prediksi dari model
        
        Return:
        - Threshold optimal
        - Nilai F1-score terbaik
        """
        
        # Fungsi objektif: meminimalkan negative F1-score
        # Kita meminimalkan negatif F1 karena PSO default mencari minimum
        def objective(threshold):
            y_pred = (y_proba >= threshold).astype(int)
            return -f1_score(y_true, y_pred)  # Return negatif F1
        
        # [1. INISIALISASI SWARM]
        # Partikel diinisialisasi secara acak antara 0 dan 1 (karena threshold)
        particles = np.random.uniform(0, 1, self.n_particles)
        velocities = np.zeros(self.n_particles)  # Kecepatan awal = 0
        
        # [2. INISIALISASI BEST POSITIONS]
        # Simpan posisi terbaik setiap partikel
        personal_best_positions = particles.copy()
        
        # Hitung skor awal setiap partikel
        personal_best_scores = np.array([objective(p) for p in particles])
        
        # Temukan partikel terbaik secara global
        global_best_idx = np.argmin(personal_best_scores)
        global_best_position = particles[global_best_idx]
        global_best_score = personal_best_scores[global_best_idx]
        
        # [3. PROSES OPTIMASI]
        for iter_num in range(self.max_iter):
            for i in range(self.n_particles):
                # [3.1 UPDATE KECEPATAN]
                # Rumus update kecepatan PSO:
                # v = w*v + c1*r1*(pbest - x) + c2*r2*(gbest - x)
                r1, r2 = np.random.rand(2)  # Faktor acak untuk eksplorasi
                cognitive = self.c1 * r1 * (personal_best_positions[i] - particles[i])
                social = self.c2 * r2 * (global_best_position - particles[i])
                velocities[i] = self.w * velocities[i] + cognitive + social
                
                # [3.2 UPDATE POSISI]
                # x = x + v
                particles[i] = np.clip(particles[i] + velocities[i], 0, 1)
                
                # [3.3 EVALUASI]
                current_score = objective(particles[i])
                
                # [3.4 UPDATE BEST POSITIONS]
                # Update personal best jika ditemukan yang lebih baik
                if current_score < personal_best_scores[i]:
                    personal_best_positions[i] = particles[i]
                    personal_best_scores[i] = current_score
                    
                    # Update global best jika ditemukan yang lebih baik
                    if current_score < global_best_score:
                        global_best_position = particles[i]
                        global_best_score = current_score
        
        # Kembalikan threshold optimal dan F1-score terbaik
        # (F1-score asli, bukan negatifnya)
        return global_best_position, -global_best_score