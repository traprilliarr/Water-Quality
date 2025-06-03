import streamlit as st
import pandas as pd
import numpy as np
from preprocess import load_and_preprocess_data
from naivebayes import GaussianNaiveBayes
from PSO import PSO
from sklearn.metrics import classification_report, f1_score, confusion_matrix, precision_score, recall_score, accuracy_score

# Konfigurasi halaman
st.set_page_config(
    page_title="Sistem Klasifikasi Kualitas Air",
    page_icon="💧",
    layout="wide"
)

# Inisialisasi session state
if 'preprocessed_data' not in st.session_state:
    st.session_state.preprocessed_data = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'nb_results' not in st.session_state:
    st.session_state.nb_results = None
if 'pso_results' not in st.session_state:
    st.session_state.pso_results = None
if 'pso_threshold' not in st.session_state:
    st.session_state.pso_threshold = 0.5  # Threshold default

# CSS untuk tampilan yang lebih baik
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border-left: 4px solid #4e73df;
    }
    .metric-title {
        font-size: 14px;
        font-weight: 600;
        color: #6c757d;
        margin-bottom: 5px;
    }
    .metric-value {
        font-size: 24px;
        font-weight: 700;
        color: #2e59d9;
    }
    .positive-delta {
        color: #1cc88a;
        font-size: 14px;
    }
    .negative-delta {
        color: #e74a3b;
        font-size: 14px;
    }
    .header {
        color: #2e59d9;
        border-bottom: 2px solid #2e59d9;
        padding-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)

def create_metric_card(title, value, delta=None):
    delta_html = ""
    if delta is not None:
        if delta >= 0:
            delta_html = f"<div class='positive-delta'>+{delta:.4f}</div>"
        else:
            delta_html = f"<div class='negative-delta'>{delta:.4f}</div>"
    
    return f"""
    <div class="metric-card">
        <div class="metric-title">{title}</div>
        <div class="metric-value">{value:.4f}</div>
        {delta_html}
    </div>
    """

# Navigasi sidebar
st.sidebar.title("Menu Navigasi")
page = st.sidebar.radio(
    "Pilih Halaman",
    ["📤 Unggah Data", "📊 Klasifikasi"]
)

# Halaman 1: Unggah Data
if page == "📤 Unggah Data":
    st.title("📤 Unggah Data Kualitas Air")
    
    uploaded_file = st.file_uploader(
        "Pilih file CSV", 
        type=["csv"],
        help="File harus berisi parameter kualitas air dan kolom is_safe"
    )
    
    if uploaded_file is not None:
        try:
            # Preprocessing data
            df = load_and_preprocess_data(uploaded_file)
            st.session_state.preprocessed_data = df
            
            # Tampilkan data
            st.success("✅ Data berhasil diproses dan disimpan!")
            st.dataframe(df)
            
            # Info dasar
            with st.expander("Informasi Dataset"):
                st.write(f"Jumlah sampel: {len(df)}")
                st.write(f"Jumlah fitur: {len(df.columns)-1}")
                st.write(f"Sampel Aman (1): {sum(df['is_safe'])}")
                st.write(f"Sampel Tidak Aman (0): {len(df) - sum(df['is_safe'])}")
        
        except Exception as e:
            st.error(f"❌ Gagal memproses data: {str(e)}")

# Halaman 2: Klasifikasi
elif page == "📊 Klasifikasi":
    st.title("📊 Klasifikasi Kualitas Air")
    
    if st.session_state.preprocessed_data is None:
        st.warning("⚠️ Silakan unggah data terlebih dahulu di halaman Unggah Data")
    else:
        df = st.session_state.preprocessed_data
        X = df.drop('is_safe', axis=1)
        y = df['is_safe']
        
        # Bagian 1: Naive Bayes Standar
        st.markdown("### 1. Klasifikasi Naive Bayes", unsafe_allow_html=True)
        
        if st.button("🔍 Jalankan Naive Bayes"):
            with st.spinner("Menjalankan Naive Bayes..."):
                try:
                    # Inisialisasi dan training model
                    gnb = GaussianNaiveBayes()
                    gnb.fit(X, y)
                    st.session_state.model = gnb
                    
                    # Prediksi dengan threshold default
                    y_pred_nb = gnb.predict(X)
                    y_proba_nb = gnb.predict_proba(X)[:, 1]
                    
                    # Hitung metrik
                    report_dict = classification_report(y, y_pred_nb, output_dict=True)
                    accuracy = accuracy_score(y, y_pred_nb)
                    precision = precision_score(y, y_pred_nb)
                    recall = recall_score(y, y_pred_nb)
                    f1 = f1_score(y, y_pred_nb)
                    
                    # Simpan hasil
                    st.session_state.nb_results = {
                        'report': report_dict,
                        'cm': confusion_matrix(y, y_pred_nb),
                        'accuracy': accuracy,
                        'precision': precision,
                        'recall': recall,
                        'f1': f1,
                        'y_pred': y_pred_nb,
                        'y_proba': y_proba_nb
                    }
                    
                    st.success("✅ Klasifikasi Naive Bayes selesai!")
                    
                except Exception as e:
                    st.error(f"❌ Error dalam Naive Bayes: {str(e)}")
        
        # Tampilkan hasil NB jika sudah ada
        if st.session_state.nb_results is not None:
            nb_results = st.session_state.nb_results
            
            st.markdown("#### Hasil Naive Bayes", unsafe_allow_html=True)
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(create_metric_card("Akurasi", nb_results['accuracy']), unsafe_allow_html=True)
            with col2:
                st.markdown(create_metric_card("Presisi", nb_results['precision']), unsafe_allow_html=True)
            with col3:
                st.markdown(create_metric_card("Recall", nb_results['recall']), unsafe_allow_html=True)
            with col4:
                st.markdown(create_metric_card("F1-Score", nb_results['f1']), unsafe_allow_html=True)
            
            with st.expander("Detail Laporan Naive Bayes"):
                st.markdown("#### Laporan Klasifikasi", unsafe_allow_html=True)
                st.table(pd.DataFrame(nb_results['report']).transpose())
                
                st.markdown("#### Confusion Matrix", unsafe_allow_html=True)
                cm_df = pd.DataFrame(
                    nb_results['cm'],
                    index=['Aktual Tidak Aman (0)', 'Aktual Aman (1)'],
                    columns=['Prediksi Tidak Aman (0)', 'Prediksi Aman (1)']
                )
                st.dataframe(
                    cm_df.style
                    .background_gradient(cmap='Blues')
                    .set_properties(**{'text-align': 'center'})
                )
            
            st.markdown("---")
            
            # Bagian 2: Optimasi PSO
            st.markdown("### 2. Optimasi Threshold dengan PSO", unsafe_allow_html=True)
            
            st.info("Optimalkan threshold klasifikasi menggunakan PSO untuk meningkatkan performa model")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                n_particles = st.slider("Jumlah Partikel", 10, 100, 30)
            with col2:
                w = st.slider("Inertia Weight (w)", 0.1, 1.0, 0.7, 0.05)
            with col3:
                max_iter = st.slider("Iterasi", 10, 200, 50)
            
            if st.button("🚀 Jalankan Optimasi PSO"):
                with st.spinner("Mengoptimalkan threshold dengan PSO..."):
                    try:
                        # Optimasi PSO
                        pso = PSO(n_particles=n_particles, w=w, max_iter=max_iter)
                        optimal_threshold, optimal_f1 = pso.optimize(
                            y, 
                            st.session_state.nb_results['y_proba']
                        )
                        
                        # Prediksi dengan threshold optimal
                        y_pred_pso = (st.session_state.nb_results['y_proba'] >= optimal_threshold).astype(int)
                        
                        # Hitung metrik
                        report_dict_pso = classification_report(y, y_pred_pso, output_dict=True)
                        accuracy_pso = accuracy_score(y, y_pred_pso)
                        precision_pso = precision_score(y, y_pred_pso)
                        recall_pso = recall_score(y, y_pred_pso)
                        
                        # Simpan hasil
                        st.session_state.pso_results = {
                            'optimal_threshold': optimal_threshold,
                            'report': report_dict_pso,
                            'cm': confusion_matrix(y, y_pred_pso),
                            'accuracy': accuracy_pso,
                            'precision': precision_pso,
                            'recall': recall_pso,
                            'f1': optimal_f1,
                            'y_pred': y_pred_pso
                        }
                        
                        # Simpan threshold untuk prediksi
                        st.session_state.pso_threshold = optimal_threshold
                        
                        st.success(f"✅ Optimasi selesai! Threshold optimal: {optimal_threshold:.4f}")
                        
                    except Exception as e:
                        st.error(f"❌ Error dalam optimasi PSO: {str(e)}")
            
            # Tampilkan hasil PSO jika sudah ada
            if st.session_state.pso_results is not None:
                pso_results = st.session_state.pso_results
                nb_results = st.session_state.nb_results
                
                st.markdown("#### Perbandingan Hasil", unsafe_allow_html=True)
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.markdown(create_metric_card(
                        "Akurasi", 
                        pso_results['accuracy'], 
                        pso_results['accuracy'] - nb_results['accuracy']
                    ), unsafe_allow_html=True)
                with col2:
                    st.markdown(create_metric_card(
                        "Presisi", 
                        pso_results['precision'], 
                        pso_results['precision'] - nb_results['precision']
                    ), unsafe_allow_html=True)
                with col3:
                    st.markdown(create_metric_card(
                        "Recall", 
                        pso_results['recall'], 
                        pso_results['recall'] - nb_results['recall']
                    ), unsafe_allow_html=True)
                with col4:
                    st.markdown(create_metric_card(
                        "F1-Score", 
                        pso_results['f1'], 
                        pso_results['f1'] - nb_results['f1']
                    ), unsafe_allow_html=True)
                
                with st.expander("Detail Laporan PSO"):
                    st.markdown("#### Laporan Klasifikasi (PSO)", unsafe_allow_html=True)
                    st.table(pd.DataFrame(pso_results['report']).transpose())
                    
                    st.markdown("#### Confusion Matrix (PSO)", unsafe_allow_html=True)
                    cm_df_pso = pd.DataFrame(
                        pso_results['cm'],
                        index=['Aktual Tidak Aman (0)', 'Aktual Aman (1)'],
                        columns=['Prediksi Tidak Aman (0)', 'Prediksi Aman (1)']
                    )
                    st.dataframe(
                        cm_df_pso.style
                        .background_gradient(cmap='Greens')
                        .set_properties(**{'text-align': 'center'})
                    )