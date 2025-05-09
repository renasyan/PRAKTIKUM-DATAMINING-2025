import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import roc_curve, auc
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Dashboard Prediksi Diabetes",
    page_icon="🩺",
    layout="wide"
)

def clear_outlier(df, kolom_numerik, metode='iqr', threshold=1.5):
    # Metode IQR (Interquartile Range)
    Q1 = df[kolom_numerik].quantile(0.25)
    Q3 = df[kolom_numerik].quantile(0.75)
    IQR = Q3 - Q1

    batas_bawah = Q1 - threshold * IQR
    batas_atas = Q3 + threshold * IQR

    return df[(df[kolom_numerik] >= batas_bawah) & (df[kolom_numerik] <= batas_atas)]

# Fungsi untuk memuat data
@st.cache_data
def load_data():
    df = pd.read_csv('Naive-Bayes-Classification-Data.csv')
    # Membersihkan outlier
    kolom_numerik = ['glucose', 'bloodpressure']
    for kolom in kolom_numerik:
        df = clear_outlier(df, kolom)
    df.dropna(inplace=True)
    return df

# Fungsi untuk memuat model
@st.cache_resource
def load_model():
    return joblib.load('model.pkl')

# Fungsi untuk membuat grafik ROC
def plot_roc_curve(model, X, y):
    y_prob = model.predict_proba(X)[:, 1]

    # Menghitung nilai ROC
    fpr, tpr, thresholds = roc_curve(y, y_prob)
    roc_auc = auc(fpr, tpr)

    # Membuat grafik ROC
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=fpr, 
            y=tpr,
            mode='lines',
            line=dict(color='#e74c3c', width=2),
            name=f'ROC Curve (AUC = {roc_auc:.2f})'
        )
    )

    # Menambahkan garis diagonal
    fig.add_trace(
        go.Scatter(
            x=[0, 1], 
            y=[0, 1],
            mode='lines',
            line=dict(color='navy', width=2, dash='dash'),
            name='Random Chance'
        )
    )

    fig.update_layout(
        title='Kurva ROC',
        xaxis=dict(title='False Positive Rate'),
        yaxis=dict(title='True Positive Rate'),
        legend=dict(x=0.7, y=0.05),
        width=700,
        height=500
    )

    return fig

# Fungsi untuk memberikan rekomendasi berdasarkan hasil prediksi
def get_recommendation(prediction, probability, glucose, bloodpressure):
    if prediction == 1:
        if probability > 0.8:
            return """
            ### Rekomendasi Tindakan: Risiko Tinggi Diabetes

            Berdasarkan hasil prediksi, Anda memiliki risiko tinggi diabetes. Rekomendasi tindakan:

            1. **Segera Konsultasi dengan Dokter**: Lakukan pemeriksaan gula darah lengkap
            2. **Evaluasi Pola Makan**: Kurangi konsumsi karbohidrat sederhana dan gula
            3. **Tingkatkan Aktivitas Fisik**: Lakukan olahraga teratur minimal 150 menit per minggu
            4. **Pantau Tekanan Darah**: Lakukan pemeriksaan tekanan darah secara rutin
            5. **Jaga Berat Badan Ideal**: Turunkan berat badan jika berlebih
            """
        else:
            return """
            ### Rekomendasi Tindakan: Risiko Sedang Diabetes

            Berdasarkan hasil prediksi, Anda memiliki risiko sedang diabetes. Rekomendasi tindakan:

            1. **Konsultasi dengan Dokter**: Lakukan pemeriksaan gula darah
            2. **Perhatikan Pola Makan**: Batasi konsumsi makanan tinggi gula dan karbohidrat olahan
            3. **Rutin Berolahraga**: Minimal 30 menit per hari, 5 kali seminggu
            4. **Pantau Tekanan Darah**: Pertahankan tekanan darah normal
            5. **Hindari Stres Berlebih**: Kelola stres dengan baik
            """
    else:
        if glucose > 50 or bloodpressure > 80:
            return """
            ### Rekomendasi Tindakan: Risiko Rendah Diabetes

            Berdasarkan hasil prediksi, Anda memiliki risiko rendah diabetes. Namun, beberapa nilai parameter Anda perlu diperhatikan. Rekomendasi tindakan:

            1. **Pemeriksaan Rutin**: Lakukan pemeriksaan gula darah setahun sekali
            2. **Pertahankan Pola Makan Sehat**: Konsumsi makanan seimbang dengan banyak serat
            3. **Jaga Aktivitas Fisik**: Tetap aktif secara fisik
            4. **Pantau Tekanan Darah**: Periksa tekanan darah secara berkala
            """
        else:
            return """
            ### Rekomendasi Tindakan: Risiko Sangat Rendah Diabetes

            Berdasarkan hasil prediksi, Anda memiliki risiko sangat rendah diabetes. Rekomendasi tindakan:

            1. **Pertahankan Gaya Hidup Sehat**: Lanjutkan pola makan seimbang dan aktivitas fisik
            2. **Pemeriksaan Berkala**: Lakukan pemeriksaan kesehatan umum secara berkala
            3. **Edukasi Diri**: Tetap update informasi tentang pencegahan diabetes
            """

# Memuat data dan model
df = load_data()
model = load_model()

# Header
st.title("Dashboard Prediksi Diabetes Berdasarkan Tekanan Darah dan Kadar Glukosa")

# Overview Dashboardnya
st.markdown("""
## Selamat Datang di Dashboard Prediksi Diabetes

Dashboard ini menggunakan model Naive Bayes untuk memprediksi kemungkinan diabetes berdasarkan kadar glukosa dan tekanan darah.

### Dataset:
Dataset berisi informasi tentang kadar glukosa, tekanan darah, dan status diabetes (0: Tidak diabetes, 1: Diabetes).
""")

# Tampilkan beberapa statistik dasar
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Jumlah Data", df.shape[0])
with col2:
    st.metric("Pasien Diabetes", df[df['diabetes'] == 1].shape[0])
with col3:
    st.metric("Pasien Non-Diabetes", df[df['diabetes'] == 0].shape[0])

# Distribusi kelas
st.subheader("Distribusi Status Diabetes dalam Dataset")
fig = px.pie(df, names='diabetes', title='Distribusi Status Diabetes', 
             color_discrete_sequence=['#3498db', '#e74c3c'],
             labels={'0':'Tidak Diabetes', '1':'Diabetes'})
st.plotly_chart(fig)

# Form untuk input data baru
st.markdown("---")
st.subheader("Prediksi Diabetes untuk Data Baru")

col1, col2 = st.columns(2)
with col1:
    glucose = st.number_input("Kadar Glukosa", min_value=int(df['glucose'].min()), max_value=int(df['glucose'].max()), value=int(df['glucose'].mean()))
with col2:
    bloodpressure = st.number_input("Tekanan Darah", min_value=int(df['bloodpressure'].min()), max_value=int(df['bloodpressure'].max()), value=int(df['bloodpressure'].mean()))

# Prediksi
if st.button("Prediksi", type="primary"):
    input_data = pd.DataFrame([[glucose, bloodpressure]], columns=['glucose', 'bloodpressure'])
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]

    # Tampilkan hasil
    st.markdown("---")
    st.subheader("Hasil Prediksi")

    if prediction == 1:
        st.error(f"Prediksi: **Diabetes** dengan probabilitas {probability[1]:.2%}")
    else:
        st.success(f"Prediksi: **Tidak Diabetes** dengan probabilitas {probability[0]:.2%}")

    # Visualisasi probabilitas
    col1, col2 = st.columns(2)

    with col1:
        # Grafik probabilitas
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=['Tidak Diabetes', 'Diabetes'],
            y=[probability[0], probability[1]],
            marker_color=['#3498db', '#e74c3c']
        ))
        fig.update_layout(
            title='Probabilitas Prediksi',
            yaxis=dict(title='Probabilitas', range=[0, 1]),
            xaxis=dict(title='Kelas')
        )
        st.plotly_chart(fig)

    with col2:
        # Grafik ROC
        X = df.drop('diabetes', axis=1)
        y = df['diabetes']
        roc_fig = plot_roc_curve(model, X, y)
        st.plotly_chart(roc_fig)

    # Tampilkan rekomendasi
    st.markdown(get_recommendation(prediction, probability[prediction], glucose, bloodpressure))

# Footer
st.markdown("---")
