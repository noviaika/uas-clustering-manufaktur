import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
import scipy.cluster.hierarchy as sch

# ============================================================
# KONFIGURASI HALAMAN
# ============================================================
st.set_page_config(
    page_title="Analisis Clustering Cacat Produk Manufaktur",
    page_icon="🏭",
    layout="wide"
)

sns.set_theme(style="whitegrid")

# ============================================================
# JUDUL & DESKRIPSI APLIKASI
# ============================================================
st.title("🏭 Analisis Clustering Cacat Produk Industri Manufaktur")
st.markdown("""
Aplikasi ini melakukan **segmentasi cacat produk manufaktur** menggunakan dua pendekatan
*Unsupervised Learning*: **K-Means Clustering** dan **Hierarchical Clustering (Agglomerative)**,
berdasarkan kombinasi **tingkat keparahan cacat (severity)** dan **biaya perbaikan (repair cost)**.

Tujuannya adalah membantu tim *Quality Assurance* (QA) memprioritaskan segmen cacat mana yang
paling merugikan secara finansial, sehingga perbaikan proses produksi bisa lebih tepat sasaran.
""")

st.caption("📌 Disusun oleh: **[NOVIA IKA SAFITRI]** — NIM: **[E12.2024.01936]**")

# ============================================================
# 1. LOAD DATA
# ============================================================
st.sidebar.header("⚙️ Pengaturan")
uploaded_file = st.sidebar.file_uploader("Upload dataset (opsional)", type=["csv"])

@st.cache_data
def load_default_data():
    return pd.read_csv("defects_data.csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("Dataset kustom berhasil dimuat.")
else:
    df = load_default_data()
    st.sidebar.info("Menggunakan dataset bawaan: defects_data.csv")

# ============================================================
# 2. DATA UNDERSTANDING
# ============================================================
st.header("1. Data Understanding (Pemahaman Data)")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Jumlah Rekaman", f"{len(df):,}")
col2.metric("Total Biaya Perbaikan", f"${df['repair_cost'].sum():,.0f}")
col3.metric("Rata-rata Biaya Perbaikan", f"${df['repair_cost'].mean():,.2f}")
col4.metric("Jenis Cacat Terbanyak", df['defect_type'].mode()[0])

tab1, tab2, tab3 = st.tabs(["📋 Data Mentah", "📊 Statistik Deskriptif", "📈 Distribusi Kategori"])

with tab1:
    st.dataframe(df.head(20), use_container_width=True)
    st.caption(f"Menampilkan 20 dari {len(df)} baris data.")

with tab2:
    st.dataframe(df.describe(), use_container_width=True)

with tab3:
    dcol1, dcol2 = st.columns(2)
    with dcol1:
        fig, ax = plt.subplots(figsize=(6, 4))
        df['severity'].value_counts().reindex(['Minor', 'Moderate', 'Critical']).plot(
            kind='bar', color=['#4C72B0', '#DD8452', '#C44E52'], ax=ax
        )
        ax.set_title("Distribusi Tingkat Keparahan (Severity)")
        ax.set_xlabel("Severity")
        ax.set_ylabel("Jumlah Kasus")
        plt.xticks(rotation=0)
        st.pyplot(fig)
    with dcol2:
        fig, ax = plt.subplots(figsize=(6, 4))
        df['defect_type'].value_counts().plot(kind='bar', color='#55A868', ax=ax)
        ax.set_title("Distribusi Jenis Cacat (Defect Type)")
        ax.set_xlabel("Defect Type")
        ax.set_ylabel("Jumlah Kasus")
        plt.xticks(rotation=0)
        st.pyplot(fig)

# ============================================================
# 3. PREPROCESSING
# ============================================================
st.header("2. Data Preprocessing")
st.markdown("""
Dua langkah utama dilakukan sebelum pemodelan:
1. **Ordinal Encoding** — mengubah `severity` (Minor, Moderate, Critical) menjadi skor numerik bertingkat (1, 2, 3).
2. **Feature Scaling (StandardScaler)** — menyamakan skala `repair_cost` dan `severity_score` agar tidak ada
   fitur yang mendominasi perhitungan jarak antar titik data.
""")

severity_mapping = {'Minor': 1, 'Moderate': 2, 'Critical': 3}
df['severity_score'] = df['severity'].map(severity_mapping)

X = df[['repair_cost', 'severity_score']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

with st.expander("Lihat contoh hasil preprocessing"):
    preview = df[['repair_cost', 'severity', 'severity_score']].head(10).copy()
    st.dataframe(preview, use_container_width=True)

# ============================================================
# 4. K-MEANS - ELBOW METHOD
# ============================================================
st.header("3. K-Means Clustering")
st.subheader("3.1 Penentuan Jumlah Klaster Optimal (Elbow Method)")

wcss = []
k_range = range(1, 11)
for k in k_range:
    kmeans_temp = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
    kmeans_temp.fit(X_scaled)
    wcss.append(kmeans_temp.inertia_)

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(list(k_range), wcss, marker='o', linestyle='--', color='darkblue', linewidth=2, markersize=8)
ax.set_title('Elbow Method untuk Penentuan Jumlah Klaster (K) Optimal', fontweight='bold')
ax.set_xlabel('Jumlah Klaster (K)')
ax.set_ylabel('WCSS (Inersia)')
ax.set_xticks(list(k_range))
ax.grid(True, linestyle=':')
st.pyplot(fig)

st.info("""
**Interpretasi:** Grafik menunjukkan penurunan WCSS yang melambat drastis pada **K = 3**,
membentuk pola "siku" (elbow). Titik ini dipilih sebagai jumlah klaster optimal karena penambahan
klaster setelah titik ini hanya memberikan penurunan WCSS yang kecil (diminishing returns).
""")

optimal_k = st.sidebar.slider("Jumlah Klaster K-Means (K)", min_value=2, max_value=6, value=3)

# ============================================================
# 5. K-MEANS - MODELING
# ============================================================
st.subheader("3.2 Hasil K-Means Clustering")

kmeans_model = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42, n_init=10)
df['cluster_kmeans'] = kmeans_model.fit_predict(X_scaled)

kcol1, kcol2 = st.columns([2, 1])
with kcol1:
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.scatterplot(
        x=df['repair_cost'], y=df['severity_score'], hue=df['cluster_kmeans'],
        palette='Set1', s=90, alpha=0.85, edgecolor='black', linewidth=0.5, ax=ax
    )
    ax.set_title(f'Segmentasi Cacat Manufaktur - K-Means (K={optimal_k})', fontweight='bold')
    ax.set_xlabel('Biaya Perbaikan Produk ($)')
    ax.set_ylabel('Tingkat Keparahan Cacat (Severity Score)')
    ax.set_yticks([1, 2, 3])
    ax.set_yticklabels(['1 (Minor)', '2 (Moderate)', '3 (Critical)'])
    ax.legend(title='Klaster', loc='upper left')
    st.pyplot(fig)

with kcol2:
    st.markdown("**Jumlah Sampel per Klaster**")
    st.dataframe(df['cluster_kmeans'].value_counts().sort_index().rename("Jumlah"), use_container_width=True)

profil_kmeans = df.groupby('cluster_kmeans')[['repair_cost', 'severity_score']].mean().round(2)
st.markdown("**Profil Rata-rata Tiap Klaster (K-Means)**")
st.dataframe(profil_kmeans, use_container_width=True)

# ============================================================
# 6. HIERARCHICAL CLUSTERING - DENDROGRAM
# ============================================================
st.header("4. Hierarchical Clustering (Agglomerative)")
st.subheader("4.1 Dendrogram")

fig, ax = plt.subplots(figsize=(11, 6))
linkage_matrix = sch.linkage(X_scaled, method='ward')
sch.dendrogram(linkage_matrix, ax=ax, no_labels=True)
ax.set_title('Dendrogram Struktur Hierarki Kasus Cacat Manufaktur (Metode Ward)', fontweight='bold')
ax.set_xlabel('Indeks Sampel Data Cacat')
ax.set_ylabel('Jarak Ketidakmiripan Euclidean (Linkage Distance)')
st.pyplot(fig)

st.info("""
**Interpretasi:** Jumlah klaster optimal ditentukan dengan mencari garis vertikal terpanjang yang
tidak terpotong oleh penggabungan lain. Dendrogram di atas menunjukkan struktur yang terbagi
secara alami menjadi **3 cabang utama**, sehingga `n_clusters=3` dipilih untuk pemodelan Agglomerative.
""")

# ============================================================
# 7. HIERARCHICAL - MODELING
# ============================================================
st.subheader("4.2 Hasil Hierarchical Clustering")

hc_model = AgglomerativeClustering(n_clusters=optimal_k, metric='euclidean', linkage='ward')
df['cluster_hierarchy'] = hc_model.fit_predict(X_scaled)

hcol1, hcol2 = st.columns([2, 1])
with hcol1:
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.scatterplot(
        x=df['repair_cost'], y=df['severity_score'], hue=df['cluster_hierarchy'],
        palette='Set2', s=90, alpha=0.85, edgecolor='black', linewidth=0.5, ax=ax
    )
    ax.set_title('Segmentasi Cacat Manufaktur - Hierarchical Clustering', fontweight='bold')
    ax.set_xlabel('Biaya Perbaikan Produk ($)')
    ax.set_ylabel('Tingkat Keparahan Cacat (Severity Score)')
    ax.set_yticks([1, 2, 3])
    ax.set_yticklabels(['1 (Minor)', '2 (Moderate)', '3 (Critical)'])
    ax.legend(title='Klaster', loc='upper left')
    st.pyplot(fig)

with hcol2:
    st.markdown("**Jumlah Sampel per Klaster**")
    st.dataframe(df['cluster_hierarchy'].value_counts().sort_index().rename("Jumlah"), use_container_width=True)

profil_hc = df.groupby('cluster_hierarchy')[['repair_cost', 'severity_score']].mean().round(2)
st.markdown("**Profil Rata-rata Tiap Klaster (Hierarchical)**")
st.dataframe(profil_hc, use_container_width=True)

# ============================================================
# 7B. PERBANDINGAN K-MEANS VS HIERARCHICAL
# ============================================================
st.header("5. Perbandingan Hasil K-Means vs Hierarchical Clustering")

crosstab = pd.crosstab(df['cluster_kmeans'], df['cluster_hierarchy'])
crosstab.index.name = "K-Means"
crosstab.columns.name = "Hierarchical"

ccol1, ccol2 = st.columns([1, 1.3])
with ccol1:
    st.markdown("**Tabel Silang (Crosstab) Label Klaster**")
    st.dataframe(crosstab, use_container_width=True)
    jumlah_beda = (df['cluster_kmeans'] != df['cluster_hierarchy']).sum()
    st.metric("Jumlah data dengan label klaster berbeda", f"{jumlah_beda} dari {len(df)}")

with ccol2:
    st.markdown("**Mengapa hasilnya bisa berbeda?**")
    st.markdown("""
- **K-Means** membagi data berdasarkan jarak ke titik pusat (*centroid*) yang dioptimalkan
  secara iteratif — cenderung menghasilkan klaster berbentuk bulat/konveks dengan ukuran mirip.
- **Hierarchical Clustering (Ward)** membangun kelompok secara bertahap dari bawah ke atas
  berdasarkan penggabungan pasangan titik/klaster terdekat — lebih sensitif terhadap struktur
  bertingkat dan tidak mengasumsikan bentuk klaster tertentu.
- Perbedaan pendekatan dasar ini membuat titik-titik yang berada **di perbatasan antar klaster**
  berpotensi mendapat label kelompok yang berbeda di antara dua metode, meskipun pola besar
  (kelompok berisiko tinggi vs rendah) tetap konsisten di keduanya.
""")

# ============================================================
# 8. INSIGHT BISNIS & REKOMENDASI
# ============================================================
st.header("6. Interpretasi Hasil & Insight Bisnis")

# Urutkan klaster berdasarkan rata-rata repair_cost agar penamaan konsisten
sorted_clusters = profil_kmeans.sort_values('repair_cost', ascending=False)
labels_order = list(sorted_clusters.index)

nama_segmen = ["🔴 Risiko Finansial Tinggi", "🟡 Risiko Menengah", "🟢 Risiko Rendah / Efisien"]
rekomendasi = [
    "Klaster dengan biaya perbaikan dan/atau tingkat keparahan tertinggi. "
    "Prioritaskan **Automated Testing** dan audit lini produksi secara intensif pada segmen ini "
    "untuk mencegah cacat lolos ke tangan konsumen.",
    "Klaster dengan karakteristik menengah antara biaya dan keparahan. "
    "Disarankan pengawasan berkala dengan kombinasi **Manual Testing** dan evaluasi proses berulang.",
    "Klaster dengan biaya perbaikan rendah dan tingkat keparahan minor. "
    "Cukup ditangani dengan **inspeksi visual rutin** tanpa perlu investasi biaya besar."
]

for i, cl in enumerate(labels_order[:min(len(labels_order), 3)]):
    row = profil_kmeans.loc[cl]
    st.markdown(f"""
**{nama_segmen[i] if i < len(nama_segmen) else f'Klaster {cl}'} (Klaster {cl})**
- Rata-rata biaya perbaikan: **${row['repair_cost']:.2f}**
- Rata-rata skor keparahan: **{row['severity_score']:.2f}** (skala 1–3)
- Jumlah kasus: **{int(df[df['cluster_kmeans']==cl].shape[0])}**
- 💡 Rekomendasi: {rekomendasi[i] if i < len(rekomendasi) else '-'}
""")

st.success("""
**Kesimpulan Umum:** Segmentasi ini memungkinkan tim QA & manajemen untuk mengalokasikan sumber daya
perbaikan proses secara lebih efisien — fokus pada klaster berisiko finansial tinggi terlebih dahulu,
alih-alih menangani seluruh laporan cacat secara seragam.
""")

st.divider()
st.subheader("📥 Unduh Hasil Clustering")
hasil_download = df[['defect_id', 'product_id', 'defect_type', 'severity', 'repair_cost',
                      'cluster_kmeans', 'cluster_hierarchy']]
csv_bytes = hasil_download.to_csv(index=False).encode('utf-8')
st.download_button(
    label="Download CSV Hasil Clustering",
    data=csv_bytes,
    file_name="hasil_clustering_cacat_manufaktur.csv",
    mime="text/csv"
)

st.caption("Dibuat untuk UAS Project Kecerdasan Buatan — Deployment & Streamlit Application")
