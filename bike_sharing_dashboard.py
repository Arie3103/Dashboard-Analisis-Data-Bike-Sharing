# Streamlit Page Configuration
import streamlit as st
st.set_page_config(page_title="Dashboard Analisis Bike Sharing", layout="wide")

# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load Dataset
@st.cache_data
def load_data():
    hour_data = pd.read_csv('hour.csv')
    hour_data['dteday'] = pd.to_datetime(hour_data['dteday'])
    return hour_data

hour_data = load_data()

# Sidebar Navigation
st.sidebar.title("Navigasi")
menu = st.sidebar.radio("Pilih Halaman:", ["Beranda", "EDA", "Clustering", "Insight"])

# Home Section
if menu == "Beranda":
    st.title("Dashboard Analisis Data Bike Sharing")
    st.write("""
    Dashboard ini dirancang untuk mengeksplorasi dan menganalisis data Bike Sharing.
    Anda dapat:
    - Melihat eksplorasi data di menu **EDA**.
    - Mengeksplorasi clustering berbasis variabel lingkungan di menu **Clustering**.
    - Melihat insight dari analisis di menu **Insight**.
    """)
    st.dataframe(hour_data.head())

# EDA Section
elif menu == "EDA":
    st.title("Eksplorasi Data (EDA)")
    
    # 1. Total Penyewaan Sepeda per Tahun
    cnt_year = hour_data.groupby('yr')['cnt'].sum().reset_index()
    cnt_year['yr'] = cnt_year['yr'].map({0: '2011', 1: '2012'})
    st.subheader("1. Total Penyewaan Sepeda per Tahun")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=cnt_year, x='yr', y='cnt', palette='muted', ax=ax)
    ax.set_title("Total Penyewaan Sepeda per Tahun")
    ax.set_xlabel("Tahun")
    ax.set_ylabel("Total Penyewaan")
    st.pyplot(fig)
    
    # 2. Total Penyewaan Sepeda Bulanan
    hour_data['month'] = hour_data['dteday'].dt.month
    cnt_month = hour_data.groupby('month')['cnt'].sum().reset_index()
    cnt_month['month'] = cnt_month['month'].map({
        1: 'Januari', 2: 'Februari', 3: 'Maret', 4: 'April',
        5: 'Mei', 6: 'Juni', 7: 'Juli', 8: 'Agustus',
        9: 'September', 10: 'Oktober', 11: 'November', 12: 'Desember'})
    st.subheader("2. Total Penyewaan Sepeda Bulanan")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=cnt_month, x='month', y='cnt', palette='viridis', ax=ax)
    ax.set_title("Total Penyewaan Sepeda Bulanan")
    ax.set_xlabel("Bulan")
    ax.set_ylabel("Total Penyewaan")
    plt.xticks(rotation=45)
    st.pyplot(fig)
    
    # 3. Penyewaan Berdasarkan Hari dalam Minggu
    hour_data['day_of_week'] = hour_data['dteday'].dt.day_name()
    cnt_day_of_week = hour_data.groupby('day_of_week')['cnt'].sum().reset_index()
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    cnt_day_of_week = cnt_day_of_week.set_index('day_of_week').reindex(day_order).reset_index()

    st.subheader("3. Penyewaan Berdasarkan Hari dalam Minggu")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=cnt_day_of_week, x='day_of_week', y='cnt', palette='coolwarm', ax=ax)
    ax.set_title("Penyewaan Berdasarkan Hari")
    ax.set_xlabel("Hari")
    ax.set_ylabel("Total Penyewaan")
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # 4. Penyewaan Berdasarkan Kondisi Cuaca
    cnt_weather = hour_data.groupby('weathersit')['cnt'].sum().reset_index()
    cnt_weather['weathersit'] = cnt_weather['weathersit'].map({
        1: 'Cerah', 2: 'Berawan', 3: 'Hujan/Salju Ringan', 4: 'Hujan/Salju Lebat'})
    st.subheader("4. Penyewaan Berdasarkan Kondisi Cuaca")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=cnt_weather, x='weathersit', y='cnt', palette='Set2', ax=ax)
    ax.set_title("Penyewaan Berdasarkan Kondisi Cuaca")
    ax.set_xlabel("Kondisi Cuaca")
    ax.set_ylabel("Total Penyewaan")
    plt.xticks(rotation=15)
    st.pyplot(fig)

    # 5. Penyewaan Berdasarkan Tipe Pengguna
    cnt_user_type = hour_data[['casual', 'registered']].sum().reset_index()
    cnt_user_type.columns = ['User Type', 'Total Rentals']
    cnt_user_type['User Type'] = cnt_user_type['User Type'].map({
        'casual': 'Kasual', 'registered': 'Terdaftar'})
    st.subheader("5. Penyewaan Berdasarkan Tipe Pengguna")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(data=cnt_user_type, x='User Type', y='Total Rentals', palette='pastel', ax=ax)
    ax.set_title("Penyewaan Berdasarkan Tipe Pengguna")
    ax.set_xlabel("Tipe Pengguna")
    ax.set_ylabel("Total Penyewaan")
    st.pyplot(fig)

    # 6. Penyewaan Berdasarkan Musim
    cnt_season = hour_data.groupby('season')['cnt'].sum().reset_index()
    cnt_season['season'] = cnt_season['season'].map({
        1: 'Musim Semi', 2: 'Musim Panas', 3: 'Musim Gugur', 4: 'Musim Dingin'})
    st.subheader("6. Penyewaan Berdasarkan Musim")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=cnt_season, x='season', y='cnt', palette='autumn', ax=ax)
    ax.set_title("Penyewaan Berdasarkan Musim")
    ax.set_xlabel("Musim")
    ax.set_ylabel("Total Penyewaan")
    st.pyplot(fig)

    # 7. Penyewaan Sepeda Bulanan per Tahun
    cnt_year_month = hour_data.groupby(['yr', 'month'])['cnt'].sum().reset_index()
    cnt_year_month['yr'] = cnt_year_month['yr'].map({0: '2011', 1: '2012'})
    cnt_year_month['month'] = cnt_year_month['month'].map({
        1: 'Januari', 2: 'Februari', 3: 'Maret', 4: 'April',
        5: 'Mei', 6: 'Juni', 7: 'Juli', 8: 'Agustus',
        9: 'September', 10: 'Oktober', 11: 'November', 12: 'Desember'})
    st.subheader("7. Penyewaan Bulanan per Tahun")
    fig, ax = plt.subplots(figsize=(14, 7))
    sns.lineplot(data=cnt_year_month, x='month', y='cnt', hue='yr', marker='o', palette='coolwarm', ax=ax)
    ax.set_title("Penyewaan Bulanan per Tahun (2011 vs 2012)")
    ax.set_xlabel("Bulan")
    ax.set_ylabel("Total Penyewaan")
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # 8. Penyewaan: Hari Kerja vs Libur
    cnt_working_holiday = hour_data.groupby('workingday')['cnt'].sum().reset_index()
    cnt_working_holiday['workingday'] = cnt_working_holiday['workingday'].map({
        0: 'Hari Libur', 1: 'Hari Kerja'})
    st.subheader("8. Penyewaan: Hari Kerja vs Hari Libur")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(data=cnt_working_holiday, x='workingday', y='cnt', palette='coolwarm', ax=ax)
    ax.set_title("Penyewaan Berdasarkan Hari Kerja vs Libur")
    ax.set_xlabel("Tipe Hari")
    ax.set_ylabel("Total Penyewaan")
    st.pyplot(fig)

# Clustering Section
elif menu == "Clustering":
    st.title("Analisis Clustering")

    # Standarisasi Data
    env_features = hour_data[['temp', 'hum', 'windspeed', 'cnt']]
    scaler = StandardScaler()
    scaled_env_features = scaler.fit_transform(env_features)

    # Elbow Method
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, random_state=42)
        kmeans.fit(scaled_env_features)
        wcss.append(kmeans.inertia_)
    
    st.subheader("Metode Elbow untuk Klaster Optimal")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(1, 11), wcss, marker='o')
    ax.set_title("Metode Elbow")
    ax.set_xlabel("Jumlah Klaster")
    ax.set_ylabel("WCSS")
    st.pyplot(fig)
# Terapkan KMeans dengan 3 Klaster
    kmeans_env = KMeans(n_clusters=3, random_state=42)
    hour_data['env_cluster'] = kmeans_env.fit_predict(scaled_env_features)

    # Visualisasi Klaster
    st.subheader("Visualisasi Klaster Berdasarkan Variabel Lingkungan")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=hour_data, x='temp', y='cnt', hue='env_cluster', palette='viridis', ax=ax)
    ax.set_title("Klaster Berdasarkan Variabel Lingkungan")
    ax.set_xlabel("Suhu (temp)")
    ax.set_ylabel("Total Penyewaan")
    st.pyplot(fig)

# Insight Section
elif menu == "Insight":
    st.title("Insight dari Analisis Data")

    st.write("""
### **Ringkasan Temuan**:

#### **1. Total Penyewaan Sepeda per Tahun**:
- Penyewaan sepeda meningkat secara signifikan pada tahun 2012 dibandingkan 2011. Hal ini menunjukkan bahwa layanan bike-sharing semakin populer di tahun kedua operasionalnya.

#### **2. Total Penyewaan Sepeda Bulanan**:
- Jumlah penyewaan sepeda mencapai puncaknya pada bulan Juli dan Agustus (musim panas).
- Bulan Desember dan Januari (musim dingin) memiliki jumlah penyewaan terendah.

#### **3. Penyewaan Berdasarkan Hari dalam Minggu**:
- Hari kerja memiliki jumlah penyewaan yang lebih tinggi dibandingkan akhir pekan. Hal ini menunjukkan bahwa sepeda lebih banyak digunakan untuk aktivitas sehari-hari seperti perjalanan ke tempat kerja atau sekolah.

#### **4. Penyewaan Berdasarkan Kondisi Cuaca**:
- Penyewaan sepeda paling banyak terjadi dalam kondisi cuaca cerah (kategori 1).
- Penyewaan menurun signifikan pada kondisi cuaca buruk seperti hujan ringan atau deras.

#### **5. Penyewaan Berdasarkan Tipe Pengguna**:
- Pengguna terdaftar menyumbang mayoritas penyewaan sepeda dibandingkan pengguna kasual.
- Pengguna kasual cenderung lebih aktif pada akhir pekan dan musim panas.

#### **6. Penyewaan Berdasarkan Musim**:
- Musim panas dan musim gugur memiliki jumlah penyewaan tertinggi, menunjukkan pengaruh suhu hangat dan cuaca nyaman terhadap penggunaan sepeda.
- Musim dingin memiliki jumlah penyewaan terendah, kemungkinan besar karena suhu rendah dan cuaca ekstrem.

#### **7. Penyewaan Bulanan per Tahun**:
- Pola penyewaan serupa terlihat di tahun 2011 dan 2012, tetapi jumlah penyewaan pada tahun 2012 secara konsisten lebih tinggi di setiap bulan.
- Peningkatan terbesar terjadi pada bulan Januari hingga Maret, yang mungkin mencerminkan promosi atau perbaikan layanan.

#### **8. Penyewaan Berdasarkan Hari Kerja vs Hari Libur**:
- Penyewaan lebih tinggi pada hari kerja dibandingkan hari libur, mencerminkan penggunaan sepeda untuk kebutuhan sehari-hari.

#### **9. Hasil Clustering Berdasarkan Variabel Lingkungan**:
- **Cluster 2**: Kondisi suhu tinggi, kelembapan sedang, dan kecepatan angin sedang memiliki jumlah penyewaan tertinggi.
- **Cluster 0**: Kondisi suhu sedang, kelembapan tinggi, dan kecepatan angin rendah memiliki jumlah penyewaan terendah.
- **Cluster 1**: Kondisi suhu rendah, kelembapan sedang, dan kecepatan angin tinggi memiliki jumlah penyewaan sedang.
- Clustering ini membantu memahami bagaimana variabel lingkungan memengaruhi pola penggunaan sepeda.
""")

    st.write("""
    ### **Rekomendasi Operasional**:
    1. **Distribusi Sepeda**:
       - Tingkatkan distribusi sepeda selama bulan-bulan musim panas dan musim gugur, karena permintaan lebih tinggi pada musim tersebut.
       - Perbanyak sepeda pada hari kerja, terutama di lokasi yang digunakan untuk perjalanan sehari-hari seperti area perkantoran atau kampus.

    2. **Promosi**:
       - Fokuskan promosi pada pengguna kasual selama akhir pekan dan musim panas untuk meningkatkan jumlah penyewaan.
       - Berikan insentif atau diskon selama musim dingin untuk mendorong penyewaan pada periode permintaan rendah.

    3. **Peningkatan Layanan**:
       - Pertimbangkan untuk menambahkan fasilitas seperti perlindungan terhadap hujan atau kelembapan untuk meningkatkan penggunaan selama cuaca buruk.
       - Sediakan sepeda tambahan di area dengan suhu tinggi (Cluster 2), karena memiliki permintaan tertinggi.

    4. **Perencanaan Strategis**:
       - Gunakan hasil clustering untuk memprediksi pola permintaan berdasarkan kondisi lingkungan, sehingga alokasi sepeda dapat lebih efisien.
       - Lakukan kampanye edukasi untuk meningkatkan kesadaran masyarakat tentang manfaat bersepeda di segala kondisi cuaca.
    """)
