import numpy as np
import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
import plotly.express as px
import altair as alt
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

st.set_option('deprecation.showPyplotGlobalUse', False)

st.write("""
Uswatun Khasanah - 2024
""")

# Sidebar untuk navigasi
with st.sidebar:
    selected = option_menu('Menu',
                           ['Home',
                            'Data Visualization',
                            'Clustering',
                            'Predict'],

                            icons = ['house-fill', 
                                     'database-fill',
                                     'lightning-fill',
                                     'stars'],
                            default_index = 0)


### Home

if selected == 'Home':

    st.subheader('Menganalisis Kepuasan Penumpang Maskapai Penerbangan')

    #page tittle
    # Menampilkan gambar dari file lokal
    from PIL import Image
    image = Image.open('dash.jpg')
    st.image(image, caption='')
    
    st.markdown("""
                Situasi bisnis yang mendasari analisis ini terjadi karena resiko kehilangan 
                penumpang yang signifikan terhadap maskapai penerbangan perusahaan, 
                dengan lebih dari separuh pelanggan melaporkan ketidakpuasan dengan layanan maskapai.
                Melihat risiko yang mengancam, langkah-langkah yang tepat sangat penting untuk dilakukan
                guna mengatasi masalah ini. Dengan fokus pada pemahaman mendalam
                terhadap pengalaman pelanggan, tujuan saya adalah tidak hanya mempertahankan
                kepercayaan dan kepuasan pelanggan, tetapi juga meminimalisir potensi kerugian
                yang dapat timbul akibat kehilangan penumpang. Hal ini menunjukkan perlunya tindakan
                yang tepat untuk mengatasi masalah ini guna mempertahankan kepercayaan dan kepuasan
                pelanggan serta meminimalisir potensi kerugian yang bisa timbul akibat kehilangan penumpang.
                Oleh karena itu, penting bagi kita untuk memahami faktor-faktor yang terlibat dalam
                kepuasan penumpang maskapai penerbangan
            """)
    
    st.write('##### DATASET AWAL')
    # Read data
    data = pd.read_csv('train.csv')
    st.write (data)
    st.markdown('Dataset ini akan melewati serangkaian langkah untuk membersihkan, menyiapkan, dan memvalidasi data agar akurat, relevan, dan representatif. Tujuannya adalah untuk mengurangi kesalahan dan bias dalam proses analisis.')

    st.write('##### DATASET AKHIR')
    df = pd.read_csv('DataCleaned.csv')
    st.write (df)
    st.markdown('Dataset ini telah dimodifikasi untuk disesuaikan dengan keperluan analisis. Proses ini melibatkan transformasi variabel dalam dataset ke format yang dapat diolah oleh algoritma yang akan digunakan, contohnya seperti mengubah variabel kategorikal menjadi numerik.')

######################### VISUALISASI 1 ######################### 
    ### Visualisasi 1 : Hubungan Satisfaction Dengan Inflight wifi service
    st.write('##### Hubungan Antara Satisfaction Dengan Inflight wifi service')
    # create graph
    plt.figure(figsize=(20, 10))
    sns.countplot(x="Inflight wifi service", data=df, hue="satisfaction", palette="Paired")

    # format graph
    plt.title("Satisfaction results by Inflight wifi service")
    sns.despine(top=True, right=True, left=False, bottom=False)

     # Update legend labels
    plt.legend(title="satisfaction", labels=['Neutral or Dissatisfied', 'Satisfied'], loc='center left', bbox_to_anchor=(1, 0.5))

    # Display graph using Streamlit
    st.pyplot(plt)
    st.markdown('''
            **Interpretasi:**
            
            Pada diagram tersebut menunjukkan hasil survei tentang kepuasan penumpang terhadap layanan wifi inflight. 
            Survei tersebut dilakukan terhadap 30.000 penumpang. 
            Hasilnya menunjukkan bahwa 89% penumpang puas dengan layanan wifi inflight, sedangkan 11% tidak puas.

            **Insight:**
            - Layanan wifi inflight umumnya diterima dengan baik oleh penumpang.
            - Masih ada sebagian kecil penumpang yang tidak puas dengan layanan wifi inflight.
            
            **Actionable Insight:**
                
            Maskapai penerbangan dapat mempertahankan tingkat kepuasan penumpang yang tinggi dengan terus meningkatkan kualitas layanan wifi inflight. 
            Hal ini dapat dilakukan dengan cara meningkatkan kecepatan wifi, memperluas jangkauan wifi, dan meningkatkan stabilitas wifi.
            ''')



######################### VISUALISASI 2 ######################### 
    ### Visualisasi 2 : Hubungan Satisfaction Dengan Ease of Online booking
    st.write('##### Hubungan Antara Satisfaction Dengan Ease of Online booking')
    # create graph
    plt.figure(figsize=(20, 10))
    sns.countplot(x="Ease of Online booking", data=df, hue="satisfaction", palette="Paired")

    # format graph
    plt.title("Satisfaction results by Ease of Online booking")
    sns.despine(top=True, right=True, left=False, bottom=False)

    # Update legend labels
    plt.legend(title="satisfaction", labels=['Neutral or Dissatisfied', 'Satisfied'], loc='center left', bbox_to_anchor=(1, 0.5))

    # Display graph using Streamlit
    st.pyplot(plt)
    st.markdown('''
            **Interpretasi:**
                
            Gambar tersebut menunjukkan hasil survei tentang kepuasan penumpang terhadap kemudahan pemesanan online. 
            Survei tersebut dilakukan terhadap 30.000 penumpang. 
            Hasilnya menunjukkan bahwa 78% penumpang puas dengan kemudahan pemesanan online, sedangkan 22% tidak puas.

            **Insight:**
            
            - Kemudahan pemesanan online umumnya diterima dengan baik oleh penumpang.
            - Masih ada sebagian kecil penumpang yang tidak puas dengan kemudahan pemesanan online.
            
            **Actionable Insight:**
                
            Maskapai penerbangan dapat menawarkan berbagai pilihan pemesanan online untuk memenuhi kebutuhan penumpang yang berbeda-beda. 
            Hal ini dapat dilakukan dengan cara menyediakan situs web pemesanan online, aplikasi pemesanan online, dan layanan pemesanan online melalui telepon.
            ''')
    
######################### VISUALISASI 3 ######################### 
    ### Visualisasi 3 : Hubungan Satisfaction Dengan Food and drink
    st.write('##### Hubungan Antara Satisfaction Dengan Food and drink')
    # create graph
    plt.figure(figsize=(20, 10))
    sns.countplot(x="Food and drink", data=df, hue="satisfaction", palette="Paired")

    # format graph
    plt.title("Satisfaction results by Food and drink")
    sns.despine(top=True, right=True, left=False, bottom=False)

    # Update legend labels
    plt.legend(title="satisfaction", labels=['Neutral or Dissatisfied', 'Satisfied'], loc='center left', bbox_to_anchor=(1, 0.5))

    # Display graph using Streamlit
    st.pyplot(plt)
    st.markdown('''
            **Interpretasi:**
            
            Berdasarkan diagram tersebut, dapat dilihat bahwa terdapat hubungan positif antara kepuasan penumpang dengan kualitas makanan dan minuman. Semakin tinggi kualitas makanan dan minuman, semakin tinggi pula tingkat kepuasan penumpang. 
            Hal ini ditunjukkan dengan grafik yang menunjukkan tren naik dari kiri ke kanan.

            **Insight:**
            
            - Mayoritas penumpang puas dengan kualitas makanan dan minuman yang disajikan di pesawat.
            - Kualitas makanan dan minuman merupakan faktor penting yang dapat memengaruhi tingkat kepuasan penumpang.
            - Maskapai penerbangan perlu terus meningkatkan kualitas makanan dan minuman untuk mempertahankan tingkat kepuasan penumpang yang tinggi.
            
            **Actionable Insight:**
                
            Maskapai penerbangan dapat bekerja sama dengan katering penerbangan yang berkualitas, maskapai dapat memastikan bahwa makanan dan minuman yang disajikan tidak hanya lezat tetapi juga bergizi. 
            Penawaran berbagai pilihan makanan dan minuman dapat memenuhi kebutuhan yang beragam dari penumpang. 
            Selain itu, melatih kru kabin untuk memberikan layanan yang ramah dan efisien saat memesan dan menikmati makanan dan minuman juga dapat meningkatkan keseluruhan pengalaman penerbangan bagi penumpang.
            ''')


######################### VISUALISASI 4 ######################### 
    ### Visualisasi 4 : Hubungan Satisfaction Dengan Online boarding
    st.write('##### Hubungan Antara Satisfaction Dengan Online boarding')
    # create graph
    plt.figure(figsize=(20, 10))
    sns.countplot(x="Online boarding", data=df, hue="satisfaction", palette="Paired")

    # format graph
    plt.title("Satisfaction results by Online boarding")
    sns.despine(top=True, right=True, left=False, bottom=False)

    # Update legend labels
    plt.legend(title="satisfaction", labels=['Neutral or Dissatisfied', 'Satisfied'], loc='center left', bbox_to_anchor=(1, 0.5))

    # Display graph using Streamlit
    st.pyplot(plt)
    st.markdown('''
            **Interpretasi:**
                
            Berdasarkan diagram tersebut, dapat dilihat bahwa terdapat hubungan positif antara kepuasan penumpang dengan layanan boarding online. 
            Semakin mudah dan nyaman layanan boarding online, semakin tinggi pula tingkat kepuasan penumpang. 
            Hal ini ditunjukkan dengan grafik yang menunjukkan tren naik dari kiri ke kanan.

            **Insight:**
                
            - Mayoritas penumpang puas dengan layanan boarding online.
            - Layanan boarding online merupakan alternatif yang lebih mudah dan nyaman dibandingkan dengan boarding tradisional.
            - Maskapai penerbangan perlu terus meningkatkan kemudahan dan kenyamanan layanan boarding online untuk mempertahankan tingkat kepuasan penumpang yang tinggi.
            
            **Actionable Insight:**
            
            Maskapai penerbangan dapat menyederhanakan proses boarding online, maskapai dapat membuat alur yang jelas dan mudah dipahami bagi penumpang. 
            Penyediaan berbagai pilihan check-in online, termasuk melalui situs web, aplikasi mobile, dan kios check-in di bandara, dapat memberikan fleksibilitas kepada penumpang dalam memilih metode yang paling nyaman bagi mereka. 
            Selain itu, melatih staf bandara untuk membantu penumpang yang mengalami kesulitan dalam menggunakan layanan boarding online dapat membantu meningkatkan efisiensi dan kepuasan penumpang selama proses boarding.
            ''')


######################### VISUALISASI 5 ######################### 
    ### Visualisasi 5 : Hubungan Satisfaction Dengan Inflight entertainment
    st.write('##### Hubungan Antara Satisfaction Dengan Inflight entertainment')

    # create graph
    plt.figure(figsize=(20, 10))
    sns.countplot(x="Inflight entertainment", data=df, hue="satisfaction", palette="Paired")

    # format graph
    plt.title("Satisfaction results by Inflight entertainment")
    sns.despine(top=True, right=True, left=False, bottom=False)

    # Update legend labels
    plt.legend(title="satisfaction", labels=['Neutral or Dissatisfied', 'Satisfied'], loc='center left', bbox_to_anchor=(1, 0.5))

    # Display graph using Streamlit
    st.pyplot(plt)
    st.markdown('''
            **Interpretasi:**
            
            Berdasarkan diagram tersebut, dapat dilihat bahwa terdapat hubungan positif antara kepuasan penumpang dengan kualitas inflight entertainment. 
            Semakin tinggi kualitas inflight entertainment, semakin tinggi pula tingkat kepuasan penumpang. 
            Hal ini ditunjukkan dengan grafik yang menunjukkan tren naik dari kiri ke kanan.

            **Insight:**
            
            - Mayoritas penumpang puas dengan kualitas inflight entertainment.
            - Inflight entertainment merupakan faktor penting yang dapat memengaruhi tingkat kepuasan penumpang.
            - Maskapai penerbangan perlu terus meningkatkan kualitas inflight entertainment untuk mempertahankan tingkat kepuasan penumpang yang tinggi.
            
            **Actionable Insight:**
            
            Maskapai penerbangan dapat meningkatkan kolaborasi dengan penyedia konten berkualitas penting untuk menyajikan variasi yang menarik. 
            Berbagai pilihan konten harus disediakan untuk memenuhi preferensi penumpang, sementara kru kabin dilatih untuk memberikan bantuan yang ramah saat menggunakan layanan tersebut.
            ''')


### DATA VISUALIZATION

if selected == 'Data Visualization':
    ### DATA DISTRIBUTION
    df = pd.read_csv('DataCleaned.csv')
    st.header('Data Distribution')
    st.write("""
    Visualisasi ini menampilkan distribusi dari fitur yang dipilih. Histogram menunjukkan bagaimana data terdistribusi 
            di sepanjang sumbu horizontal, sementara frekuensi kemunculan nilainya ditampilkan di sumbu vertikal. 
            Semakin tinggi puncak histogram, semakin sering nilai tersebut muncul dalam dataset. 
            Selain itu, adanya garis KDE (Kernel Density Estimate) membantu menunjukkan estimasi kepadatan probabilitas dari data. 
            Semakin tinggi atau curamnya puncak KDE, semakin besar kepadatan probabilitas pada rentang nilai tersebut.
    """)

    # Menghapus kolom 'Unnamed: 0'jika ada
    if 'Unnamed: 0' in df.columns:
        df.drop(columns=['Unnamed: 0'], inplace=True)


    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns

    selected_feature = st.selectbox("Pilih fitur untuk plot distribusi:", numeric_columns)

    st.subheader(f"Distribusi Kolom {selected_feature}")
    plt.figure(figsize=(10, 6))
    sns.histplot(df[selected_feature], kde=True, color='skyblue', bins=30)  # Ubah jumlah bins jika diperlukan
    plt.xlabel(selected_feature)
    plt.ylabel('Frekuensi')
    plt.title(f"Distribusi {selected_feature}")
    st.pyplot(plt)
###########################################################################################

    ### DATA RELATIONSHIP
    df = pd.read_csv('DataCleaned.csv')
    st.header('Relation')

    # Menghapus kolom 'Unnamed: 0'jika ada
    if 'Unnamed: 0' in df.columns:
        df.drop(columns=['Unnamed: 0'], inplace=True)

    # Plot matriks korelasi menggunakan heatmap
    fig, ax = plt.subplots()
    sns.heatmap(df.corr(numeric_only=True), annot=True, fmt=".2f", cbar=True, cmap="Blues")
    plt.xlabel('Fitur')
    plt.ylabel('Fitur')
    plt.title('Matriks Korelasi Antar Fitur Numerik')
    plt.gcf().set_size_inches(15, 10)
    st.pyplot(fig)

    st.write("""
            Visualisasi di atas memperlihatkan matriks korelasi antara fitur-fitur numerik dalam dataset. 
            Matriks korelasi digunakan untuk memahami hubungan linier antara variabel-variabel numerik. 
            Korelasi berkisar antara -1 hingga 1, di mana nilai 1 menandakan korelasi positif sempurna, nilai -1 menandakan korelasi negatif sempurna, dan nilai 0 menandakan tidak adanya korelasi. 
            Dari visualisasi korelasi di atas, dapat dilihat bahwa setiap variabel memiliki nilai yang mencerminkan hubungannya dengan variabel lainnya.
    """)

###########################################################################################

    ### DATA COMPOSITION
    df = pd.read_csv('DataCleaned.csv')
    st.header('Composition')

    # Menghapus kolom 'Unnamed: 0'jika ada
    if 'Unnamed: 0' in df.columns:
        df.drop(columns=['Unnamed: 0'], inplace=True)

    # Select numeric columns
    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns

    # Calculate composition by satisfaction level
    satisfaction_composition = df.groupby('satisfaction')[numeric_columns].mean()

    # Plot heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(satisfaction_composition.T, annot=True, cmap='YlGnBu')
    plt.title('Composition by Satisfaction Level')
    plt.xticks(ticks=[0.5, 1.5], labels=['Neutral or Dissatisfied', 'Satisfied'])
    plt.xlabel('Satisfaction')
    plt.ylabel('Feature')
    st.pyplot(plt)

    st.write ("""
              Visualisasi di atas menampilkan rata-rata nilai fitur numerik berdasarkan tingkat kepuasan penumpang (satisfaction level). 
              Setiap sel dalam heatmap menunjukkan rata-rata nilai fitur tersebut untuk setiap kategori tingkat kepuasan, yang terbagi menjadi dua yaitu"Neutral or Dissatisfied" dan "Satisfied". 
              Warna dalam heatmap menggambarkan intensitas nilai, di mana warna yang lebih gelap menunjukkan nilai yang lebih tinggi dan warna yang lebih terang menunjukkan nilai yang lebih rendah.
              Visualisasi ini berguna untuk melihat pola hubungan antara tingkat kepuasan penumpang dengan berbagai fitur numerik dalam dataset. Misalnya, kita dapat melihat apakah terdapat perbedaan dalam nilai fitur-fitur berdasarkan tingkat kepuasan, apakah penumpang yang merasa puas memiliki nilai fitur yang lebih tinggi daripada penumpang yang merasa netral atau tidak puas.
              """)

###########################################################################################

    ### DATA COMPARISON
    df = pd.read_csv('DataCleaned.csv')
    st.header('Comparation')

    ########################## Age Group
    # Calculate average ratings
    avg_satisfied_by_age_group = df.groupby('Age_group')['satisfaction'].mean()

    # Plot the comparison
    plt.figure(figsize=(10, 6))
    plt.plot(avg_satisfied_by_age_group.index, avg_satisfied_by_age_group.values, marker='o', linestyle='-', color='blue', label='Satisfied')
    plt.plot(avg_satisfied_by_age_group.index, 1 - avg_satisfied_by_age_group.values, marker='o', linestyle='-', color='red', label='Neutral or Dissatisfied')
    plt.title('Comparison of Satisfied And Neutral or Dissatisfied by Age Group')
    plt.xlabel('Age Group')
    plt.ylabel('Average Rating')
    plt.xticks(avg_satisfied_by_age_group.index)
    plt.legend()
    st.pyplot(plt)

    # Explanation
    st.write("""
            Visualisasi di atas digunakan untuk membandingkan rata-rata penilaian untuk "Satisfied and Neutral or Dissatisfied" berdasarkan kelompok usia pelanggan. 
            Dari visualisasi ini, dapat diamati apakah terdapat perbedaan dalam rata-rata penilaian untuk kategori-kategori ini di berbagai kelompok usia.
            """)
    
###########Clusster
if selected == 'Clustering':
    # Fungsi untuk mengubah kepuasan menjadi dua cluster: satisfied dan neutral or dissatisfied
    # Fungsi untuk mengubah kepuasan menjadi dua cluster: satisfied dan neutral or dissatisfied
    def convert_satisfaction_to_cluster(satisfaction):
        if satisfaction == 'satisfied':
            return 'satisfied'
        else:
            return 'neutral or dissatisfied'

    # Fungsi untuk melakukan clustering
    @st.cache
    def perform_clustering(data, num_clusters):
        X = data[['Inflight wifi service', 'Ease of Online booking', 'Food and drink', 'Online boarding', 'Inflight entertainment']]
        kmeans = KMeans(n_clusters=num_clusters)
        clusters = kmeans.fit_predict(X)
        data['Cluster'] = clusters
        # Ubah cluster menjadi 'satisfied' atau 'neutral or dissatisfied' berdasarkan kepuasan
        data['Cluster'] = data['Cluster'].map({0: 'satisfied', 1: 'neutral or dissatisfied'})
        return data

    # Sidebar
    st.sidebar.title("Clustering Options")
    selected_action = st.sidebar.radio("Select Action", ["Display Overall Clusters", "Perform Clustering"])

    # Jika pengguna memilih untuk menampilkan diagram scatter plot cluster secara keseluruhan
    if selected_action == "Display Overall Clusters":
        st.title("Overall Clustering Visualization")
        # Membaca dataset
        data = pd.read_csv('DataCleaned.csv')
        # Visualisasi data setelah clustering
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='Inflight wifi service', y='Ease of Online booking', hue='satisfaction', data=data)
        plt.title('Clustering Visualization')
        plt.xlabel('Inflight wifi service')
        plt.ylabel('Ease of Online booking')
        st.pyplot(plt)

    # Jika pengguna memilih untuk melakukan clustering berdasarkan inputan
    elif selected_action == "Perform Clustering":
        st.title("Perform Clustering")
        # Memilih jumlah cluster
        num_clusters = st.sidebar.slider("Number of Clusters", 1, 5, 3)
        # Membaca dataset
        data = pd.read_csv('DataCleaned.csv')
        # Melakukan konversi kepuasan menjadi cluster
        data['Cluster'] = data['satisfaction'].apply(convert_satisfaction_to_cluster)
        # Melakukan clustering
        clustered_data = perform_clustering(data, num_clusters)
        # Menampilkan hasil clustering
        st.write("Clustered Data:")
        st.write(clustered_data)
        # Visualisasi data setelah clustering
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='Inflight wifi service', y='Ease of Online booking', hue='Cluster', data=clustered_data)
        plt.title('Clustering Visualization')
        plt.xlabel('Inflight wifi service')
        plt.ylabel('Ease of Online booking')
        st.pyplot(plt)


################### Predict
# Path ke file CSV dalam folder
FILE_PATH = "DataCleaned.csv"

# Membaca file CSV ke dalam DataFrame
df = pd.read_csv(FILE_PATH)

# Train the KNeighborsClassifier model
X = df.drop(columns=['satisfaction', 'Gender', 'Customer Type', 'Age', 'Type of Travel', 'Class', 'Flight Distance', 'Online boarding', 'Departure/Arrival time convenient', 'Gate location', 'Seat comfort', 'On-board service', 'Leg room service', 'Baggage handling', 'Checkin service', 'Inflight service', 'Cleanliness', 'Departure Delay in Minutes', 'Arrival Delay in Minutes', 'Age_group', 'distance_group'])
y = df['satisfaction']
knn_model = KNeighborsClassifier()
knn_model.fit(X, y)


    # Graduation Prediction Page
if selected == "Predict":

    # page title
    st.title("Graduation Prediction using ML")

    # Box for IPS input

    st.subheader("Input kepuasan")
    col1, col2, col3 = st.columns(3)

    with col1:
        Inflight_wifi_service = st.number_input('Inflight wifi service', min_value=0.0, max_value=5.0, step=0.01, format="%.2f")

    with col2:
        Ease_of_Online_booking = st.number_input('Ease of Online booking', min_value=0.0, max_value=5.0, step=0.01, format="%.2f")

    with col3:
         Food_and_drink = st.number_input('Food and drink', min_value=0.0, max_value=5.0, step=0.01, format="%.2f")

    with col1:
        Online_boarding = st.number_input('Online boarding', min_value=0.0, max_value=5.0, step=0.01, format="%.2f")

    with col2:
        Inflight_entertainment = st.number_input('Inflight entertainment', min_value=0.0, max_value=5.0, step=0.01, format="%.2f")


    knn_diagnosis = ''

    # creating a button for Prediction    
    if st.button("Kepuasan Test Result"):

        user_input = [Inflight_wifi_service,  Ease_of_Online_booking ,  Food_and_drink,  Online_boarding, Inflight_entertainment]

        user_input = [float(x) for x in user_input]

        # Perform prediction using the trained model
        knn_prediction = knn_model.predict([user_input])

        if knn_prediction[0] == 1:
            knn_diagnosis = "Tidak Puas"
        else:
            knn_diagnosis = "Puas" 

    st.success(knn_diagnosis)
