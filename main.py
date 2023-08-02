import streamlit as st
import joblib
from preprocessing_data import full_preprocessing


PATH_MODEL = r'model/pipe_svc_lienar_c1_balanced.jb'

svc_model = joblib.load(PATH_MODEL)


st.title('Klasifikasi Sentimen dari Ulasan Aplikasi Satu Sehat')
st.write('Tentukan sentimen dari ulasan yang diperoleh ')


contoh_teks = st.selectbox(
    'Contoh teks review / ulasan aplikasi Satu Sehat: ',
    ('Aplikasi yg sangat Belum Sempurna. Sedih melihatnya.', 'Bisa jalan sih cuma lelet', 'Biar pun masuk pakai no dan email yang terdaftar tidak bisa juga loging2......tambah parah di update', 'Aplikasi ini sangat bagus dan bisa diandalkan'))

if contoh_teks != None:
    processed_example = full_preprocessing(contoh_teks)
    hasil = svc_model.predict([processed_example])
    st.write(f"Sentimen dari '{contoh_teks}' adalah: {str(hasil[0])}",
             )
else:
    print('Maaf, ada masalah')

st.subheader('Masukkan :blue[ulasan] yang anda peroleh ke bawah:')

ulasan = st.text_area('Teks Ulasan:', )

if st.button('Prediksi Sentimen'):
    processed_sentence = full_preprocessing(ulasan)
    # Jika hasil preprocessing berupa kalimat kosong
    if len(processed_sentence) == 0:
        st.write(f"Tidak bisa melakukan klasifikasi. Kalimat yang anda masukkan terlalu singkat atau hanya berisi angka, emoticon atau kata-kata stopword.")
        st.write(f"Ulasan anda: {ulasan}")
    else:
        hasil = svc_model.predict([processed_sentence])
        st.write(f"Sentimen dari '{ulasan}' adalah: :blue[{str(hasil[0])}]",
                 )

else:
    st.write('Belum ada ulasan yang masuk')
