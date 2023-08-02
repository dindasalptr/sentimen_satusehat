import re
import pandas as pd
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

BASE_DATA_PATH = r'data/'

# Meload stopword full yang digunakan pada pelatihan proyek ini
txt_stopword = pd.read_csv(f"{BASE_DATA_PATH}stopword_indo_satusehat.txt", names=[
                           "stopwords"], header=None)
stopword_full = set(txt_stopword.stopwords.values)

# Loading kata-kata slang ke dictionary
normalized_word = pd.read_csv(
    f'{BASE_DATA_PATH}kamus_alay_satusehat_deploy.csv')
normalized_word_dict = {}
for index, row in normalized_word.iterrows():
    if row[0] not in normalized_word_dict:
        normalized_word_dict[row[0]] = row[1]

# Membuat objek Sastrawi untuk stemming
factory = StemmerFactory()
stemmer = factory.create_stemmer()


def clean_text(review_text):
    # Mengganti titik di akhir text dengan spasi
    review_text = re.sub(r'(\w)[.](\w)', r'\1 \2', review_text.lower())
    # Mengganti baris baru \n dan lebih dari satu titik dengan spasi
    review_text = re.sub(r'\n', ' ', review_text)
    review_text = re.sub(r'[.]+', ' ', review_text)
    # hanya ambil alfabet dan spasi
    review_text = "".join(c.lower()
                          for c in review_text if c.isalpha() or c in [" "])
    # buang spasi berlebih dan gabung lagi
    review_text = " ".join(kata for kata in review_text.split())
    return review_text


def normalized_term(sentence):
    words = sentence.split()  # Memisahkan teks menjadi kata-kata
    normalized_words = [normalized_word_dict.get(
        word, word) for word in words]  # Normalisasi kata-kata
    # Menggabungkan kembali kata-kata menjadi teks
    normalized_document = ' '.join(normalized_words)
    return normalized_document

# Membuat fungsi menghapus stopword


def remove_stopword(sentence):
    result = sentence.split(' ')
    result = [word for word in result if word not in stopword_full]
    result = " ".join(result)
    return result


def full_preprocessing(raw_sentence: str):
    # Membersihkan teks dari tanda baca, angka emoticon dll, hanya menyisakan huruf
    sentence = clean_text(raw_sentence)
    sentence = normalized_term(sentence)  # Menormalkan kata-kata gaul / slang
    sentence = remove_stopword(sentence)  # Menghilangkan stopword
    # Mengubah kata imbuhan menjadi kata dasar
    sentence = stemmer.stem(sentence)
    return sentence
