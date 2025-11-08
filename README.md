1. Sentiment Analysis - McDonald's Reviews  

Proyek ini merupakan tugas mata kuliah **Teknik Pengembangan Model**, dengan fokus pada **pengembangan model Machine Learning** untuk menganalisis **sentimen pelanggan terhadap McDonald’s**.  

Dataset berasal dari **tiga sumber data ulasan berbeda**, digabungkan untuk mendapatkan pandangan yang lebih luas terhadap opini pelanggan.  
Model dibuat untuk mengklasifikasikan ulasan menjadi **positif**, **negatif**, atau **netral** dengan menggunakan pendekatan **Natural Language Processing (NLP)**.

---

2. Tujuan Proyek
- Menerapkan tahapan **pengembangan model Machine Learning** sesuai teori mata kuliah.  
- Melakukan **preprocessing** dan **pembersihan teks** dari berbagai sumber.  
- Mengembangkan model untuk analisis sentimen konsumen McDonald’s.  
- Mengevaluasi model dengan metrik **ROC-AUC** 

---

3. Ringkasan Dataset
a. Usernamne : kolom ini berisi username pelanggan
b. Rating : kolom ini berisikan rating antara 1-5
c. review_text : kolom ini beirisi ulasan komentar pelanggan
d. source : kolom ini berisi sumber dataset. ex: dataset_1 berarti data yang berasal dari sumber data pertama


> **Total data:** 35,421 baris.  
> **Jumlah sumber:** 3 sumber data

---

4. Metodologi
a. **Preprocessing:**  
   - Case folding  
   - Stopword removal  
   - Tokenization  
   - Stemming  

b. **Feature Extraction:**  
   - TF-IDF  

c. **Modeling:**  
   - Algoritma: Naive Bayes / SVM / Logistic Regression
   - Evaluasi performa dengan confusion matrix dan ROC-AUC.  

---

5. Hasil
Model terbaik menunjukkan performa dengan **akurasi sebesar 91%** dengan algoritma SVM, dan mampu mengenali pola sentimen pelanggan dengan baik.  

---

6. Tools & Libraries
- Google Colab  
- pandas, numpy, scikit-learn, nltk, matplotlib, seaborn, dan joblib  

---
## DOCKER


Analisis sentimen berbasis Machine Learning menggunakan tiga algoritma (SVM, Logistic Regression, dan Naive Bayes) yang sudah dikemas dalam Docker agar dapat dijalankan di lingkungan mana pun tanpa perlu menginstal library tambahan.

Seluruh pipeline berjalan otomatis: mulai dari load data, cleaning, training, evaluasi, hingga menyimpan file output (gambar, model .pkl, CSV hasil prediksi).

---

## 1. Struktur Repository
projectTPM
├─ Dockerfile
├─ main.py
├─ requirements.txt
├─ data/
   └─ data_gabungan.csv


Folder *output* (models, images, csv) **tidak disimpan di repository**, tetapi dibuat otomatis saat container dijalankan. Bisa dibuat di lokal dick D dengan nama folder **"docker_hasil"**

---

## 2. Cara Build Docker Image

Pastikan berada di folder project lalu jalankan:
Image akan terbuat dengan nama `sentiment-mcd3:latest`.

---

## 3. Cara Menjalankan Container

Container membutuhkan *mounting volume* agar semua file output (gambar, model, csv) tersimpan ke komputer lokal.

Sesuaikan path lokal folder output kamu, lalu jalankan:

Windows CMD:
docker run --rm ^
  -v "D:\docker_hasil:/app/images" ^
  -v "D:\docker_hasil:/app/models" ^
  -v "D:\docker_hasil:/app/output" ^
  katarina77/sentiment-mcd3:latest


Semua file hasil akan muncul di: D:\docker_hasil\

---

## 4. Isi Container

Container ini menjalankan `main.py` yang mencakup:

### ✔ Explorasi data  
### ✔ Cleaning dan preprocessing  
### ✔ Training model:  
- SVM + tuning  
- Logistic Regression (varian Count/Tfidf + SVD)  
- Naive Bayes  
### ✔ Confusion Matrix dan ROC Curve  
### ✔ Perbandingan model  
### ✔ Implementasi dengan data yang sudah dimasukan (tanpa input manual karena docker memiliki sifat non interaktif)
Container otomatis menjalankan prediksi contoh teks: "Ayamnya enak banget, tapi aplikasinya error pas mau bayar"

Hasilnya disimpan ke: /app/output/auto_inference_results.csv

---

## 5. Hasil Output

Setelah container selesai, folder output berisi:

- `*.png` → grafik,
- `*.pkl` → model final,
- `auto_inference_results.csv` → hasil implementasi,
- `all_models_bin.pkl` → bundle 3 model final,
dsb.
---

## 6. Catatan Penting

- Semua path output di `main.py` telah disesuaikan menjadi `/app/output/`.
- File dataset **tidak boleh** disimpan dalam container, harus di-mount dari luar.
- Jika ingin push image ke Docker Hub:

docker tag sentiment-mcd3:latest katarina77/sentiment-mcd3:latest
docker push katarina77/sentiment-mcd3:latest

atau pada docker hub: https://hub.docker.com/repository/docker/katarina77/sentiment-mcd3/tags

---



