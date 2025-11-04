1. Sentiment Analysis - McDonald's Reviews  

Proyek ini merupakan tugas mata kuliah **Teknik Pengembangan Model**, dengan fokus pada **pengembangan model Machine Learning** untuk menganalisis **sentimen pelanggan terhadap McDonald’s**.  

Dataset berasal dari **tiga sumber data ulasan berbeda**, digabungkan untuk mendapatkan pandangan yang lebih luas terhadap opini pelanggan.  
Model dibuat untuk mengklasifikasikan ulasan menjadi **positif**, **negatif**, atau **netral** dengan menggunakan pendekatan **Natural Language Processing (NLP)**.

---

2. Tujuan Proyek
- Menerapkan tahapan **pengembangan model Machine Learning** sesuai teori mata kuliah.  
- Melakukan **preprocessing** dan **pembersihan teks** dari berbagai sumber.  
- Mengembangkan model untuk analisis sentimen konsumen McDonald’s.  
- Mengevaluasi model dengan metrik seperti **accuracy**, **precision**, **recall**, dan **F1-score**.

---

3. Ringkasan Dataset
| Atribut | Deskripsi |
|----------|------------|
| `review_text` | Isi ulasan pelanggan |
| `rating` | Skor numerik (jika tersedia) |
| `sentiment_label` | Kategori hasil labeling (positif/negatif/netral) |
| `source` | Asal platform data |

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
   - Evaluasi performa dengan confusion matrix.  

---

5. Hasil
Model terbaik menunjukkan performa dengan **akurasi sebesar 91%** dengan algoritma SVM, dan mampu mengenali pola sentimen pelanggan dengan baik.  

---

6. Tools & Libraries
- Google Colab  
- pandas, numpy, scikit-learn, nltk, matplotlib, seaborn  

---

