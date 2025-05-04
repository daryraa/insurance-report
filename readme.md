# Laporan Proyek Machine Learning -  Dary Ramadhan Abdussalam
## Domain Proyek

Asuransi kesehatan di Indonesia menghadapi tantangan besar, terutama dalam pengelolaan klaim biaya perawatan. Jaminan Kesehatan Nasional (JKN) yang dikelola oleh BPJS Kesehatan kini mencakup mayoritas penduduk, namun banyaknya klaim menyebabkan defisit keuangan signifikan (Bramastya & Sari, 2024). Untuk menjamin keberlanjutan pembiayaan, dibutuhkan perhitungan klaim yang akurat agar premi sesuai dengan risiko dan anggaran dapat direncanakan lebih baik (Utami & Arifin, 2022; Bramastya & Sari, 2024). Penelitian global menunjukkan bahwa teknik regresi berbasis machine learning dapat memprediksi biaya kesehatan dengan akurasi tinggi menggunakan data demografi dan klinis pasien (Kumar & Lee, 2021).

### Manfaat Prediksi Biaya Asuransi

Memprediksi biaya klaim memiliki beberapa manfaat penting:

* **Penetapan premi yang adil**: Model prediksi mampu memperkirakan besaran klaim mendatang berdasarkan faktor risiko individu (usia, penyakit bawaan, gaya hidup) sehingga tarif premi dapat disesuaikan dengan risiko sebenarnya (Utami & Arifin, 2022).
* **Manajemen anggaran dan risiko**: Estimasi biaya akurat membantu perusahaan asuransi atau pemerintah merencanakan alokasi dana kesehatan dan mengurangi potensi defisit anggaran (Bramastya & Sari, 2024).
* **Deteksi anomali klaim**: Perbedaan signifikan antara prediksi dan realisasi klaim dapat diidentifikasi sebagai indikasi potensi penipuan atau kesalahan pencatatan (Benson & Campbell, 2023).
* **Pengembangan intervensi kesehatan**: Hasil prediksi dapat digunakan untuk merancang program pencegahan dini bagi kelompok berisiko tinggi, sehingga mengurangi beban penyakit kronis di masyarakat (Kumar & Lee, 2021).

### Pendekatan Regresi

Untuk masalah prediksi biaya yang bersifat nilai kontinu, metode regresi (misalnya regresi linier, pohon keputusan, random forest, atau gradient boosting) sangat sesuai (Kumar & Lee, 2021; Benson & Campbell, 2023). Model regresi mempelajari hubungan antara fitur pasien (usia, jenis kelamin, indeks massa tubuh, kebiasaan merokok, dll.) dan total biaya kesehatan (Utami & Arifin, 2022). Penelitian sebelumnya menunjukkan bahwa algoritma regresi ensemble mampu menangkap pola kompleks pada data klaim kesehatan, sehingga menghasilkan prediksi yang lebih akurat dibanding metode tradisional (Benson & Campbell, 2023; Kumar & Lee, 2021). Dengan demikian, proyek ini akan menerapkan berbagai teknik regresi untuk membangun model prediktif biaya asuransi, guna meningkatkan efisiensi finansial sistem asuransi kesehatan (Bramastya & Sari, 2024).


## Business Understanding

Pada bagian ini, dijelaskan proses klarifikasi masalah yang akan diselesaikan melalui pendekatan regresi berbasis machine learning dalam konteks prediksi biaya klaim asuransi kesehatan di Indonesia.

### Problem Statements

Masalah utama yang melatarbelakangi proyek ini adalah sebagai berikut:

* **Pernyataan Masalah 1**
  Defisit anggaran pada program Jaminan Kesehatan Nasional (JKN) yang diselenggarakan oleh BPJS Kesehatan disebabkan oleh ketidakmampuan dalam memperkirakan biaya klaim secara akurat berdasarkan profil risiko peserta.

* **Pernyataan Masalah 2**
  Ketidakpastian dalam menentukan premi asuransi yang proporsional menyebabkan premi yang tidak sesuai dengan risiko, sehingga mengganggu stabilitas finansial sistem asuransi nasional.

* **Pernyataan Masalah 3**
  Kesulitan dalam mengidentifikasi faktor-faktor risiko yang secara signifikan mempengaruhi biaya klaim kesehatan, sehingga pengembangan intervensi kesehatan preventif tidak efektif dan efisien.

### Goals

Tujuan dari proyek ini yang berfokus pada masing-masing pernyataan masalah di atas adalah:

* **Jawaban Pernyataan Masalah 1**
  Mengembangkan model prediksi berbasis machine learning dengan pendekatan regresi untuk memperkirakan biaya klaim secara tepat berdasarkan karakteristik peserta, sehingga membantu BPJS Kesehatan mengelola anggaran secara efektif.

* **Jawaban Pernyataan Masalah 2**
  Menghasilkan rekomendasi premi asuransi yang lebih akurat dan adil melalui prediksi biaya klaim yang mencerminkan risiko individu berdasarkan data demografi, gaya hidup, dan parameter kesehatan peserta.

* **Jawaban Pernyataan Masalah 3**
  Mengidentifikasi dan mengevaluasi fitur-fitur yang paling berpengaruh terhadap besaran biaya klaim, sehingga hasil ini dapat digunakan untuk merancang intervensi kesehatan preventif yang lebih efektif, khususnya bagi kelompok risiko tinggi.

## Data Understanding

Dataset diambil dari [kaggle.com/mirichoi0218/insurance](https://www.kaggle.com/datasets/mirichoi0218/insurance), terdiri dari 1338 sampel dengan fitur-fitur berikut:

* age: usia pemegang polis
* sex: jenis kelamin
* bmi: indeks massa tubuh
* children: jumlah tanggungan anak
* smoker: status perokok
* region: wilayah tempat tinggal
* charges: biaya klaim asuransi (target)

![alt text](image.png)
Mayoritas berada di rentang 25–35, menunjukkan bahwa sebagian besar peserta berada pada kondisi overweight. BMI dapat dijadikan indikator risiko dalam prediksi biaya kesehatan.

![alt text](image-1.png)
Wilayah southeast memiliki rata-rata biaya tertinggi, sementara northeast dan northwest lebih rendah. Wilayah geografis dapat memengaruhi biaya asuransi (mungkin karena perbedaan biaya layanan kesehatan atau prevalensi penyakit).

![alt text](image-2.png)
Untuk non-perokok, hubungan BMI dan charges tidak terlalu kuat, Untuk perokok, terlihat pola bahwa semakin tinggi BMI, semakin tinggi biaya klaim. Ada interaksi antara variabel bmi dan smoker. Ini berpotensi dijadikan fitur interaksi dalam model.

![alt text](image-3.png)
Grafik di atas menunjukkan bahwa distribusi biaya asuransi (charges) bersifat right-skewed, artinya sebagian besar individu memiliki biaya yang relatif rendah, sementara sebagian kecil memiliki biaya sangat tinggi (outlier).


## Data Preparation

Tahapan data preparation mencakup seluruh proses pembersihan, transformasi, dan encoding data sebelum digunakan dalam pelatihan model. Berikut langkah-langkah yang dilakukan:

1. **Pemisahan Fitur dan Target**
   Data dipisahkan menjadi fitur independen (X) dan target (y), di mana kolom 'charges' digunakan sebagai target atau variabel dependen.

2. **Encoding Fitur Kategorikal**

   * Kolom `region` yang memiliki empat kategori dikonversi menggunakan Label Encoding agar dapat direpresentasikan dalam bentuk numerik.
   * Fitur `sex` dan `smoker`, yang merupakan kategorikal biner, dikonversi menggunakan One-Hot Encoding dengan opsi `drop='first'` untuk menghindari multikolinearitas.

3. **Normalisasi/Standardisasi Data**
   Seluruh fitur numerik (termasuk hasil encoding) distandarisasi menggunakan `StandardScaler` dari scikit-learn. Ini dilakukan untuk memastikan semua fitur berada dalam skala yang sebanding agar model machine learning, khususnya yang sensitif terhadap skala (seperti regresi linear), dapat bekerja dengan optimal.

4. **Pembentukan Dataset Final**
   Setelah encoding dan scaling, seluruh fitur digabung menjadi satu DataFrame baru. Data kemudian dibagi menjadi data latih dan data uji dengan proporsi 80:20 menggunakan fungsi `train_test_split`, dengan parameter `random_state` ditetapkan agar hasil eksperimen dapat direproduksi.

5. **Eksperimen Subset Fitur**
   Untuk keperluan eksperimen, data juga disiapkan dalam tiga subset berbeda:

   * Hanya fitur numerik: `age`, `bmi`, `children`, `region`
   * Fitur dengan korelasi kuat terhadap target: `age`, `bmi`, `smoker_yes`
   * Seluruh fitur hasil preprocessing (gabungan semua fitur numerik dan kategorikal encoded)

Seluruh proses ini dilakukan secara sistematis dan berurutan untuk memastikan kualitas data yang konsisten dan optimal untuk pelatihan model regresi.

Langkah-langkah preprocessing meliputi:

* Label Encoding untuk fitur 'region'
* One-Hot Encoding untuk fitur kategorikal biner: 'sex' dan 'smoker' (drop='first')
* StandardScaler untuk fitur numerik dan hasil encoding agar skala fitur seragam
* Split data 80:20 untuk training dan testing
* Pengecekan duplicate value dan juga missing value

## Modeling

Tahapan ini membahas mengenai model machine learning yang digunakan untuk menyelesaikan permasalahan. Tiga algoritma regresi digunakan dalam eksperimen ini, yaitu:

1. Linear Regression
2. Random Forest Regressor
3. Gradient Boosting Regressor

Masing-masing model dilatih dan diuji menggunakan tiga subset fitur berbeda: (1) fitur numerik saja, (2) fitur dengan korelasi kuat, dan (3) seluruh fitur yang telah diencoding dan diskalakan. Seluruh model menggunakan parameter default dari pustaka scikit-learn tanpa tuning lanjutan.

Setiap model memiliki karakteristik tersendiri:

1. Linear Regression sederhana dan interpretatif, namun kurang mampu menangkap hubungan non-linear.
2. Random Forest Regressor menangani non-linearitas dengan baik dan relatif tahan terhadap overfitting, namun cenderung memiliki kompleksitas model yang tinggi.
3. Gradient Boosting Regressor unggul dalam mempelajari hubungan kompleks dalam data dan menghasilkan performa prediksi yang lebih baik, meskipun membutuhkan waktu pelatihan lebih lama dan tuning yang lebih hati-hati.

Berdasarkan evaluasi terhadap ketiga subset fitur, Gradient Boosting Regressor dengan seluruh fitur memberikan hasil terbaik dan dipilih sebagai model final karena mencapai keseimbangan antara akurasi dan generalisasi.

## Evaluation

Metrik evaluasi yang digunakan meliputi:

1. Mean Absolute Error (MAE): rata-rata selisih absolut antara nilai prediksi dan nilai aktual
2. Root Mean Squared Error (RMSE): akar dari rata-rata kuadrat selisih antara nilai prediksi dan aktual, sensitif terhadap outlier
3. R² Score (Koefisien Determinasi): proporsi variasi target yang dapat dijelaskan oleh model

| Fitur Digunakan   | Model             | MAE (↓)       | RMSE (↓)      | R² Score (↑) |
| ----------------- | ----------------- | ------------- | ------------- | ------------ |
| **Semua Fitur**   | Gradient Boosting | **2351.95**   | **3851.46**   | **0.9040**   |
| **Numerik Saja**  | Gradient Boosting | 9048.54       | 11925.54      | 0.0798       |
| **Korelasi Kuat** | Gradient Boosting | 9048.54       | 11925.54      | 0.0798       |

Evaluasi dilakukan pada masing-masing kombinasi fitur dan algoritma. Gradient Boosting dengan seluruh fitur menunjukkan performa terbaik dengan nilai MAE sebesar 2351.95, RMSE sebesar 3851.46, dan R² sebesar 0.9040. Ini berarti model dapat menjelaskan lebih dari 90% variasi pada data biaya klaim asuransi.

Sebaliknya, model yang hanya menggunakan fitur numerik atau fitur dengan korelasi tinggi menghasilkan performa yang jauh lebih rendah, menunjukkan bahwa keberadaan fitur kategorikal dan hasil encoding memberikan kontribusi signifikan terhadap akurasi model.


**Referensi:**

* Kumar, V., & Lee, K. (2021). Predicting healthcare costs with ensemble machine learning. *IEEE Access*, 9, 11234–11247.
* Utami, M., & Arifin, S. (2022). Analisis biaya klaim asuransi kesehatan berdasarkan data demografi. *Jurnal Statistik dan Data*, 10(4), 45–58.
* Benson, L., & Campbell, A. (2023). Machine learning methods for health insurance premium prediction. *Journal of Insurance & Data Science*, 15(2), 77–95.
* Bramastya, R., & Sari, T. (2024). Financial sustainability of Indonesia’s national health insurance system. *Health Policy and Planning*, 39(1), 86–98.

