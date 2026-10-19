# Studi Kasus Sesi 1- Telco Customer Churn: dari Descriptive sampai Prescriptive Analytics

| **Nama** | Diosi Putri Arlita |
|---|---|
| **NPM** | G1A023012 |

Studi kasus analitik end-to-end pada dataset publik **Telco Customer Churn (IBM, ±7.000 pelanggan)**.
Pertanyaan bisnis: *pelanggan mana yang kemungkinan berhenti berlangganan, mengapa, dan tindakan retensi apa yang layak dilakukan dengan anggaran terbatas?*

Proyek ini disusun mengikuti kerangka kuliah praktisi **"Make Sense of Data with Analysis and AI"** (Andria Arisal, BRIN, Informatics Practitioner Lecture Series, Universitas Bengkulu): pertanyaan bisnis → pemahaman data → data preparation → pemodelan → evaluasi → rekomendasi, dengan empat level analitik.

## Pemetaan materi ke kode

| Konsep dari materi | Implementasi |
|---|---|
| Business question → analytical question | Target `Churn`; unit analisis satu pelanggan; keputusan akhir berupa daftar target kampanye (`src/prescriptive.py`) |
| Data understanding dan kualitas data (*garbage in, garbage out*) | `src/quality.py`: nilai kosong, duplikat, kategori tidak konsisten, aturan bisnis (rule engine) |
| Data preparation | `src/data_loader.py` (pembersihan) dan `src/features.py` (feature engineering, preprocessing) |
| Descriptive: *apa yang terjadi?* | `src/descriptive.py` |
| Diagnostic: *mengapa terjadi?* | `src/diagnostic.py`: uji chi-square, Mann-Whitney, Cramér's V, segmen berisiko |
| Predictive: *apa yang akan terjadi?* | `src/modeling.py` |
| Prescriptive: *apa yang harus dilakukan?* | `src/prescriptive.py`: expected net benefit, daftar target, aksi per segmen, analisis sensitivitas |
| Jebakan data leakage | `src/leakage.py` dan pemisahan preprocessing di dalam `Pipeline` |
| Jebakan data tidak seimbang | Stratified split, `class_weight`, metrik PR-AUC/recall, tuning threshold, kalibrasi probabilitas |
| Keputusan tetap di tangan manusia | Aturan aksi bersifat eksplisit dan dapat diubah; parameter bisnis dapat diatur lewat CLI |

## Struktur repositori

```
telco-churn-analytics/
├── data/raw/                  # tempat file CSV (diunduh otomatis bila belum ada)
├── reports/
│   ├── figures/               # grafik hasil analisis
│   └── tables/                # tabel hasil analisis (CSV)
├── models/                    # model terlatih (joblib, tidak di-commit)
├── src/
│   ├── config.py              # path, parameter, daftar kolom, parameter kampanye
│   ├── data_loader.py         # muat dan bersihkan data
│   ├── quality.py             # laporan kualitas data dan aturan bisnis
│   ├── leakage.py             # audit kebocoran data
│   ├── descriptive.py
│   ├── diagnostic.py
│   ├── features.py
│   ├── modeling.py
│   ├── prescriptive.py
│   └── run_pipeline.py        # menjalankan seluruh alur
├── tests/
├── requirements.txt
└── README.md
```

## Menjalankan

```bash
git clone https://github.com/<username>/telco-churn-analytics.git
cd telco-churn-analytics
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m src.run_pipeline
```

Dataset diunduh otomatis ke `data/raw/Telco-Customer-Churn.csv` dari repositori IBM. Jika tanpa internet, unduh manual file `Telco-Customer-Churn.csv` (tersedia juga di Kaggle sebagai *Telco Customer Churn*) lalu letakkan di folder tersebut, atau tunjuk lokasinya:

```bash
python -m src.run_pipeline --data path/ke/Telco-Customer-Churn.csv
```

Parameter kampanye retensi dapat diubah:

```bash
python -m src.run_pipeline --offer-cost 20 --success-rate 0.25 --horizon-months 12 \
                           --gross-margin 0.4 --budget-share 0.2
```

Menjalankan tes:

```bash
pytest
```

Seluruh tabel dan grafik ditulis ke `reports/tables` dan `reports/figures`.

## Alur analisis

### 1. Kualitas data dan preparation
- `TotalCharges` berisi string kosong untuk pelanggan dengan `tenure = 0`. Aturan: kosong dan `tenure = 0` diisi 0; kasus lain diisi `tenure × MonthlyCharges`.
- Kategori `No internet service` dan `No phone service` diseragamkan menjadi `No`, karena informasinya sudah dibawa oleh `InternetService` dan `PhoneService`.
- Aturan bisnis diperiksa eksplisit: layanan tambahan aktif tanpa internet, `MultipleLines` aktif tanpa telepon, nilai tagihan tidak masuk akal.

### 2. Descriptive
Tingkat churn keseluruhan, churn per kontrak, metode pembayaran, jenis internet, dan kelompok lama berlangganan, serta distribusi `tenure`, `MonthlyCharges`, dan `TotalCharges` berdasarkan status churn.

### 3. Diagnostic
- Uji chi-square dengan koreksi Bonferroni dan ukuran efek Cramér's V untuk fitur kategorikal.
- Uji Mann-Whitney dengan rank-biserial untuk fitur numerik.
- Segmen dua dimensi dengan churn rate dan lift tertinggi (minimum 80 pelanggan per segmen).
- Efek layanan tambahan pada pelanggan berinternet.

Hasil diagnostic menunjukkan **asosiasi**, bukan sebab-akibat. Hipotesis penyebab perlu divalidasi dengan data tambahan atau eksperimen.

### 4. Predictive dan dua jebakan utama

**Data leakage**
- `customerID` dikeluarkan dari fitur.
- Kolom pasca-outcome yang ada di sebagian distribusi dataset (`Churn Label`, `Churn Value`, `Churn Score`, `Churn Reason`, `Churn Category`, `CLTV`, `Customer Status`) dibuang otomatis bila ada. Versi CSV standar tidak memuat kolom tersebut, tetapi versi Excel IBM memuatnya.
- Deteksi otomatis: setiap fitur diuji sendirian terhadap target (AUC; fitur kategorikal memakai target encoding out-of-fold). Fitur dengan AUC ≥ 0,90 ditandai mencurigakan dan dibuang.
- Seluruh preprocessing (imputasi, scaling, one-hot) berada di dalam `Pipeline`, sehingga hanya dipelajari dari data latih pada setiap fold.
- Threshold dipilih dari prediksi out-of-fold pada data latih, bukan dari data uji.

**Data tidak seimbang (churn sekitar seperempat dari pelanggan)**
- Pembagian data stratified, cross-validation 5-fold stratified.
- Akurasi tidak dipakai sebagai kriteria utama. Model pembanding `dummy_majority` menunjukkan akurasi tinggi dengan recall 0, sama seperti pelajaran dalam materi (akurasi 89% tetapi recall 0%).
- Kriteria pemilihan model: PR-AUC; metrik pendamping: recall, precision, F2, ROC-AUC, balanced accuracy.
- Model dengan dan tanpa `class_weight` dibandingkan (`logreg_unweighted` vs `logreg_balanced`).
- Probabilitas dikalibrasi (`CalibratedClassifierCV`, sigmoid) karena pembobotan kelas menggeser probabilitas, sedangkan tahap prescriptive memakai probabilitas sebagai input perhitungan nilai.
- Threshold dituning dengan F2 (recall dibobotkan lebih tinggi dari precision), sebab biaya menghubungi pelanggan jauh lebih kecil daripada kehilangan pelanggan.

Model yang dibandingkan: baseline mayoritas, Logistic Regression (tanpa dan dengan bobot kelas), Random Forest, dan HistGradientBoosting.

### 5. Prescriptive
Untuk setiap pelanggan pada data uji:

```
nilai_pelanggan      = MonthlyCharges × horizon_bulan × gross_margin
expected_net_benefit = P(churn) × success_rate × nilai_pelanggan − biaya_penawaran
```

- Pelanggan dengan expected net benefit positif diurutkan; sebanyak `budget_share` dari populasi dipilih sebagai target.
- Setiap target mendapat aksi lewat aturan berurutan di `ACTION_RULES` (`src/prescriptive.py`): bundel dukungan teknis untuk pelanggan fiber bulanan tanpa tech support, program onboarding untuk pelanggan baru, insentif kontrak 1 tahun, insentif pembayaran otomatis, dan panggilan retensi sebagai default.
- Strategi dibandingkan pada data uji: tanpa kampanye, hubungi semua, acak, top probabilitas, top expected net benefit dengan anggaran, dan semua expected benefit positif tanpa anggaran.
- Analisis sensitivitas terhadap success rate dan biaya penawaran, karena kedua angka tersebut asumsi, bukan hasil pengukuran.

## Asumsi dan keterbatasan

- `success_rate`, `offer_cost`, `horizon_months`, dan `gross_margin` adalah asumsi ilustratif. Nilai sebenarnya harus berasal dari bisnis, dan efektivitas tiap aksi sebaiknya diuji lewat A/B test sebelum diterapkan penuh.
- Dataset berupa potret satu waktu tanpa tanggal, sehingga pembagian data tidak berbasis waktu. Pada kasus nyata, validasi berbasis waktu lebih tepat.
- Aturan aksi disusun dari pola diagnostic dan harus ditinjau pemilik bisnis. Keputusan akhir tetap oleh manusia.
- Data publik dan bersih relatif terhadap data operasional nyata; proporsi waktu untuk data preparation pada kasus nyata jauh lebih besar.

## Lisensi data
Dataset Telco Customer Churn disediakan oleh IBM sebagai sampel data publik. Periksa ketentuan penggunaan pada sumber aslinya.
