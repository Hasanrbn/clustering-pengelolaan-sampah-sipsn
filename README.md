# Klasifikasi Prioritas Intervensi Pengelolaan Sampah Plastik Perdesaan Menggunakan K-Means Clustering

**Kelas Praktikum | Data Science IF-D | ID KELOMPOK 38**

Proyek Data Science yang mengelompokkan kabupaten di Indonesia (wilayah perdesaan) berdasarkan indikator timbulan dan kinerja pengelolaan sampah dari **SIPSN (Sistem Informasi Pengelolaan Sampah Nasional)** menggunakan algoritma **K-Means Clustering** untuk mendukung penentuan prioritas intervensi kebijakan.

### Anggota Kelompok
1. **Muhammad Hasan Rabbani** — `123240005`
2. **M. Diandra Al Fashya Munandar** — `123240022`
3. **Muhammad Faaiz** — `123240009`
4. **Arilda Sarifah Umahatika** — `123240030`

---

## Struktur Proyek

| Berkas | Keterangan |
| :--- | :--- |
| `analisis_sampah.ipynb` | **Notebook utama** berisi seluruh alur analisis (Tahap 0–10: *Business Understanding*, *Data Cleaning*, EDA, *Preprocessing*, validasi *Train-Test Split*, pemodelan K-Means $k=3$, visualisasi PCA 2D & Boxplot, hingga ekspor hasil) |
| `Statistik_Pengelolaan_Sampah_2026_2026-10-22.xls` | Dataset mentah SIPSN KLHK (format tabel HTML-Excel yang dibaca menggunakan `pandas.read_html`) |
| `Hasil_Clustering_Prioritas_Kabupaten.csv` | Luaran tabel hasil pengelompokan prioritas intervensi untuk 414 Kabupaten |
| `Hasil_Clustering_Prioritas_Kabupaten.xlsx` | Luaran Excel berisi 2 *sheet*: daftar lengkap 414 Kabupaten terurut (`Data_414_Kabupaten`) dan ringkasan statistik per klaster (`Ringkasan_Statistik_Klaster`) |
| `requirements.txt` | Daftar *library* Python yang dibutuhkan |

---

## Alur Tahapan dalam Notebook

1. **Tahap 0 — Business Understanding:** Latar belakang ketimpangan pengelolaan sampah desa vs kota, tujuan, dan rumusan pertanyaan bisnis.
2. **Tahap 1–3 — Data Understanding, Audit & Cleaning:** Validasi 511 wilayah unik (414 Kabupaten & 97 Kota) dari 519 baris mentah, perbaikan pemisah ribuan Indonesia (`Timbulan (tpd)`), imputasi median, dan normalisasi skala persentase.
3. **Tahap 4 — Exploratory Data Analysis (EDA):** Analisis statistik deskriptif, perbandingan Kabupaten (*proxy* perdesaan) vs Kota, analisis jenis TPA, dan matriks korelasi Spearman.
4. **Tahap 5 — Preprocessing & Feature Engineering:** Penyaringan khusus **414 Kabupaten**, transformasi logaritmik (`np.log1p`), dan standarisasi Z-Score (`StandardScaler`).
5. **Tahap 6 — Evaluasi Klaster & Validasi Model:** Evaluasi $k = 2\text{--}7$ (*Elbow Method*, *Silhouette Score*, *Davies-Bouldin Index*), pengujian stabilitas dengan **Train-Test Split (80:20)**, serta pelatihan model K-Means final ($k = 3$, `random_state=42`).
6. **Tahap 7–8 — Cluster Profiling & Visualisasi:** Profil statistik tiap klaster, daftar *Top 5 Kabupaten* per klaster, proyeksi **PCA 2D** ($76,31\%$ varians terjelaskan), dan **Boxplot 2×2** skala *symlog*.
7. **Tahap 9–10 — Ekspor Hasil & Kesimpulan Kebijakan:** Penyimpanan luaran ke format `.csv` dan `.xlsx` serta perumusan rekomendasi intervensi per klaster.

---

## Ringkasan Hasil Klasterisasi (414 Kabupaten)

- **Klaster 0 — Prioritas Sangat Tinggi / Kritis (153 Kabupaten — 37,0%):** Tingkat sampah terkelola mendekati nol (rata-rata $0,79\%$) dengan fasilitas bank sampah (BSU/BSI) rata-rata $< 1\text{ unit}$. Fokus intervensi pada infrastruktur dasar dan penutupan *open dumping*.
- **Klaster 1 — Prioritas Tinggi / Beban Timbulan Ekstrem (146 Kabupaten — 35,3%):** Volume timbulan sangat besar (rata-rata $484,79\text{ tpd}$) dengan jaringan BSU aktif (rata-rata $82,19\text{ unit}$), namun $\approx 67,4\%$ sampah belum terkelola optimal. Fokus pada penguatan fasilitas hilir (BSI, TPS3R, RDF).
- **Klaster 2 — Prioritas Moderat / Perintis Efisien (115 Kabupaten — 27,8%):** Timbulan terkendali ($111,03\text{ tpd}$) dengan capaian pengelolaan $\approx 29,5\%$. Fokus pada replikasi kelembagaan dan insentif pemilahan sampah.

---

## Cara Menjalankan

### 1. Lingkungan Lokal (VS Code / Jupyter)
Pastikan berkas dataset `Statistik_Pengelolaan_Sampah_2026_2026-10-22.xls` berada di direktori yang sama dengan `analisis_sampah.ipynb`, lalu jalankan:
```bash
pip install -r requirements.txt
jupyter notebook analisis_sampah.ipynb
```
