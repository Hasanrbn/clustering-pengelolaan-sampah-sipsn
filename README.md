**Kelas Praktikum | Data Science IF-D**
**Clustering Pengelolaan Sampah SIPSN**

Proyek Data Science menggunakan K-Means Clustering untuk mengumpulkan wilayah berdasarkan indikator timbulan dan pengelolaan sampah dari SIPSN guna mendukung identifikasi prioritas intervensi.

Anggota Kelompok:
1. Muhammad Hasan Rabbani — 123240005
2. M. Diandra Al Fashya Munandar — 123240022
3. Muhammad Faaiz — 123240009
4. Arilda Sarifah Umahatika — 123240030

## Struktur proyek
| Berkas | Keterangan |
|---|---|
| `analisis_sampah.ipynb` | **Notebook utama** (EDA, preprocessing, pemodelan K-Means, uji sensitivitas) |
| `Statistik_Pengelolaan_Sampah_2026_2026-10-22.xls` | Data mentah SIPSN (sebenarnya berkas HTML) |
| `output/` | Grafik (PNG) dan tabel (CSV) hasil notebook |
| `requirements.txt` | Daftar library Python yang dibutuhkan |

## Cara menjalankan
```bash
pip install -r requirements.txt
jupyter notebook analisis_sampah.ipynb      # lalu Kernel > Restart & Run All
```
Di Google Colab: unggah notebook, jalankan, lalu unggah berkas `.xls` saat diminta.
Pengaturan (jumlah klaster, metode imputasi, dll.) ada di sel "KONFIGURASI" pada bagian 0.