# -*- coding: utf-8 -*-
"""
Analisis Pengelolaan Sampah SIPSN - Progress 1 (versi revisi lengkap)

Cara pakai:
  1. pip install lxml pandas numpy matplotlib seaborn scikit-learn
  2. Taruh file Statistik_Pengelolaan_Sampah_*.xls di folder yang sama dengan skrip ini
  3. python analisis_sampah_revisi.py
Semua grafik (PNG) dan tabel (CSV) tersimpan otomatis di folder 'output/'.
"""

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

try:
    from IPython.display import display
except ImportError:
    display = print

# =====================================================================
# KONFIGURASI
# =====================================================================
SHOW_PLOTS = False            # True = tampilkan jendela grafik (skrip berhenti sampai jendela ditutup)
IMPUTASI_BSU_BSI = 'median'   # 'median' (sesuai slide lama) atau 'nol' (anggap "-" = tidak ada fasilitas)
TEST_SIZE = 0.20
RANDOM_STATE = 42
AMBANG_REDUNDAN = 0.80        # |rho Spearman| >= ini => fitur turunan dianggap redundan
NAMA_FILE_POLA = 'Statistik_Pengelolaan_Sampah_*.xls'

try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:             # jika dijalankan di Jupyter/Colab
    BASE_DIR = Path.cwd()
OUT_DIR = BASE_DIR / 'output'
OUT_DIR.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False
pd.set_option('display.width', 220)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.float_format', lambda v: f'{v:.3f}')


# =====================================================================
# FUNGSI BANTU
# =====================================================================
def judul(teks):
    print("\n" + "=" * 70)
    print(teks)
    print("=" * 70)


def simpan_fig(nama):
    plt.tight_layout()
    plt.savefig(OUT_DIR / nama, dpi=200, bbox_inches='tight')
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close()


def simpan_tabel(df, nama, index=True):
    df.to_csv(OUT_DIR / nama, index=index, encoding='utf-8-sig')


_POLA_RIBUAN_ID = re.compile(r'^\d{1,3}(\.\d{3})+(,\d+)?$')   # '2.784' atau '2.784,5' (format Indonesia)


def ke_numerik(s):
    """Ubah kolom ke angka. Mengenali 3 bentuk teks:
       '2.784,0' (Indonesia, ribuan titik) | '42,7' (koma desimal) | '2784.0' (sudah dinormalkan pandas)."""
    if pd.api.types.is_numeric_dtype(s):
        return s

    def konversi(v):
        if pd.isna(v):
            return np.nan
        if isinstance(v, (int, float, np.number)):
            return float(v)
        t = str(v).strip()
        if _POLA_RIBUAN_ID.match(t):
            t = t.replace('.', '').replace(',', '.')
        elif ',' in t:
            t = t.replace(',', '.')
        try:
            return float(t)
        except ValueError:
            return np.nan
    return s.map(konversi)


def normalisasi_persen(s):
    """Jika skala persen terbaca basis 10.000, ubah ke 0-100."""
    return s / 100.0 if s.max() > 100 else s


def audit_kolom(df, kolom_num):
    """Audit per kolom: missing, outlier (IQR), nilai nol, nilai berulang."""
    rows = []
    for c in df.columns:
        s = df[c]
        row = {
            'Kolom': c,
            'Tipe': 'Numerik' if c in kolom_num else 'Kategori/Lainnya',
            'Missing': int(s.isna().sum()),
            'Missing_%': round(s.isna().mean() * 100, 2),
            'Nilai_unik': int(s.nunique()),
            'Nilai_berulang': int(s.dropna().duplicated().sum()),
            'Outlier_IQR': np.nan,
            'Outlier_%': np.nan,
            'Nilai_nol': np.nan,
        }
        if c in kolom_num:
            q1, q3 = s.quantile([.25, .75])
            iqr = q3 - q1
            n_out = int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum())
            row['Outlier_IQR'] = n_out
            row['Outlier_%'] = round(n_out / s.notna().sum() * 100, 2)
            row['Nilai_nol'] = int((s == 0).sum())
        rows.append(row)
    return pd.DataFrame(rows).set_index('Kolom')


# =====================================================================
judul("1. DATA UNDERSTANDING & DATA DICTIONARY")
# =====================================================================
file_cari = sorted(BASE_DIR.glob(NAMA_FILE_POLA))
if not file_cari:
    raise FileNotFoundError(f"Tidak ada file '{NAMA_FILE_POLA}' di folder: {BASE_DIR}")
file_path = file_cari[-1]
print("File data :", file_path.name)

# File SIPSN sebenarnya HTML bertipe .xls. thousands/decimal = format angka Indonesia.
tables = pd.read_html(file_path, thousands='.', decimal=',')
print(f"Jumlah tabel terbaca: {len(tables)}")
for i, t in enumerate(tables):
    print(f"  tabel[{i}] ukuran {t.shape}")
df_raw = max(tables, key=len).copy()          # tabel data wilayah = tabel dengan baris terbanyak
df_raw.columns = df_raw.columns.astype(str).str.strip()
print(f"Tabel utama dipakai: {df_raw.shape[0]} baris, {df_raw.shape[1]} kolom")

data_dictionary = pd.DataFrame([
    ['Provinsi', 'Kategori (teks)', 'Nama provinsi', 'Identitas geografis', 'Identitas'],
    ['Kota/Kabupaten', 'Kategori (teks)', 'Nama wilayah', 'Identitas; dasar proxy perdesaan (Jenis_Wilayah)', 'Identitas'],
    ['Timbulan (tpd)', 'Numerik (integer)', 'Ton/hari', 'Volume sampah', 'Fitur K-Means'],
    ['% S. Terkelola', 'Numerik (float)', 'Persen (0-100)', 'Kinerja penanganan sampah', 'Fitur K-Means'],
    ['BSU Aktif', 'Numerik (integer)', 'Unit', 'Bank Sampah Unit aktif (akses infrastruktur)', 'Fitur K-Means'],
    ['BSI Aktif', 'Numerik (integer)', 'Unit', 'Bank Sampah Induk aktif (akses infrastruktur)', 'Fitur K-Means'],
    ['BSU Tidak Aktif', 'Numerik (integer)', 'Unit', 'Bank sampah unit tidak aktif', 'Tidak dipakai'],
    ['BSI Tidak Aktif', 'Numerik (integer)', 'Unit', 'Bank sampah induk tidak aktif', 'Tidak dipakai'],
    ['Sampah Terkelola BSU', 'Numerik', 'Cek satuan di portal SIPSN', 'Sampah yang dikelola bank sampah unit', 'Kandidat fitur UAS'],
    ['Sampah Terkelola BSI', 'Numerik', 'Cek satuan di portal SIPSN', 'Sampah yang dikelola bank sampah induk', 'Kandidat fitur UAS'],
    ['% S. Belum Terkelola', 'Numerik (float)', 'Persen (0-100)', 'Komplemen dari % S. Terkelola (diverifikasi di bagian 5.3)', 'Dibuang (redundan)'],
    ['Jenis TPA', 'Kategori (teks)', 'Sanitary/Controlled/Open dumping/dll', 'Jenis tempat pemrosesan akhir', 'Profiling klaster'],
    ['Status RIPS', 'Kategori (teks)', 'Status dokumen', 'Status rencana induk pengelolaan sampah', 'Profiling klaster'],
    ['Kelompok', 'Kategori (teks)', '-', 'Pengelompokan bawaan SIPSN', 'Tidak dipakai'],
    ['Verifikasi Lapangan', 'Kategori (teks)', '-', 'Status verifikasi data', 'Metadata'],
    ['Nama TPA', 'Teks', '-', 'Nama TPA', 'Dibuang (kosong 100%)'],
    ['Nilai Kinerja PS', 'Numerik', '-', 'Nilai kinerja', 'Dibuang (kosong 100%)'],
    ['Link RIPS', 'Teks (URL)', '-', 'Tautan dokumen RIPS', 'Dibuang (banyak kosong)'],
    ['Progress SA', 'Numerik', '-', 'Progres', 'Dibuang (banyak kosong)'],
    ['No', 'Integer', '-', 'Nomor urut baris', 'Dibuang (bukan informasi)'],
], columns=['Variabel', 'Tipe Data', 'Satuan/Format', 'Makna', 'Peran'])
data_dictionary['Dtype terbaca'] = data_dictionary['Variabel'].map(
    lambda v: str(df_raw[v].dtype) if v in df_raw.columns else '(tidak ada di data)')
print("\n--- KAMUS DATA ---")
display(data_dictionary)
simpan_tabel(data_dictionary, 'tabel_kamus_data.csv', index=False)


# =====================================================================
judul("2. KUALITAS DATA (AUDIT MISSING, OUTLIER, DUPLIKASI PER KOLOM)")
# =====================================================================
numeric_features = ['Timbulan (tpd)', '% S. Terkelola', 'BSU Aktif', 'BSI Aktif']

df_audit = df_raw.mask(df_raw == '-')               # "-" = penanda kosong
base = df_audit.dropna(subset=['Kota/Kabupaten']).copy()
print(f"Baris mentah: {len(df_raw)} | baris kosong dibuang: {len(df_raw) - len(base)} | baris valid: {len(base)}")

# 2.1 Cek konversi angka (menangkap nilai yang gagal terbaca)
num_before = base[numeric_features].apply(ke_numerik)
gagal = []
for c in numeric_features:
    n_gagal = int((base[c].notna() & num_before[c].isna()).sum())
    gagal.append({'Kolom': c, 'Gagal_dikonversi': n_gagal})
print("\n--- Cek konversi angka (harus 0 semua) ---")
display(pd.DataFrame(gagal).set_index('Kolom'))

# 2.2 Audit per kolom
base_audit = base.copy()
base_audit[numeric_features] = num_before
tabel_audit = audit_kolom(base_audit, numeric_features)
print("\n--- AUDIT PER KOLOM (missing, outlier IQR, nilai nol, nilai berulang) ---")
display(tabel_audit)
simpan_tabel(tabel_audit, 'tabel_audit_kolom.csv')

print("\n--- Ringkasan missing (hanya kolom yang punya missing) ---")
display(tabel_audit.loc[tabel_audit['Missing'] > 0, ['Missing', 'Missing_%']])

# 2.3 Duplikasi
dups_baris = int(base.duplicated().sum())
dups_kunci = int(base.duplicated(subset=['Provinsi', 'Kota/Kabupaten']).sum())
print(f"\nDuplikat baris penuh                 : {dups_baris}")
print(f"Duplikat kunci (Provinsi+Kota/Kab)   : {dups_kunci}")
print(f"Wilayah unik                          : {base[['Provinsi', 'Kota/Kabupaten']].drop_duplicates().shape[0]}")

# 2.4 Outlier & nilai nol (keputusan: dipertahankan, diredam log1p)
print("\n--- Outlier (IQR) & nilai nol pada fitur numerik ---")
display(tabel_audit.loc[numeric_features, ['Outlier_IQR', 'Outlier_%', 'Nilai_nol']])
nol_timbulan = num_before['Timbulan (tpd)'].eq(0)
print(f"Wilayah dengan timbulan = 0: {int(nol_timbulan.sum())} "
      f"(kemungkinan belum melapor; dipertahankan dan diberi flag, bukan dihapus)")
if nol_timbulan.any():
    print(base.loc[nol_timbulan, ['Provinsi', 'Kota/Kabupaten']].head(10).to_string(index=False))


# =====================================================================
judul("3. DATA CLEANING & IMPUTASI")
# =====================================================================
df_clean = base.copy()

# 3.1 Buang kolom non-informatif
cols_to_drop = ['Nama TPA', 'Nilai Kinerja PS', 'Progress SA', 'Link RIPS', 'No']
dibuang = [c for c in cols_to_drop if c in df_clean.columns]
df_clean.drop(columns=dibuang, inplace=True)
print("Kolom dibuang:", dibuang)

# 3.2 Konversi numerik + imputasi (median dihitung dari seluruh data; bersifat deskriptif)
log_imputasi = []
for col in numeric_features:
    df_clean[col] = ke_numerik(df_clean[col])
    n_na = int(df_clean[col].isna().sum())
    if n_na > 0:
        if col in ['BSU Aktif', 'BSI Aktif'] and IMPUTASI_BSU_BSI == 'nol':
            nilai, metode = 0, 'nol (anggap "-" = tidak ada fasilitas)'
        else:
            nilai, metode = df_clean[col].median(), 'median'
        df_clean[col] = df_clean[col].fillna(nilai)
        log_imputasi.append({'Kolom': col, 'Jumlah_diisi': n_na, 'Metode': metode, 'Nilai': nilai})
df_clean['% S. Terkelola'] = normalisasi_persen(df_clean['% S. Terkelola'])
assert df_clean['% S. Terkelola'].between(0, 100).all(), "Ada % terkelola di luar 0-100"

# 3.3 Kategori
df_clean['Jenis TPA'] = df_clean['Jenis TPA'].fillna('Tidak diisi')
df_clean['Status RIPS'] = df_clean['Status RIPS'].fillna('Tidak diisi')


def kelompok_rips(v):
    """Gabungkan status RIPS yang terlalu rinci (beberapa hanya 2-5 wilayah) menjadi 4 kelompok."""
    t = str(v).strip().lower()
    if t == 'ada dan sudah disahkan':
        return 'Disahkan'
    if t == 'tidak ada':
        return 'Tidak ada'
    if t.startswith('ada') or t in ('dalam perencanaan', 'tidak sesuai'):
        return 'Ada, belum final'
    return 'Tidak diketahui'            # 'Belum ada informasi', 'Tidak diisi', dll.


df_clean['RIPS_Grup'] = df_clean['Status RIPS'].map(kelompok_rips)

# 3.4 Flag timbulan nol (tidak masuk model)
df_clean['Flag_Timbulan_Nol'] = df_clean['Timbulan (tpd)'].eq(0)

print("\n--- Log imputasi ---")
display(pd.DataFrame(log_imputasi) if log_imputasi else "Tidak ada imputasi numerik")
print("Dasar pemilihan metode imputasi BSU/BSI:")
for col in ['BSU Aktif', 'BSI Aktif']:
    print(f"  {col}: nilai 0 yang dilaporkan eksplisit = {int(tabel_audit.loc[col, 'Nilai_nol'])} wilayah | "
          f"tanda '-' (tidak ada data) = {int(tabel_audit.loc[col, 'Missing'])} wilayah")
if IMPUTASI_BSU_BSI == 'median':
    print("  -> Karena 0 dilaporkan secara eksplisit, '-' dibedakan dari 0: dianggap 'tidak melapor', diimputasi median.")
else:
    print("  -> '-' dianggap 'tidak ada fasilitas' dan diisi 0.")

# 3.5 Sebelum vs sesudah
nb = num_before.copy()
nb['% S. Terkelola'] = normalisasi_persen(nb['% S. Terkelola'])
sebelum_sesudah = pd.DataFrame({
    'Missing_sebelum': nb.isna().sum(),
    'Missing_sesudah': df_clean[numeric_features].isna().sum(),
    'Mean_sebelum': nb.mean(),
    'Mean_sesudah': df_clean[numeric_features].mean(),
    'Median_sebelum': nb.median(),
    'Median_sesudah': df_clean[numeric_features].median(),
    'Max_sebelum': nb.max(),
    'Max_sesudah': df_clean[numeric_features].max(),
})
print("\n--- KONDISI SEBELUM vs SESUDAH (fitur numerik) ---")
display(sebelum_sesudah)
simpan_tabel(sebelum_sesudah, 'tabel_sebelum_sesudah_numerik.csv')

kat_sebelum_sesudah = pd.DataFrame({
    'Missing_sebelum': base[['Jenis TPA', 'Status RIPS']].isna().sum(),
    'Missing_sesudah': df_clean[['Jenis TPA', 'Status RIPS']].isna().sum(),
    'Strategi': ["isi 'Tidak diisi'", "isi 'Tidak diisi'"],
})
print("\n--- Sebelum vs sesudah (kategori) ---")
display(kat_sebelum_sesudah)
simpan_tabel(kat_sebelum_sesudah, 'tabel_sebelum_sesudah_kategori.csv')

print(f"\nBentuk data: mentah {df_raw.shape} -> bersih {df_clean.shape}")
print(f"Total missing pada fitur numerik: sebelum {int(nb.isna().sum().sum())} -> sesudah {int(df_clean[numeric_features].isna().sum().sum())}")


# =====================================================================
judul("4. EXPLORATORY DATA ANALYSIS (EDA)")
# =====================================================================
# ---- 4.1 Statistik deskriptif (semua kalimat dihitung dari data) ----
stats_desc = df_clean[numeric_features].describe().T
stats_desc['Skewness'] = df_clean[numeric_features].skew()
print("--- 4.1 STATISTIK DESKRIPTIF ---")
display(stats_desc)
simpan_tabel(stats_desc, 'tabel_statistik_deskriptif.csv')

tim = df_clean['Timbulan (tpd)']
ter = df_clean['% S. Terkelola']
idx_max = tim.idxmax()
persen_lt10 = (ter < 10).mean() * 100
print("\n[Kesimpulan statistik deskriptif]")
print(f"- Timbulan: rata-rata {tim.mean():.1f} tpd | median {tim.median():.1f} tpd | "
      f"maksimum {tim.max():,.1f} tpd ({df_clean.loc[idx_max, 'Kota/Kabupaten']}) | skewness {tim.skew():.2f}")
print(f"- % terkelola: rata-rata {ter.mean():.1f}% | median {ter.median():.1f}%")
print(f"- {persen_lt10:.1f}% wilayah mengelola kurang dari 10% sampahnya")

# ---- 4.2 Representativitas & bias ----
print("\n--- 4.2 REPRESENTATIVITAS & BIAS SAMPEL ---")
df_clean['Jenis_Wilayah'] = np.where(
    df_clean['Kota/Kabupaten'].str.startswith('Kabupaten'), 'Kabupaten (Proxy Desa)', 'Kota')
rep_summary = (df_clean.groupby('Jenis_Wilayah')['% S. Terkelola']
               .agg(['count', 'mean', 'median', 'std']).reset_index())
display(rep_summary)
simpan_tabel(rep_summary, 'tabel_kabupaten_vs_kota.csv', index=False)

n_kab = int((df_clean['Jenis_Wilayah'] != 'Kota').sum())
n_kota = int((df_clean['Jenis_Wilayah'] == 'Kota').sum())
print(f"Cakupan: {len(df_clean)} wilayah = {n_kab} kabupaten + {n_kota} kota "
      f"(proporsi kabupaten {n_kab / len(df_clean):.1%})")
print("\nCatatan: kolom Kelompok = daftar program '343 TPA' (bukan klasifikasi kota/kabupaten), "
      "jadi TIDAK dipakai untuk memvalidasi Jenis_Wilayah. Validasi lewat awalan nama wilayah:")
print(pd.crosstab(df_clean['Jenis_Wilayah'], df_clean['Kota/Kabupaten'].str.split().str[0]))

pola_jawa = 'Jawa|Banten|Jakarta|Yogyakarta'
mask_jawa = df_clean['Provinsi'].str.contains(pola_jawa, case=False, na=False)
prov_jawa = sorted(df_clean.loc[mask_jawa, 'Provinsi'].unique())
share_jawa = tim[mask_jawa].sum() / tim.sum()
print(f"Provinsi Pulau Jawa terdeteksi ({len(prov_jawa)}): {prov_jawa}")
print(f"Jumlah wilayah di Jawa: {int(mask_jawa.sum())} dari {len(df_clean)} ({mask_jawa.mean():.1%})")
print(f"Share timbulan Pulau Jawa: {share_jawa:.1%}   (klaim slide lama: 55,3%)")
top_prov = df_clean.groupby('Provinsi')['Timbulan (tpd)'].sum().sort_values(ascending=False)
print("5 provinsi dengan timbulan terbesar:")
print((top_prov.head(5) / tim.sum()).map('{:.1%}'.format).to_string())
print(f"Jumlah provinsi dalam data: {df_clean['Provinsi'].nunique()}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.barplot(data=rep_summary, x='Jenis_Wilayah', y='mean', hue='Jenis_Wilayah',
            palette='Set2', ax=axes[0], legend=False)
axes[0].set_title('Perbandingan Kinerja Pengelolaan: Kabupaten vs Kota', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Jenis Wilayah (Proxy Perdesaan)')
axes[0].set_ylabel('Rata-rata Sampah Terkelola (%)')
for p in axes[0].patches:
    h = p.get_height()
    axes[0].annotate(f'{h:.1f}%', (p.get_x() + p.get_width() / 2., h / 2),
                     ha='center', va='center', fontsize=11, color='white', fontweight='bold')
sns.boxplot(data=df_clean, x='Jenis_Wilayah', y='% S. Terkelola', hue='Jenis_Wilayah',
            palette='Set2', ax=axes[1], legend=False)
axes[1].set_title('Distribusi Sebaran Persentase Terkelola', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Jenis Wilayah')
axes[1].set_ylabel('Persentase Terkelola (%)')
simpan_fig('fig_kab_vs_kota.png')

# ---- 4.3 Visualisasi fitur & target ----
print("\n--- 4.3 VISUALISASI FITUR DAN TARGET ---")
# (a) Histogram semua fitur numerik
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
warna = ['skyblue', 'lightgreen', 'orange', 'plum']
for ax, col, w in zip(axes.ravel(), numeric_features, warna):
    sns.histplot(df_clean[col], kde=True, ax=ax, color=w)
    ax.set_title(f'Distribusi {col} (skewness {df_clean[col].skew():.2f})', fontsize=11, fontweight='bold')
    ax.set_xlabel(col)
simpan_fig('fig_distribusi_fitur.png')

# (b) Boxplot semua fitur numerik (outlier dipertahankan)
fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
for ax, col, w in zip(axes, numeric_features, warna):
    sns.boxplot(y=df_clean[col], ax=ax, color=w)
    ax.set_title(f'Boxplot {col}', fontsize=11, fontweight='bold')
fig.suptitle('Outlier dipertahankan (kondisi riil), diredam dengan log1p', fontsize=11)
simpan_fig('fig_boxplot_fitur.png')

# (c) Kinerja berdasarkan Jenis TPA
tpa_perf = (df_clean.groupby('Jenis TPA')['% S. Terkelola'].agg(['mean', 'count'])
            .reset_index().sort_values('mean', ascending=False))
tpa_perf['Label'] = tpa_perf['Jenis TPA'] + ' (n=' + tpa_perf['count'].astype(str) + ')'
print("Rata-rata % terkelola per Jenis TPA:")
display(tpa_perf[['Jenis TPA', 'count', 'mean']])
simpan_tabel(tpa_perf, 'tabel_jenis_tpa.csv', index=False)

plt.figure(figsize=(10, 5))
ax = sns.barplot(data=tpa_perf, x='mean', y='Label', hue='Label', palette='viridis', legend=False)
plt.title('Rata-rata % Sampah Terkelola Berdasarkan Jenis TPA', fontsize=13, fontweight='bold')
plt.xlabel('Rata-rata Sampah Terkelola (%)')
plt.ylabel('Jenis TPA')
for p in ax.patches:
    w = p.get_width()
    ax.annotate(f'{w:.1f}%', (w + 1, p.get_y() + p.get_height() / 2.),
                ha='left', va='center', fontsize=10, color='black')
plt.xlim(0, 100)
simpan_fig('fig_jenis_tpa.png')

# ---- 4.4 Korelasi & 3 insight ----
print("\n--- 4.4 KORELASI SPEARMAN & 3 INSIGHT ---")
corr_matrix = df_clean[numeric_features].corr(method='spearman')
display(corr_matrix)
simpan_tabel(corr_matrix, 'tabel_korelasi_spearman.csv')
plt.figure(figsize=(7, 5))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, vmin=-1, vmax=1)
plt.title('Matriks Korelasi Spearman', fontsize=12, fontweight='bold')
simpan_fig('fig_korelasi.png')

c_tim_bsu = corr_matrix.loc['Timbulan (tpd)', 'BSU Aktif']
c_bsu_ter = corr_matrix.loc['BSU Aktif', '% S. Terkelola']
c_bsi_ter = corr_matrix.loc['BSI Aktif', '% S. Terkelola']
c_tim_ter = corr_matrix.loc['Timbulan (tpd)', '% S. Terkelola']
skew = df_clean[numeric_features].skew()
nol = df_clean[df_clean['% S. Terkelola'] == 0]
share_od = (nol['Jenis TPA'] == 'Open Dumping').mean() if len(nol) else float('nan')
od_semua = (df_clean['Jenis TPA'] == 'Open Dumping').mean()
tanpa_terkendali = (~nol['Jenis TPA'].isin(['Controlled Landfill', 'Sanitary Landfill'])).mean() if len(nol) else float('nan')


def kekuatan(r):
    a = abs(r)
    return 'lemah' if a < 0.3 else 'sedang' if a < 0.5 else 'cukup kuat' if a < 0.7 else 'kuat'

print("\nINSIGHT UNTUK KEPUTUSAN CLEANING/PEMODELAN (angka dihitung dari data):")
print(f"1. Skala: skewness fitur {skew.round(2).to_dict()}. Jarak Euclidean K-Means akan didominasi "
      f"wilayah besar (timbulan maks {tim.max():,.0f} vs median {tim.median():,.0f} tpd) -> "
      f"perlu log1p + StandardScaler. (Timbulan vs BSU Aktif: rho = {c_tim_bsu:.2f}, {kekuatan(c_tim_bsu)}.)")
print(f"2. Kandidat klaster prioritas: {len(nol)} wilayah ({len(nol) / len(df_clean):.1%}) dengan 0% terkelola; "
      f"{tanpa_terkendali:.1%} di antaranya tanpa TPA terkendali (Open Dumping {share_od:.1%}; sisanya tidak punya TPA "
      f"atau kolom TPA kosong); sebagai pembanding, Open Dumping pada seluruh wilayah = {od_semua:.1%}. "
      f"Hanya {1 - tanpa_terkendali:.1%} wilayah 0% terkelola yang punya TPA terkendali "
      f"-> penanda kuat untuk zona darurat intervensi.")
print(f"3. Proxy infrastruktur: BSU Aktif (rho = {c_bsu_ter:.2f}, {kekuatan(c_bsu_ter)}) dan "
      f"BSI Aktif (rho = {c_bsi_ter:.2f}, {kekuatan(c_bsi_ter)}) berkorelasi positif dengan % terkelola "
      f"(Timbulan: rho = {c_tim_ter:.2f}, {kekuatan(c_tim_ter)}) -> layak jadi fitur "
      f"pengganti data perilaku rumah tangga. Catatan: korelasi tidak membuktikan sebab-akibat.")

# ---- 4.5 Verifikasi angka yang dipakai di slide ----
print("\n--- 4.5 VERIFIKASI ANGKA KLAIM DI SLIDE ---")
print(f"Share timbulan Jawa        : {share_jawa:.1%}   (klaim lama 55,3%)")
print(f"Wilayah 0% terkelola       : {len(nol)} ({len(nol) / len(df_clean):.1%})   (klaim lama 64 / 12,5%)")
print("Komposisi Jenis TPA pada wilayah 0% terkelola:")
print(nol['Jenis TPA'].value_counts(normalize=True).round(3).to_string())
print(f"Wilayah <10% terkelola     : {persen_lt10:.1f}%   (klaim lama 47,2%)")
rips = df_clean.groupby('Status RIPS')['% S. Terkelola'].agg(['count', 'mean']).round(1)
print("Rata-rata % terkelola menurut Status RIPS (klaim lama: disahkan 33,2 vs tanpa 15,4):")
print(rips.to_string())
rips_g = df_clean.groupby('RIPS_Grup')['% S. Terkelola'].agg(['count', 'mean']).round(1)
print("Rata-rata % terkelola menurut kelompok RIPS (dipakai untuk slide):")
print(rips_g.to_string())
print(f"Open Dumping pada seluruh wilayah: {od_semua:.1%}   (angka 60,3% di slide lama = share SELURUH wilayah, bukan khusus wilayah 0% terkelola)")
print("3 timbulan terbesar:")
print(df_clean.nlargest(3, 'Timbulan (tpd)')[['Provinsi', 'Kota/Kabupaten', 'Timbulan (tpd)']].to_string(index=False))


# =====================================================================
judul("5. DATA PREPROCESSING & FEATURE SELECTION")
# =====================================================================
# ---- 5.1 Encoding (untuk profiling klaster, BUKAN input K-Means) ----
print("--- 5.1 ENCODING KATEGORIK (One-Hot) ---")
print("Status RIPS asli terlalu rinci (beberapa kategori hanya 2-5 wilayah), digabung menjadi RIPS_Grup:")
display(pd.crosstab(df_clean['Status RIPS'], df_clean['RIPS_Grup']))
df_enc = pd.get_dummies(df_clean[['Jenis TPA', 'RIPS_Grup', 'Jenis_Wilayah']], dtype=int)
print(f"One-hot menghasilkan {df_enc.shape[1]} kolom dari 3 variabel kategorik (Jenis TPA, RIPS_Grup, Jenis_Wilayah):")
print(list(df_enc.columns))
print("Alasan: K-Means memakai jarak Euclidean pada fitur numerik; variabel kategorik dienkode one-hot "
      "hanya untuk profiling klaster (membandingkan komposisi TPA/RIPS per klaster) setelah klaster terbentuk.")
simpan_tabel(df_enc, 'tabel_encoding_onehot.csv')

# ---- 5.2 Cek ketidakseimbangan kelas ----
print("\n--- 5.2 CEK IMBALANCE ---")
imb_rows = []
for kol in ['Jenis_Wilayah', 'Jenis TPA', 'Status RIPS', 'RIPS_Grup']:
    vc = df_clean[kol].value_counts()
    imb_rows.append({'Variabel': kol, 'Kelas': len(vc),
                     'Mayoritas': f"{vc.index[0]} ({vc.iloc[0]})",
                     'Minoritas': f"{vc.index[-1]} ({vc.iloc[-1]})",
                     'Rasio_maks:min': round(vc.iloc[0] / vc.iloc[-1], 1)})
display(pd.DataFrame(imb_rows).set_index('Variabel'))
simpan_tabel(pd.DataFrame(imb_rows), 'tabel_imbalance.csv', index=False)
print("Catatan: Status RIPS asli sangat tidak seimbang (ada kategori hanya 2 wilayah), maka digabung menjadi "
      "4 kelompok (RIPS_Grup) sebelum one-hot; rasio mayoritas:minoritas turun (bandingkan baris Status RIPS vs RIPS_Grup).")
print("Penanganan: ini clustering (tanpa label target), jadi tidak ada SMOTE/oversampling. "
      "Ketidakseimbangan ditangani dengan split berstrata pada Jenis_Wilayah dan diperhitungkan saat membaca profil klaster.")

# ---- 5.3 Feature engineering & feature selection ----
print("\n--- 5.3 FEATURE ENGINEERING & FEATURE SELECTION ---")
df_clean['Tidak_Terkelola_tpd'] = df_clean['Timbulan (tpd)'] * (1 - df_clean['% S. Terkelola'] / 100.0)
df_clean['Total_Bank_Sampah'] = df_clean['BSU Aktif'] + df_clean['BSI Aktif']
turunan = ['Tidak_Terkelola_tpd', 'Total_Bank_Sampah']
print("Fitur baru: Tidak_Terkelola_tpd = Timbulan x (1 - %terkelola)  [beban sampah belum tertangani]")
print("            Total_Bank_Sampah   = BSU Aktif + BSI Aktif          [total kapasitas bank sampah]")

kandidat = numeric_features + turunan
corr_kand = df_clean[kandidat].corr(method='spearman')
print("\nKorelasi Spearman antar kandidat fitur:")
display(corr_kand)
simpan_tabel(corr_kand, 'tabel_korelasi_kandidat.csv')
plt.figure(figsize=(8, 6))
sns.heatmap(corr_kand, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, vmin=-1, vmax=1)
plt.title('Korelasi Spearman Kandidat Fitur (termasuk fitur turunan)', fontsize=12, fontweight='bold')
simpan_fig('fig_korelasi_kandidat.png')

keputusan = []
selected_features = list(numeric_features)
for f in numeric_features:
    rho_target = corr_kand.loc[f, '% S. Terkelola'] if f != '% S. Terkelola' else np.nan
    alasan = f"skewness {df_clean[f].skew():.2f} (di-log1p); missing awal {tabel_audit.loc[f, 'Missing_%']:.1f}%"
    if f != '% S. Terkelola':
        alasan += f"; rho dengan % terkelola = {rho_target:.2f}"
    else:
        alasan += "; indikator kinerja inti"
    keputusan.append([f, 'Dipakai', alasan])

for f in turunan:
    sisa = corr_kand.loc[f, numeric_features].abs()
    mitra, rho_max = sisa.idxmax(), sisa.max()
    if rho_max >= AMBANG_REDUNDAN:
        keputusan.append([f, 'Dibuang (redundan)',
                          f"|rho| dengan {mitra} = {rho_max:.2f} >= {AMBANG_REDUNDAN}; informasi sudah terwakili"])
    else:
        selected_features.append(f)
        keputusan.append([f, 'Dipakai (fitur baru)',
                          f"|rho| maks dengan fitur lain = {rho_max:.2f} < {AMBANG_REDUNDAN}; menambah informasi baru"])

for c in dibuang:
    alasan = "nomor urut, bukan informasi" if c == 'No' else f"missing {tabel_audit.loc[c, 'Missing_%']:.1f}%"
    keputusan.append([c, 'Dibuang', alasan])
for c in ['Jenis TPA', 'Status RIPS']:
    keputusan.append([c, 'Profiling saja', "kategorik; tidak cocok untuk jarak Euclidean K-Means, dipakai membaca profil klaster"])
keputusan.append(['RIPS_Grup', 'Profiling saja (fitur baru)',
                  "penggabungan 10 status RIPS menjadi 4 kelompok agar one-hot tidak terlalu jarang"])

if '% S. Belum Terkelola' in df_clean.columns:
    belum = normalisasi_persen(ke_numerik(df_clean['% S. Belum Terkelola']))
    jumlah = df_clean['% S. Terkelola'] + belum
    rho_belum = df_clean['% S. Terkelola'].corr(belum, method='spearman')
    print(f"\nCek '% S. Belum Terkelola': jumlah dengan % terkelola = {jumlah.min():.2f} s.d. {jumlah.max():.2f}; "
          f"rho Spearman = {rho_belum:.2f}")
    if jumlah.between(99.5, 100.5).all():
        keputusan.append(['% S. Belum Terkelola', 'Dibuang (redundan)',
                          f"komplemen % terkelola (jumlah keduanya {jumlah.min():.1f}-{jumlah.max():.1f}%, rho = {rho_belum:.2f})"])
    else:
        keputusan.append(['% S. Belum Terkelola', 'Tidak dipakai',
                          f"rho dengan % terkelola = {rho_belum:.2f}; tidak dipakai agar tidak menggandakan informasi"])
for c in ['Sampah Terkelola BSU', 'Sampah Terkelola BSI']:
    if c in df_clean.columns:
        df_clean[c] = ke_numerik(df_clean[c])
        mitra = 'BSU Aktif' if c.endswith('BSU') else 'BSI Aktif'
        rho_m = df_clean[c].corr(df_clean[mitra], method='spearman')
        rho_t = df_clean[c].corr(df_clean['% S. Terkelola'], method='spearman')
        pct_nol = (df_clean[c] == 0).mean() * 100
        if abs(rho_m) >= AMBANG_REDUNDAN:
            keputusan.append([c, 'Dibuang (redundan)',
                f"|rho| dengan {mitra} = {rho_m:.2f} >= {AMBANG_REDUNDAN}; {pct_nol:.0f}% bernilai 0"])
        else:
            keputusan.append([c, 'Tidak dipakai (kandidat UAS)',
                f"rho dengan {mitra} = {rho_m:.3f} (< {AMBANG_REDUNDAN}), rho dengan % terkelola = {rho_t:.2f}; "
                f"{pct_nol:.0f}% bernilai 0 dan satuan belum dikonfirmasi -> dievaluasi lagi di UAS"])
sudah = {k[0] for k in keputusan} | {'Provinsi', 'Kota/Kabupaten', 'Jenis_Wilayah', 'Flag_Timbulan_Nol'}
for c in df_clean.columns:
    if c not in sudah and c != 'Split':
        keputusan.append([c, 'Tidak dipakai', 'di luar cakupan indikator model / metadata'])
keputusan.append(['Provinsi, Kota/Kabupaten', 'Identitas', 'label wilayah untuk interpretasi hasil klaster'])

tabel_keputusan = pd.DataFrame(keputusan, columns=['Fitur/Kolom', 'Keputusan', 'Alasan & bukti'])
print("\n--- TABEL KEPUTUSAN FITUR ---")
with pd.option_context('display.max_colwidth', 120):
    display(tabel_keputusan)
simpan_tabel(tabel_keputusan, 'tabel_keputusan_fitur.csv', index=False)
print(f"\nFitur akhir K-Means ({len(selected_features)}): {selected_features}")

# ---- 5.4 Split train/test + log1p + StandardScaler (fit hanya di train) ----
print("\n--- 5.4 SPLIT TRAIN/TEST + SCALING ---")
idx_train, idx_test = train_test_split(
    df_clean.index, test_size=TEST_SIZE, random_state=RANDOM_STATE,
    stratify=df_clean['Jenis_Wilayah'])
df_clean['Split'] = np.where(df_clean.index.isin(idx_train), 'train', 'test')

prop = pd.DataFrame({
    'Seluruh': df_clean['Jenis_Wilayah'].value_counts(normalize=True),
    'Train': df_clean.loc[idx_train, 'Jenis_Wilayah'].value_counts(normalize=True),
    'Test': df_clean.loc[idx_test, 'Jenis_Wilayah'].value_counts(normalize=True),
})
print(f"Split {int((1 - TEST_SIZE) * 100)}/{int(TEST_SIZE * 100)}, stratify=Jenis_Wilayah, random_state={RANDOM_STATE}")
print("Proporsi Jenis_Wilayah (membuktikan stratifikasi terjaga):")
display(prop)

scaler = StandardScaler()
scaler.fit(np.log1p(df_clean.loc[idx_train, selected_features]))      # fit HANYA di train
X_train = scaler.transform(np.log1p(df_clean.loc[idx_train, selected_features]))
X_test = scaler.transform(np.log1p(df_clean.loc[idx_test, selected_features]))
assert not np.isnan(X_train).any() and not np.isnan(X_test).any()
print(f"X_train: {X_train.shape} | X_test: {X_test.shape}")
print("Mean X_train per fitur (harus ~0):", X_train.mean(axis=0).round(3))
print("Std  X_train per fitur (harus ~1):", X_train.std(axis=0).round(3))
print("Mean X_test  per fitur (boleh menyimpang sedikit):", X_test.mean(axis=0).round(3))

# ---- 5.5 Matriks fitur akhir ----
X_all = scaler.transform(np.log1p(df_clean[selected_features]))
df_scaled = pd.DataFrame(X_all, index=df_clean.index, columns=[f"{c}_scaled" for c in selected_features])
df_final_model = pd.concat(
    [df_clean[['Provinsi', 'Kota/Kabupaten', 'Jenis_Wilayah', 'Split']], df_scaled], axis=1)
print("\n--- MATRIKS FITUR AKHIR SIAP PEMODELAN K-MEANS ---")
print(f"Total wilayah: {df_final_model.shape[0]} | Jumlah fitur: {len(selected_features)} | "
      f"train: {len(idx_train)} | test: {len(idx_test)}")
display(df_final_model.head())
simpan_tabel(df_final_model, 'fitur_akhir_kmeans.csv', index=False)
simpan_tabel(df_clean, 'data_bersih.csv', index=False)


judul("SELESAI")
print(f"Semua grafik & tabel tersimpan di: {OUT_DIR}")
for f in sorted(OUT_DIR.iterdir()):
    print("  -", f.name)