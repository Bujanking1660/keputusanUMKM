# 🏪 Sistem Keputusan Bantuan UMK — Algoritma k-Nearest Neighbors (kNN)

> Simulasi & visualisasi interaktif perhitungan jarak Euclidean berbasis algoritma **kNN** untuk mendukung keputusan pemberian bantuan modal kepada Usaha Mikro dan Kecil (UMK).

---

## 📌 Deskripsi Proyek

Proyek ini mengimplementasikan algoritma **k-Nearest Neighbors (kNN)** untuk mengklasifikasikan kelayakan UMK dalam menerima bantuan modal usaha. Sistem menggunakan 4 fitur utama sebagai kriteria penilaian dan menghasilkan keputusan berupa:

| Keputusan | Keterangan |
|-----------|------------|
| ✅ **YA** | UMK layak menerima bantuan modal |
| ⏳ **TUNDA** | Perlu evaluasi lebih lanjut |
| ❌ **TIDAK** | UMK tidak memenuhi kriteria bantuan |

Hasil klasifikasi divisualisasikan dalam sebuah **dashboard HTML interaktif** yang menampilkan langkah-langkah perhitungan jarak secara transparan dan dapat diekspor ke PNG.

---

## ✨ Fitur Utama

- 🔢 **Perhitungan kNN Step-by-Step** — Menampilkan formula jarak kuadrat Euclidean (D²) untuk setiap data latih terhadap data uji
- 📊 **Dashboard Interaktif** — Antarmuka web modern dengan tab per UMK uji dan toggle parameter `k` (k=3 / k=5)
- 🏷️ **Visualisasi Voting** — Menampilkan hasil voting tetangga terdekat beserta keputusan akhir secara real-time
- 🔄 **Tie-Breaker Otomatis** — Menangani hasil seri voting dengan strategi *closest neighbor*
- 📤 **Export PNG** — Ekspor tampilan tabel perhitungan langsung dari browser
- 📓 **Jupyter Notebook** — Tersedia versi notebook (`main.ipynb`) untuk eksplorasi interaktif

---

## 🗂️ Struktur Proyek

```
keputusanUMKM/
│
├── 📄 main.py               # Skrip utama: kalkulasi kNN + generator dashboard HTML
├── 📓 main.ipynb            # Jupyter Notebook versi interaktif
├── 🌐 dashboard.html        # Dashboard HTML interaktif (output utama)
├── 🧪 test_knn.py           # Unit test algoritma kNN
├── 🔧 create_notebook.py    # Helper: generator file notebook
│
├── 📁 data/
│   └── data.csv             # Dataset UMK latih (16 data)
│
└── 📁 output_export/        # Hasil export PNG dari dashboard
    ├── kNN_BukidGraffer_k3.png
    ├── kNN_BukidGraffer_k5.png
    ├── kNN_UmkBandeng_k3.png
    ├── kNN_UmkBandeng_k5.png
    ├── kNN_UmkPamurbaya_k3.png
    └── kNN_UmkPamurbaya_k5.png
```

---

## 📊 Dataset

Dataset latih terdiri dari **16 data UMK** dengan 4 fitur numerik:

| Fitur | Variabel | Keterangan |
|-------|----------|------------|
| X1 | `Lama_Usaha` | Lama usaha berjalan (dalam tahun) |
| X2 | `Jumlah_Pekerja` | Jumlah tenaga kerja |
| X3 | `Omzet` | Omzet usaha (dalam juta rupiah) |
| X4 | `Jumlah_Aset` | Jumlah aset yang dimiliki |

**Label Kelas:** `Hasil_Keputusan` → `YA` / `TUNDA` / `TIDAK`

### Contoh Data Latih (`data/data.csv`)

| Nama UKM | Lama Usaha | Jml Pekerja | Omzet | Jml Aset | Keputusan |
|----------|-----------|-------------|-------|----------|-----------|
| Sanggar Azalea | 2 | 14 | 3 | 2 | TUNDA |
| Kedurus Sejahtera | 6 | 7 | 5 | 4 | TIDAK |
| Maju Jaya Sablon | 5 | 1 | 3 | 2 | YA |
| Batik Semanggi | 6 | 1 | 2 | 3 | YA |
| ... | ... | ... | ... | ... | ... |

### Data Uji (Query)

| Nama UMK | Lama Usaha | Jml Pekerja | Omzet | Jml Aset |
|----------|-----------|-------------|-------|----------|
| Umk Pamurbaya | 4 | 15 | 4 | 6 |
| Umk Bandeng | 3 | 28 | 4 | 10 |
| Bukid Graffer | 2 | 12 | 1 | 3 |

---

## ⚙️ Cara Penggunaan

### Prasyarat

```bash
pip install pandas numpy
```

### 1. Jalankan Skrip Utama

```bash
python main.py
```

Skrip akan:
1. Membaca data latih dari `data/data.csv`
2. Menghitung jarak Euclidean kuadrat untuk setiap data uji terhadap semua data latih
3. Menentukan keputusan berdasarkan majority voting (k=3 dan k=5)
4. Mencetak tabel simulasi perhitungan ke konsol
5. Menghasilkan file `dashboard.html`

### 2. Buka Dashboard

Buka file `dashboard.html` di browser favorit Anda:

```bash
start dashboard.html        # Windows
open dashboard.html         # macOS
xdg-open dashboard.html     # Linux
```

### 3. Jalankan via Jupyter Notebook

```bash
jupyter notebook main.ipynb
```

---

## 🧮 Cara Kerja Algoritma

### Rumus Jarak Euclidean Kuadrat (D²)

Untuk setiap data latih $x_i$ dan data uji $q$:

$$D^2(x_i, q) = \sum_{j=1}^{4}(x_{ij} - q_j)^2$$

### Alur Klasifikasi

```
Data Uji (Query)
      │
      ▼
Hitung D² ke semua data latih
      │
      ▼
Urutkan berdasarkan jarak terkecil
      │
      ▼
Ambil K tetangga terdekat (k=3 atau k=5)
      │
      ▼
Majority Voting → Jika seri: pilih label dari tetangga terdekat
      │
      ▼
Keputusan: YA / TUNDA / TIDAK
```

### Contoh Perhitungan (Umk Pamurbaya, k=3)

Query: `(4, 15, 4, 6)`

```
Sanggar Azalea:    (2-4)² + (14-15)² + (3-4)² + (2-6)² = 4+1+1+16 = 22
Kedurus Sejahtera: (6-4)² + (7-15)²  + (5-4)² + (4-6)² = 4+64+1+4 = 73
...
```

---

## 🌐 Dashboard Interaktif

Dashboard web (`dashboard.html`) menyediakan:

- **Tab navigasi** per UMK uji (Umk Pamurbaya, Umk Bandeng, Bukid Graffer)
- **Toggle parameter k** (k=3 / k=5) dengan pembaruan otomatis seluruh tabel
- **Profil UMK Uji** — menampilkan nilai fitur query secara visual
- **Kotak Keputusan** — hasil akhir klasifikasi dengan warna kode
- **Tabel Perhitungan** — formula D² lengkap per baris data latih
- **Tabel Ringkasan** — keputusan akhir semua data uji sekaligus
- **Export PNG** — simpan tampilan sebagai gambar beresolusi tinggi

### Tampilan Kode Warna Keputusan

| Keputusan | Warna |
|-----------|-------|
| YA | 🟢 Hijau |
| TUNDA | 🟡 Oranye/Kuning |
| TIDAK | 🔴 Merah |

---

## 🧪 Testing

Jalankan unit test untuk memvalidasi implementasi kNN:

```bash
python test_knn.py
```

---

## 🛠️ Teknologi

| Komponen | Teknologi |
|----------|-----------|
| Backend / Logika | Python 3, NumPy, Pandas |
| Frontend Dashboard | HTML5, CSS3 Vanilla, JavaScript |
| Tipografi | Google Fonts (Outfit, Plus Jakarta Sans) |
| Export Gambar | [html2canvas](https://html2canvas.hertzen.com/) |
| Notebook | Jupyter Notebook |

---

## 📋 Output Contoh (Konsol)

```
====================================================================================================
ANALISIS kNN UNTUK: Umk Pamurbaya
Data Uji: Lama Usaha=4, Pekerja=15, Omzet=4, Aset=6
Keputusan (k=3): TUNDA (Votes: {'TUNDA': 2, 'TIDAK': 1} - UNANIMOUS/MAJORITY)
Keputusan (k=5): TUNDA (Votes: {'TUNDA': 3, 'TIDAK': 2} - UNANIMOUS/MAJORITY)
----------------------------------------------------------------------------------------------------
```

---

## 📄 Lisensi

Proyek ini dibuat untuk keperluan akademis/edukasi.  
Bebas digunakan dan dimodifikasi dengan menyertakan atribusi.

---

<div align="center">
  <sub>Dibuat dengan ❤️ untuk mendukung pengambilan keputusan UMKM yang transparan dan berbasis data</sub>
</div>
