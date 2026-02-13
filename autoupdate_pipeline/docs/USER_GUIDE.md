# 📖 Panduan Pengguna Auto-Update Pipeline

## Daftar Isi
1. [Quick Start](#quick-start)
2. [Menambah Event Baru](#menambah-event-baru)
3. [Mengelola Events](#mengelola-events)
4. [Menjalankan Pipeline](#menjalankan-pipeline)
5. [Rollback Model](#rollback-model)

---

## Quick Start

```bash
# 1. Cek status pipeline
python scripts/check_status.py

# 2. Tambah event gempa baru
python scripts/add_new_event.py add -d 2026-02-10 -s GTO -m Large -a NE

# 3. Validasi pending events
python scripts/add_new_event.py validate

# 4. Jalankan pipeline (jika 20 events tercapai)
python scripts/run_pipeline.py
```

---

## Menambah Event Baru

### Format Perintah
```bash
python scripts/add_new_event.py add -d TANGGAL -s STASIUN -m MAGNITUDE -a AZIMUTH
```

### Parameter

| Parameter | Alias | Nilai Valid |
|-----------|-------|-------------|
| `--date` | `-d` | Format: YYYY-MM-DD |
| `--station` | `-s` | SBG, SCN, KPY, LWA, LPS, SRG, SKB, CLP, YOG, TRT, LUT, ALR, SMI, SRO, TNT, TND, GTO, LWK, PLU, TRD, JYP, AMB, GSI, MLB |
| `--magnitude` | `-m` | Large, Medium, Moderate, Normal |
| `--azimuth` | `-a` | N, NE, E, SE, S, SW, W, NW, Normal |

### Contoh
```bash
# Gempa besar dari Gorontalo arah Timur Laut
python scripts/add_new_event.py add -d 2026-02-15 -s GTO -m Large -a NE

# Gempa medium dari Tretes arah Barat Daya
python scripts/add_new_event.py add -d 2026-02-16 -s TRT -m Medium -a SW

# Kondisi normal (tidak ada gempa)
python scripts/add_new_event.py add -d 2026-02-17 -s MLB -m Normal -a Normal
```

---

## Mengelola Events

### Lihat Semua Events
```bash
# Semua events (pending + validated)
python scripts/add_new_event.py list

# Hanya pending
python scripts/add_new_event.py list --type pending

# Hanya validated
python scripts/add_new_event.py list --type validated
```

### Validasi Pending Events
```bash
python scripts/add_new_event.py validate
```

### Hapus Event
```bash
# Hapus event tertentu (gunakan ID dari list)
python scripts/add_new_event.py delete --id GTO_20260215

# Hapus semua pending events
python scripts/add_new_event.py clear --confirm
```

### Cek Status Pipeline
```bash
python scripts/check_status.py
```

---

## Menjalankan Pipeline

### Automatic Trigger
Pipeline otomatis trigger ketika:
- Validated events ≥ 20, ATAU
- 90 hari sejak training terakhir

### Manual Run
```bash
# Cek apakah siap
python scripts/run_pipeline.py --check-only

# Jalankan pipeline
python scripts/run_pipeline.py

# Force run (meski threshold belum tercapai)
python scripts/run_pipeline.py --force
```

---

## Rollback Model

```bash
# Lihat versi tersedia
python scripts/rollback_model.py --list

# Rollback ke versi terakhir
python scripts/rollback_model.py

# Rollback ke versi spesifik
python scripts/rollback_model.py --version 1.0.0
```

---

## Workflow Lengkap

```
[Add Event] → [Pending] → [Validate] → [Validated]
                                            ↓
                                    (20 events?)
                                            ↓
                              [Train Challenger Model]
                                            ↓
                              [Compare with Champion]
                                            ↓
                    [Challenger Wins?] → [Deploy] → [Archive Old]
                            ↓
                    [Champion Wins] → [Keep Current]
```

---

## Daftar Stasiun Valid

| Kode | Lokasi | Kode | Lokasi |
|------|--------|------|--------|
| SBG | Sabang | LUT | Lombok Utara |
| SCN | Sicincin | ALR | Alor |
| KPY | Kepahiang | SMI | Saumlaki |
| LWA | Liwa | SRO | Sorong |
| LPS | Lampung Selatan | TNT | Ternate |
| SRG | Serang | TND | Tondano |
| SKB | Sukabumi | GTO | Gorontalo |
| CLP | Cilacap | LWK | Luwuk |
| YOG | Yogyakarta | PLU | Palu |
| TRT | Tretes | TRD | Tarakan |
| JYP | Jayapura | AMB | Ambon |
| GSI | Gunungsitoli | MLB | Meulaboh |
