# 🗂️ Boring File Tasks — Automated

Five Python scripts that handle the file chores nobody wants to do manually.

---

## Scripts

| # | Script | What It Does |
|---|--------|--------------|
| 1 | `temp_cache_cleaner.py` | Finds and removes old temp and cache files |
| 2 | `nested_zip_extractor.py` | Recursively extracts deeply nested zip archives |
| 3 | `bulk_format_converter.py` | Batch converts images, audio, and documents |
| 4 | `metadata_extractor.py` | Pulls metadata from media files into a CSV |
| 5 | `stale_folder_purger.py` | Removes empty and stale directories |

---

## 1. Stale Temp & Cache Cleaner

Scans system temp and app cache directories for files that haven't been touched in a configurable number of days. Shows a full report before deleting anything.

**No dependencies** — standard library only.

```bash
# Preview what would be cleaned (30-day threshold by default)
python temp_cache_cleaner.py --dry-run

# Clean files not accessed in 14 days
python temp_cache_cleaner.py --days 14

# Skip the confirmation prompt
python temp_cache_cleaner.py --days 30 --auto-confirm

# Scan custom directories instead of system defaults
python temp_cache_cleaner.py --paths /tmp /var/tmp ~/my_cache
```

### Options

| Flag | Description | Default |
|------|-------------|---------|
| `--days N` | Remove files not accessed in N days | 30 |
| `--paths` | Custom directories to scan | System temp/cache paths |
| `--dry-run` | Preview without deleting | — |
| `--auto-confirm` | Skip the yes/no prompt | — |
| `--log FILE` | Path for the deletion log | `cleanup_log.txt` |

---

## 2. Nested Zip Extractor

Recursively extracts zip archives no matter how deep the nesting goes. Flattens everything into a single clean output folder, handles filename conflicts automatically, and generates a manifest of every extracted file.

**No dependencies** — standard library only.

```bash
# Extract all zips in a directory (flattened output)
python nested_zip_extractor.py /path/to/zips

# Specify a custom output folder
python nested_zip_extractor.py /path/to/zips --output /path/to/extracted

# Keep original directory structures from each zip
python nested_zip_extractor.py /path/to/zips --preserve-structure

# Preview what would be extracted
python nested_zip_extractor.py /path/to/zips --dry-run
```

### Options

| Flag | Description | Default |
|------|-------------|---------|
| `--output DIR` | Output directory | `<input>_extracted/` |
| `--preserve-structure` | Keep internal zip folder structure | Flatten |
| `--dry-run` | Preview without extracting | — |

---


