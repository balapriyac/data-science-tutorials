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
## 3. Bulk Format Converter

Converts files between formats in batch. Supports images (via Pillow), audio (via pydub + ffmpeg), and basic document conversions (via python-docx).

```bash
# Convert all PNGs to WebP
python bulk_format_converter.py /path/to/images --format webp

# Convert WAVs to MP3 at 192 kbps
python bulk_format_converter.py /path/to/audio --format mp3 --quality 192

# Convert JPEGs to PNG at quality 90
python bulk_format_converter.py /path/to/photos --format png --quality 90

# Convert DOCX files to TXT
python bulk_format_converter.py /path/to/docs --format txt

# Preview conversions without running them
python bulk_format_converter.py /path/to/files --format webp --dry-run
```

### Supported Conversions

| Category | Input Formats | Output Formats |
|----------|---------------|----------------|
| Images | PNG, JPG, BMP, GIF, TIFF, WebP, ICO, PPM | PNG, JPG, BMP, GIF, TIFF, WebP |
| Audio | MP3, WAV, OGG, FLAC, AAC, WMA, M4A | MP3, WAV, OGG, FLAC, AAC |
| Documents | DOCX, TXT | DOCX, TXT |

### Options

| Flag | Description | Default |
|------|-------------|---------|
| `--format FORMAT` | Target format (**required**) | — |
| `--quality N` | Quality (images) or bitrate in kbps (audio) | Format-specific defaults |
| `--output DIR` | Output directory | `<input>/converted/` |
| `--dry-run` | Preview without converting | — |
| `--no-recursive` | Only process top-level files | Scan subdirectories |

---

## 4. Media Metadata Extractor

Extracts metadata from image, audio, and video files and exports everything to a single CSV. Handles EXIF data (including GPS), audio tags, and video stream specs.

```bash
# Harvest metadata from all media in a folder
python 4_metadata_harvester.py /path/to/media

# Save to a specific CSV file
python metadata_extractor.py /path/to/media --output report.csv

# Only process images
python metadata_extractor.py /path/to/media --type images

# Process audio and video only
python metadata_extractor.py /path/to/media --type audio --type video
```

### Options

| Flag | Description | Default |
|------|-------------|---------|
| `--output FILE` | Output CSV path | `<input>/metadata.csv` |
| `--type TYPE` | Filter by type: `images`, `audio`, `video`. Repeatable. | All types |

### What Gets Extracted

- **Images:** Dimensions, format, camera make/model, date taken, GPS coordinates, aperture, ISO, focal length, and more EXIF fields.
- **Audio:** Duration, bitrate, sample rate, channels, title, artist, album, genre, year, track number.
- **Video:** Duration, container format, video codec, resolution, FPS, color space, audio codec, subtitle tracks, and any embedded tags.

---






