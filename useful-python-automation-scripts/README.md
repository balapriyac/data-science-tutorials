# Python Automation Scripts

5 practical Python scripts to automate boring everyday tasks and save hours of manual work.

## 📦 Getting Started

### Prerequisites
- Python 3.11 or higher
- pip (Python package manager)


### Requirements.txt
```
watchdog>=2.1.0    # For file monitoring
Pillow>=9.0.0      # For image conversion
pandas>=1.3.0      # For data file conversion
openpyxl>=3.0.0    # For Excel files
PyPDF2>=3.0.0      # For PDF text extraction
```

Optional dependencies (install only if needed):
```bash
pip install pytesseract   # For OCR in screenshots (requires Tesseract OCR installed)
```

## 🚀 Scripts Overview

### 1. Automatic File Organizer
Automatically sorts messy folders by file type and date.

**Usage:**
```bash
python file_organizer.py
```

**What it does:**
- Organizes Downloads, Desktop, or any folder
- Groups files by type (Images, Documents, Videos, etc.)
- Optional date-based subfolders
- Handles duplicate filenames
- Maintains organization log

**Example:**
```bash
# Organize Downloads folder
python file_organizer.py ~/Downloads

# Or run interactively
python file_organizer.py
# Then select: 1. Show statistics, 2. Organize files
```

---

### 2. Batch File Renamer
Rename hundreds of files at once with flexible patterns.

**Usage:**
```bash
python batch_renamer.py [directory]
```

**What it does:**
- Add prefixes/suffixes
- Sequential numbering
- Date-based naming
- Text replacement (with regex)
- Case conversion
- Preview before applying

**Example:**
```bash
# Rename all vacation photos
python batch_renamer.py ~/Photos/Vacation
# Select files: *.jpg
# Add prefix: "Bali_2025_"
# Add sequential numbers: 001, 002, 003...
```

---

### 3. Smart Backup Manager
Create intelligent incremental backups automatically.

**Usage:**
```bash
python backup_manager.py <source_dir> <backup_dir>
```

**What it does:**
- Only backs up changed files (incremental)
- Compresses backups to save space
- Maintains multiple backup versions
- Automatic cleanup of old backups
- Easy restoration

**Example:**
```bash
# Backup Documents folder
python backup_manager.py ~/Documents ~/Backups/Documents

# Interactive menu:
# 1. Create new backup
# 2. List backups
# 3. Restore backup
```

**Tip:** Schedule with cron/Task Scheduler for automatic daily backups.

---

### 4. Duplicate File Finder
Find and remove duplicate files to free up disk space.

**Usage:**
```bash
python duplicate_finder.py <directory1> [directory2] ...
```

**What it does:**
- Scans directories for exact duplicates
- Uses hash comparison (not just filenames)
- Shows wasted space
- Safe deletion with preview
- Exports detailed reports

**Example:**
```bash
# Find duplicates in Downloads and Documents
python duplicate_finder.py ~/Downloads ~/Documents

# Shows:
# - Duplicate sets: 47
# - Wasted space: 3.2 GB
# - Interactive deletion options
```

---

### 5. Desktop Screenshot Organizer
Automatically organize screenshots by date.

**Usage:**
```bash
python screenshot_organizer.py [screenshots_directory]
```

**What it does:**
- Organizes screenshots by year/month
- Archives old screenshots (30+ days)
- Extracts dates from filenames
- Cleans up desktop automatically
- Searchable organization

**Example:**
```bash
# Organize Desktop screenshots
python screenshot_organizer.py ~/Desktop

# Or run interactively
python screenshot_organizer.py
# 1. Preview organization
# 2. Organize screenshots
# 3. Show statistics
```
---
