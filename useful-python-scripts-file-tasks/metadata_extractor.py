"""
Media Metadata Extractor
------------------------
Scans a directory of media files (images, audio, video), extracts
all available metadata from each file, and exports everything to a
clean CSV. Supports EXIF data from images, tags from audio files,
and technical specs from video files.

Usage:
    python metadata_extractor.py /path/to/media
    python metadata_extractor.py /path/to/media --output metadata.csv
    python metadata_extractor.py /path/to/media --type images
    python metadata_extractor.py /path/to/media --type audio --type video

Dependencies:
    pip install Pillow piexif mutagen

    For video metadata: ffprobe is required (comes with ffmpeg)
        https://ffmpeg.org/download.html
"""

import os
import sys
import csv
import subprocess
import argparse
import json
from pathlib import Path
from datetime import datetime
from collections import OrderedDict


# ---------------------------------------------------------------------------
# Format Definitions
# ---------------------------------------------------------------------------

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".tif", ".webp"}
AUDIO_EXTENSIONS = {".mp3", ".wav", ".ogg", ".flac", ".aac", ".wma", ".m4a", ".opus"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov", ".wmv", ".webm", ".flv", ".m4v"}

# EXIF tag ID -> human-readable name (common subset)
EXIF_TAG_NAMES = {
    0x010F: "Camera Make",
    0x0110: "Camera Model",
    0x0112: "Orientation",
    0x011A: "X Resolution",
    0x011B: "Y Resolution",
    0x0128: "Resolution Unit",
    0x0131: "Software",
    0x0132: "Date/Time",
    0x8769: "EXIF IFD",
    0x9000: "EXIF Version",
    0x9003: "Date/Time Original",
    0x9004: "Date/Time Digitized",
    0x9201: "Shutter Speed",
    0x9202: "Aperture",
    0x9203: "ISO",
    0x9204: "Brightness",
    0x9207: "Metering Mode",
    0x9209: "Flash",
    0x920A: "Focal Length",
    0xA001: "Color Space",
    0xA002: "Exif Image Width",
    0xA003: "Exif Image Height",
    0x8825: "GPS IFD",
    0x0001: "GPS Latitude Ref",
    0x0002: "GPS Latitude",
    0x0003: "GPS Longitude Ref",
    0x0004: "GPS Longitude",
    0x0005: "GPS Altitude Ref",
    0x0006: "GPS Altitude",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def format_size(size_bytes: int) -> str:
    """Human-readable file size."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} PB"


def detect_media_type(ext: str) -> str | None:
    """Classify file extension as image, audio, video, or None."""
    ext = ext.lower()
    if ext in IMAGE_EXTENSIONS:
        return "image"
    if ext in AUDIO_EXTENSIONS:
        return "audio"
    if ext in VIDEO_EXTENSIONS:
        return "video"
    return None


def gps_dms_to_decimal(dms: tuple, ref: str) -> float:
    """
    Convert GPS degrees/minutes/seconds to decimal degrees.
    dms is a tuple like ((d, 1), (m, 1), (s, 100))
    ref is 'N', 'S', 'E', or 'W'
    """
    try:
        degrees = dms[0][0] / dms[0][1]
        minutes = dms[1][0] / dms[1][1]
        seconds = dms[2][0] / dms[2][1]
        decimal = degrees + minutes / 60 + seconds / 3600
        if ref in ("S", "W"):
            decimal *= -1
        return round(decimal, 7)
    except (ZeroDivisionError, IndexError, TypeError):
        return 0.0


# ---------------------------------------------------------------------------
# Extractors
# ---------------------------------------------------------------------------

def extract_image_metadata(file_path: Path) -> OrderedDict:
    """Extract EXIF and basic metadata from an image file."""
    from PIL import Image

    meta = OrderedDict()
    meta["File Name"] = file_path.name
    meta["File Size"] = format_size(file_path.stat().st_size)
    meta["File Type"] = "Image"

    try:
        img = Image.open(file_path)
        meta["Dimensions"] = f"{img.size[0]} x {img.size[1]}"
        meta["Mode"] = img.mode
        meta["Format"] = img.format

        # Try to pull EXIF data
        try:
            import piexif
            exif_dict = piexif.load(str(file_path))

            # Base IFD (0th)
            for tag_id, value in exif_dict.get("0th", {}).items():
                tag_name = EXIF_TAG_NAMES.get(tag_id, f"Tag_{tag_id}")
                if isinstance(value, bytes):
                    try:
                        value = value.decode("utf-8").rstrip("\x00")
                    except UnicodeDecodeError:
                        value = value.hex()
                meta[tag_name] = value

            # EXIF IFD
            for tag_id, value in exif_dict.get("Exif", {}).items():
                tag_name = EXIF_TAG_NAMES.get(tag_id, f"Exif_{tag_id}")
                if isinstance(value, bytes):
                    try:
                        value = value.decode("utf-8").rstrip("\x00")
                    except UnicodeDecodeError:
                        value = value.hex()
                meta[tag_name] = value

            # GPS data
            gps = exif_dict.get("GPS", {})
            if gps:
                lat_ref = gps.get(1, b"").decode("utf-8") if isinstance(gps.get(1), bytes) else str(gps.get(1, ""))
                lon_ref = gps.get(3, b"").decode("utf-8") if isinstance(gps.get(3), bytes) else str(gps.get(3, ""))

                if 2 in gps and lat_ref:
                    meta["GPS Latitude"] = gps_dms_to_decimal(gps[2], lat_ref)
                if 4 in gps and lon_ref:
                    meta["GPS Longitude"] = gps_dms_to_decimal(gps[4], lon_ref)
                if 6 in gps:
                    alt = gps[6]
                    meta["GPS Altitude (m)"] = round(alt[0] / alt[1], 2) if alt[1] != 0 else 0

        except Exception:
            meta["EXIF"] = "Not available or unsupported"

    except Exception as e:
        meta["Error"] = str(e)

    return meta


def extract_audio_metadata(file_path: Path) -> OrderedDict:
    """Extract tags and technical info from an audio file using mutagen."""
    import mutagen

    meta = OrderedDict()
    meta["File Name"] = file_path.name
    meta["File Size"] = format_size(file_path.stat().st_size)
    meta["File Type"] = "Audio"

    try:
        audio = mutagen.File(file_path)

        if audio is None:
            meta["Error"] = "Could not read audio file"
            return meta

        # Technical info
        if hasattr(audio, "info"):
            info = audio.info
            if hasattr(info, "length"):
                length = info.length
                minutes, seconds = divmod(int(length), 60)
                meta["Duration"] = f"{minutes}:{seconds:02d}"
            if hasattr(info, "bitrate"):
                meta["Bitrate"] = f"{info.bitrate} bps"
            if hasattr(info, "sample_rate"):
                meta["Sample Rate"] = f"{info.sample_rate} Hz"
            if hasattr(info, "channels"):
                meta["Channels"] = info.channels

        # Tag info (normalized across formats)
        tag_map = {
            "title":   ["title", "TIT2", "TITLE"],
            "artist":  ["artist", "TPE1", "ARTIST"],
            "album":   ["album", "TALB", "ALBUM"],
            "genre":   ["genre", "TCON", "GENRE"],
            "year":    ["date", "TDRC", "year", "YEAR"],
            "track":   ["tracknumber", "TRCK", "track"],
            "comment": ["comment", "COMM", "COMMENT"],
        }

        if audio.tags:
            for display_name, possible_keys in tag_map.items():
                for key in possible_keys:
                    # Try exact key
                    if key in audio.tags:
                        val = audio.tags[key]
                        if isinstance(val, list):
                            val = val[0]
                        meta[display_name.title()] = str(val)
                        break
                    # Try case-insensitive
                    for tag_key in audio.tags:
                        if tag_key.lower().startswith(key.lower()):
                            val = audio.tags[tag_key]
                            if isinstance(val, list):
                                val = val[0]
                            meta[display_name.title()] = str(val)
                            break

    except Exception as e:
        meta["Error"] = str(e)

    return meta


def extract_video_metadata(file_path: Path) -> OrderedDict:
    """Extract video metadata using ffprobe."""
    meta = OrderedDict()
    meta["File Name"] = file_path.name
    meta["File Size"] = format_size(file_path.stat().st_size)
    meta["File Type"] = "Video"

    try:
        # Run ffprobe to get JSON metadata
        cmd = [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_streams",
            "-show_format",
            str(file_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)

        if result.returncode != 0:
            meta["Error"] = "ffprobe failed — is ffmpeg installed?"
            return meta

        data = json.loads(result.stdout)

        # Format-level info
        fmt = data.get("format", {})
        if "duration" in fmt:
            duration = float(fmt["duration"])
            minutes, seconds = divmod(int(duration), 60)
            meta["Duration"] = f"{minutes}:{seconds:02d}"
        if "bit_rate" in fmt:
            meta["Overall Bitrate"] = f"{int(fmt['bit_rate']) // 1000} kbps"
        if "format_name" in fmt:
            meta["Container"] = fmt["format_name"]

        # Stream-level info
        for i, stream in enumerate(data.get("streams", [])):
            codec_type = stream.get("codec_type", "unknown")
            prefix = f"Stream {i} ({codec_type.title()})"

            if codec_type == "video":
                meta[f"{prefix} Codec"] = stream.get("codec_name", "unknown")
                meta[f"{prefix} Resolution"] = f"{stream.get('width', '?')}x{stream.get('height', '?')}"
                meta[f"{prefix} FPS"] = stream.get("r_frame_rate", "unknown")
                meta[f"{prefix} Bit Rate"] = f"{int(stream.get('bit_rate', 0)) // 1000} kbps" if stream.get("bit_rate") else "unknown"
                meta[f"{prefix} Color Space"] = stream.get("pix_fmt", "unknown")

            elif codec_type == "audio":
                meta[f"{prefix} Codec"] = stream.get("codec_name", "unknown")
                meta[f"{prefix} Sample Rate"] = f"{stream.get('sample_rate', '?')} Hz"
                meta[f"{prefix} Channels"] = stream.get("channels", "unknown")
                meta[f"{prefix} Bit Rate"] = f"{int(stream.get('bit_rate', 0)) // 1000} kbps" if stream.get("bit_rate") else "unknown"

            elif codec_type == "subtitle":
                meta[f"{prefix} Codec"] = stream.get("codec_name", "unknown")
                meta[f"{prefix} Language"] = stream.get("tags", {}).get("language", "unknown")

        # Tags
        tags = fmt.get("tags", {})
        if tags:
            for key, value in tags.items():
                meta[f"Tag: {key.title()}"] = value

    except FileNotFoundError:
        meta["Error"] = "ffprobe not found — install ffmpeg"
    except subprocess.TimeoutExpired:
        meta["Error"] = "ffprobe timed out"
    except Exception as e:
        meta["Error"] = str(e)

    return meta


# ---------------------------------------------------------------------------
# Core Logic
# ---------------------------------------------------------------------------

def harvest(input_dir: Path, file_types: list[str] | None = None) -> list[OrderedDict]:
    """
    Scan input_dir for media files and extract metadata from each.
    file_types filters to specific types: ['images'], ['audio'], ['video'], or None for all.
    """
    all_results = []
    type_counts = {"image": 0, "audio": 0, "video": 0, "skipped": 0}

    extractors = {
        "image": extract_image_metadata,
        "audio": extract_audio_metadata,
        "video": extract_video_metadata,
    }

    # Normalize type filter
    allowed_types = None
    if file_types:
        allowed_types = set()
        for t in file_types:
            t = t.lower().rstrip("s")  # "images" -> "image"
            allowed_types.add(t)

    for file_path in sorted(input_dir.rglob("*")):
        if not file_path.is_file():
            continue

        media_type = detect_media_type(file_path.suffix)
        if media_type is None:
            type_counts["skipped"] += 1
            continue

        if allowed_types and media_type not in allowed_types:
            type_counts["skipped"] += 1
            continue

        print(f"   [{media_type.upper():>5}] {file_path.name}", end=" ", flush=True)

        try:
            extractor = extractors[media_type]
            meta = extractor(file_path)
            all_results.append(meta)
            type_counts[media_type] += 1
            print("✓")
        except Exception as e:
            print(f"✗ ({e})")
            error_meta = OrderedDict()
            error_meta["File Name"] = file_path.name
            error_meta["File Type"] = media_type.title()
            error_meta["Error"] = str(e)
            all_results.append(error_meta)
            type_counts[media_type] += 1

    return all_results, type_counts


def export_csv(results: list[OrderedDict], output_path: Path) -> None:
    """Export metadata results to CSV, normalizing all fields across file types."""
    if not results:
        return

    # Collect all unique keys in order of appearance
    all_keys = []
    seen = set()
    for row in results:
        for key in row:
            if key not in seen:
                all_keys.append(key)
                seen.add(key)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        for row in results:
            writer.writerow(row)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extract metadata from image, audio, and video files and export to CSV."
    )
    parser.add_argument(
        "input", type=str,
        help="Directory containing media files to scan"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output CSV file path (default: 'metadata.csv' in the input directory)"
    )
    parser.add_argument(
        "--type", dest="types", action="append", default=None,
        choices=["images", "audio", "video"],
        help="Limit to specific file types. Can be used multiple times. (default: all)"
    )
    args = parser.parse_args()

    input_dir = Path(args.input).resolve()
    if not input_dir.exists():
        print(f"❌ Directory not found: {input_dir}")
        sys.exit(1)

    output_path = Path(args.output) if args.output else input_dir / "metadata.csv"

    print(f"\n📂 Scanning: {input_dir}")
    if args.types:
        print(f"🎯 Types:    {', '.join(args.types)}")
    print(f"📄 Output:   {output_path}")
    print()

    # Harvest
    results, type_counts = harvest(input_dir, args.types)

    if not results:
        print("\n⚠️  No media files found.")
        return

    # Export
    export_csv(results, output_path)

    # Summary
    errors = sum(1 for r in results if "Error" in r)
    print(f"\n{'=' * 50}")
    print(f"  ✅ Harvest complete")
    print(f"  Images processed:  {type_counts['image']}")
    print(f"  Audio processed:   {type_counts['audio']}")
    print(f"  Video processed:   {type_counts['video']}")
    print(f"  Skipped:           {type_counts['skipped']}")
    if errors:
        print(f"  Errors:            {errors}")
    print(f"  CSV saved to:      {output_path}")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
