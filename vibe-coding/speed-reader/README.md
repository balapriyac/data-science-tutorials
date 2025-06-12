# RSVP Speed Reader 🚀

A minimalist command-line speed reading application using RSVP (Rapid Serial Visual Presentation) technique. Read faster, comprehend better, and eliminate distractions with this clean terminal-based reader.

## Features

- **Focused Reading**: Words appear one at a time in a fixed position, eliminating eye movement
- **Variable Speed**: Adjustable reading speed from 50 to 1000+ words per minute
- **Real-time Controls**: Pause, navigate, and adjust speed while reading
- **Adaptive Display**: Uses 40% center area of your terminal with clean borders
- **Progress Tracking**: Visual progress indicator and word count
- **Multiple Input Methods**: Read from files or paste text directly
- **Clean Interface**: Static controls, distraction-free design
- **Smart Timing**: Longer words get slightly more display time

## Interface Overview

```
                    Speed: 300 WPM                Progress: 45/150
              ┌────────────────────────────────────────────┐
              │                                            │
              │                                            │
              │              c u r r e n t                 │
              │                                            │
              │                                            │
              └────────────────────────────────────────────┘

              SPACE = Pause/Resume
              ↑/↓ = Speed Up/Down
              ←/→ = Previous/Next
              Q = Quit
```

## 🚀 Quick Start

### Installation

```bash
# Clone or download the script
git clone https://github.com/yourusername/rsvp-speed-reader.git
cd rsvp-speed-reader

# Or download directly
curl -O https://raw.githubusercontent.com/yourusername/rsvp-speed-reader/main/rsvp_reader.py
```

### Basic Usage

```bash
# Read a text file
python3 rsvp_reader.py -f your_document.txt

# Read with custom speed (300 WPM)
python3 rsvp_reader.py -f book.txt -w 300

# Read text directly from command line
python3 rsvp_reader.py "Your text content here to speed read through"

# Read with word chunks (2 words at a time)
python3 rsvp_reader.py -f article.txt -c 2 -w 200

# Try with sample text (no arguments)
python3 rsvp_reader.py

# Make it executable
chmod +x rsvp_reader.py

# Then run
./rsvp_reader.py
```

## 🎮 Controls

While reading, use these keyboard controls:

| Key | Action |
|-----|--------|
| `SPACE` | Pause/Resume reading |
| `↑` | Increase speed by 25 WPM |
| `↓` | Decrease speed by 25 WPM |
| `→` | Skip to next word |
| `←` | Go back to previous word |
| `Q` | Quit application |

## ⚙️ Command Line Options

```bash
python3 rsvp_reader.py [OPTIONS] [INPUT]
```

### Arguments

- `INPUT` - Text file path or direct text string (optional)

### Options

- `-f, --file` - Treat input as file path
- `-w, --wpm INTEGER` - Words per minute (default: 150)
- `-c, --chunk INTEGER` - Words per chunk (default: 1)
- `-h, --help` - Show help message

### Examples with Options

```bash
# Slow reading with larger chunks
python3 rsvp_reader.py -f textbook.txt -w 120 -c 3

# Fast reading for familiar content
python3 rsvp_reader.py -f news.txt -w 500

# Medium pace for technical content
python3 rsvp_reader.py -f documentation.txt -w 200 -c 2
```

## 📋 Requirements

- **Python 3.6+** - No additional packages required
- **Terminal** - Works in any terminal that supports ANSI escape codes
- **Operating System** - Linux, macOS, Windows (with proper terminal)


### Recommended Starting Points

- **New to speed reading**: Start at 150 WPM
- **Experienced reader**: Start at 250 WPM  
- **Technical content**: Use 120-180 WPM with chunks
- **Light reading**: 300-400 WPM is comfortable


## 🐛 Troubleshooting

### Common Issues

**Terminal not clearing properly**
```bash
# Reset terminal
reset
# Or try
tput reset
```

**Keyboard controls not responding**
- Ensure terminal has focus
- Try running in different terminal emulator
- Check if running over SSH with proper terminal settings

**Words appearing off-center**
- Resize terminal window
- Try different terminal dimensions
- App adapts to terminal size automatically

**Speed too fast/slow to start**
```bash
# Start with comfortable speed
python3 rsvp_reader.py -f file.txt -w 150

# Adjust in real-time with ↑/↓ keys
```

## Acknowledgments

- Built through collaborative "vibe coding"
- Inspired by RSVP research and speed reading techniques

Happy Speed Reading! 📚⚡
