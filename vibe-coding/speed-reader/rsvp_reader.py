#!/usr/bin/env python3
"""
RSVP Speed Reading App
A command-line speed reading application using Rapid Serial Visual Presentation
"""

import sys
import time
import argparse
import threading
import termios
import tty
import select
from pathlib import Path

class RSVPReader:
    def __init__(self, text, wpm=150, chunk_size=1):
        self.text = text
        self.wpm = wpm  # words per minute
        self.chunk_size = chunk_size  # words per chunk
        self.words = self._prepare_text()
        self.current_index = 0
        self.is_paused = False
        self.is_running = False
        self.delay = 60.0 / (wpm * chunk_size)  # seconds between words/chunks
        
    def _prepare_text(self):
        """Clean and split text into words"""
        # Remove extra whitespace and split into words
        words = ' '.join(self.text.split()).split()
        
        # Group words into chunks if chunk_size > 1
        if self.chunk_size > 1:
            chunks = []
            for i in range(0, len(words), self.chunk_size):
                chunk = ' '.join(words[i:i + self.chunk_size])
                chunks.append(chunk)
            return chunks
        return words
    
    def _clear_screen(self):
        """Clear terminal screen"""
        print('\033[2J\033[H', end='')
    
    def _get_terminal_size(self):
        """Get terminal dimensions"""
        try:
            import shutil
            cols, rows = shutil.get_terminal_size()
            return cols, rows
        except:
            return 80, 24  # fallback
    
    def _center_text(self, text, width=None):
        """Center text on screen"""
        if width is None:
            width, _ = self._get_terminal_size()
        padding = max(0, (width - len(text)) // 2)
        return ' ' * padding + text
    
    def _get_display_area(self):
        """Get the 40% center rectangle dimensions"""
        cols, rows = self._get_terminal_size()
        
        # Use 40% of screen dimensions
        display_width = int(cols * 0.4)
        display_height = int(rows * 0.4)
        
        # Center the display area
        start_col = (cols - display_width) // 2
        start_row = (rows - display_height) // 2
        
        return start_col, start_row, display_width, display_height
    
    def _create_large_text(self, text):
        """Create larger text using spacing and bold"""
        # Use bold and character spacing for larger appearance
        return f"\033[1m{' '.join(text)}\033[0m"
    
    def _draw_static_interface(self):
        """Draw the static interface elements (called once)"""
        self._clear_screen()
        
        start_col, start_row, display_width, display_height = self._get_display_area()
        cols, rows = self._get_terminal_size()
        
        # Draw the static elements
        # Top info area
        info_row = start_row - 2
        if info_row > 0:
            speed_text = f"Speed: {self.wpm} WPM"
            progress_text = f"Progress: {self.current_index + 1}/{len(self.words)}"
            
            # Position cursor and draw info
            print(f'\033[{info_row};{start_col}H{speed_text}')
            print(f'\033[{info_row};{start_col + display_width - len(progress_text)}H{progress_text}')
        
        # Bottom controls area
        controls_row = start_row + display_height + 2
        if controls_row < rows:
            controls = [
                "SPACE = Pause/Resume",
                "↑/↓ = Speed Up/Down", 
                "←/→ = Previous/Next",
                "Q = Quit"
            ]
            
            for i, control in enumerate(controls):
                control_row = controls_row + i
                if control_row < rows:
                    print(f'\033[{control_row};{start_col}H{control}')
        
        # Draw border around display area (optional light border)
        for i in range(display_height):
            row = start_row + i
            if row < rows:
                # Left and right border positions
                print(f'\033[{row};{start_col-1}H│')
                print(f'\033[{row};{start_col + display_width}H│')
        
        # Top and bottom borders
        border_line = '─' * display_width
        if start_row > 0:
            print(f'\033[{start_row-1};{start_col}H{border_line}')
        if start_row + display_height < rows:
            print(f'\033[{start_row + display_height};{start_col}H{border_line}')
    
    def _display_word(self, word):
        """Display current word in the center area"""
        start_col, start_row, display_width, display_height = self._get_display_area()
        
        # Clear only the word display area
        word_row = start_row + display_height // 2
        
        # Clear the word line
        print(f'\033[{word_row};{start_col}H{" " * display_width}')
        
        # Create large text
        large_word = self._create_large_text(word)
        
        # Center the word in the display area
        word_start_col = start_col + (display_width - len(large_word)) // 2
        if word_start_col < start_col:
            word_start_col = start_col
        
        # Display the word
        print(f'\033[{word_row};{word_start_col}H{large_word}')
        
        # Update speed display if it changed
        info_row = start_row - 2
        if info_row > 0:
            speed_text = f"Speed: {self.wpm} WPM"
            print(f'\033[{info_row};{start_col}H{speed_text}')
            
            # Update progress
            progress_text = f"Progress: {self.current_index + 1}/{len(self.words)}"
            print(f'\033[{info_row};{start_col + display_width - len(progress_text)}H{progress_text}')
        
        # Show pause status in the display area if paused
        if self.is_paused:
            pause_row = word_row + 2
            pause_text = "PAUSED"
            pause_col = start_col + (display_width - len(pause_text)) // 2
            print(f'\033[{pause_row};{pause_col}H\033[33m{pause_text}\033[0m')
        else:
            # Clear pause text if not paused
            pause_row = word_row + 2
            print(f'\033[{pause_row};{start_col}H{" " * display_width}')
        
        # Move cursor to bottom to avoid interfering
        print(f'\033[{self._get_terminal_size()[1]};1H', end='')
        sys.stdout.flush()
    
    def _get_keyboard_input(self):
        """Non-blocking keyboard input handler"""
        old_settings = termios.tcgetattr(sys.stdin)
        try:
            tty.setraw(sys.stdin.fileno())
            
            while self.is_running:
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    key = sys.stdin.read(1)
                    
                    if key.lower() == 'q':
                        self.is_running = False
                        break
                    elif key == ' ':
                        self.is_paused = not self.is_paused
                    elif key == '\x1b':  # Escape sequence (arrow keys)
                        key2 = sys.stdin.read(1)
                        key3 = sys.stdin.read(1)
                        if key2 == '[':
                            if key3 == 'A':  # Up arrow - increase speed
                                self.wpm = min(self.wpm + 25, 1000)
                                self.delay = 60.0 / (self.wpm * self.chunk_size)
                            elif key3 == 'B':  # Down arrow - decrease speed
                                self.wpm = max(self.wpm - 25, 50)
                                self.delay = 60.0 / (self.wpm * self.chunk_size)
                            elif key3 == 'C':  # Right arrow - next word
                                if self.current_index < len(self.words) - 1:
                                    self.current_index += 1
                            elif key3 == 'D':  # Left arrow - previous word
                                if self.current_index > 0:
                                    self.current_index -= 1
                    
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
    
    def start(self):
        """Start the RSVP reading session"""
        self.is_running = True
        self.current_index = 0
        
        # Draw the static interface once
        self._draw_static_interface()
        
        # Start keyboard input handler in separate thread
        input_thread = threading.Thread(target=self._get_keyboard_input, daemon=True)
        input_thread.start()
        
        try:
            while self.is_running and self.current_index < len(self.words):
                if not self.is_paused:
                    current_word = self.words[self.current_index]
                    self._display_word(current_word)
                    
                    # Adjust delay for longer words (optional feature)
                    word_delay = self.delay
                    if len(current_word) > 8:
                        word_delay *= 1.2
                    elif len(current_word) < 4:
                        word_delay *= 0.8
                    
                    time.sleep(word_delay)
                    
                    if not self.is_paused:  # Check again after sleep
                        self.current_index += 1
                else:
                    # Still update display when paused
                    current_word = self.words[self.current_index]
                    self._display_word(current_word)
                    time.sleep(0.1)
            
            # Reading completed
            if self.is_running:
                self._clear_screen()
                start_col, start_row, display_width, display_height = self._get_display_area()
                completion_row = start_row + display_height // 2
                
                completion_text = "Reading Complete!"
                words_text = f"Words read: {len(self.words)}"
                speed_text = f"Average speed: {self.wpm} WPM"
                
                print(f'\033[{completion_row};{start_col + (display_width - len(completion_text)) // 2}H{completion_text}')
                print(f'\033[{completion_row + 1};{start_col + (display_width - len(words_text)) // 2}H{words_text}')
                print(f'\033[{completion_row + 2};{start_col + (display_width - len(speed_text)) // 2}H{speed_text}')
                
                time.sleep(3)
                
        except KeyboardInterrupt:
            pass
        finally:
            self.is_running = False
            self._clear_screen()
            print("Thanks for using RSVP Reader!")

def load_text_file(file_path):
    """Load text from file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='RSVP Speed Reading App')
    parser.add_argument('input', nargs='?', help='Text file to read or text string')
    parser.add_argument('-w', '--wpm', type=int, default=150, 
                       help='Words per minute (default: 150)')
    parser.add_argument('-c', '--chunk', type=int, default=1,
                       help='Words per chunk (default: 1)')
    parser.add_argument('-f', '--file', action='store_true',
                       help='Treat input as file path')
    
    args = parser.parse_args()
    
    if not args.input:
        # If no input provided, use sample text
        text = """
        Speed reading is a collection of reading methods which attempt to increase rates of reading 
        without substantially reducing comprehension or retention. Methods include chunking and 
        eliminating subvocalization. The many available speed reading training programs include 
        books, videos, software, and seminars.
        """
    elif args.file or Path(args.input).is_file():
        text = load_text_file(args.input)
    else:
        text = args.input
    
    if not text.strip():
        print("Error: No text to read.")
        sys.exit(1)
    
    print("RSVP Speed Reader")
    print("================")
    print(f"Text length: {len(text.split())} words")
    print(f"Reading speed: {args.wpm} WPM")
    print(f"Chunk size: {args.chunk} words")
    print("\nStarting in 3 seconds...")
    print("Use SPACE to pause/resume, arrow keys to control")
    
    time.sleep(3)
    
    reader = RSVPReader(text, wpm=args.wpm, chunk_size=args.chunk)
    reader.start()

if __name__ == "__main__":
    main()

