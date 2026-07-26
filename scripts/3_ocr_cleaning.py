"""
Phase 1: OCR cleaning.

Safe to run in parallel: each teammate takes a disjoint file-index range
and runs this independently. No shared state between ranges, so there's
nothing to coordinate here.

Deduplication is NOT done in this script - it needs a single global pass
over all cleaned files (see dedup_global.py) because a duplicate line can
appear in two different people's file ranges.
"""

import os
import re
import glob
import argparse
import unicodedata
from collections import Counter
from typing import Optional

DEVANAGARI_THRESHOLD = 0.80
SINGLE_CHAR_DOMINANCE = 0.60
MAX_TOKEN_LENGTH = 25  # real Nepali words/compounds rarely exceed this; longer = OCR garbage merge
NEPALI_PUNCTUATION = set('।॥,-')
DEVANAGARI_DIGITS = set('०१२३४५६७८९')


def is_devanagari_char(ch: str) -> bool:
    return 0x0900 <= ord(ch) <= 0x097F


def is_devanagari_digit(ch: str) -> bool:
    return ch in DEVANAGARI_DIGITS


def has_long_digit_sequence(text: str, min_length: int = 6) -> bool:
    count = 0
    for ch in text:
        if is_devanagari_digit(ch):
            count += 1
            if count >= min_length:
                return True
        else:
            count = 0
    return False


def has_dominant_char(text: str, threshold: float = SINGLE_CHAR_DOMINANCE) -> bool:
    """True if any single character makes up more than `threshold` of the line."""
    chars = [ch for ch in text if ch != ' ']
    if not chars:
        return False
    most_common_count = Counter(chars).most_common(1)[0][1]
    return (most_common_count / len(chars)) > threshold


def has_oversized_token(text: str, max_length: int = MAX_TOKEN_LENGTH) -> bool:
    """True if any space-separated token exceeds max_length - usually an OCR
    error where spaces were dropped and multiple words got merged into one
    unrealistically long blob."""
    return any(len(tok) > max_length for tok in text.split(' '))


def clean_line(text: str) -> Optional[str]:
    if not text or not text.strip():
        return None

    text = unicodedata.normalize('NFC', text)

    # keep only Devanagari chars, Nepali punctuation, and spaces
    cleaned_chars = [
        ch for ch in text
        if is_devanagari_char(ch) or ch in NEPALI_PUNCTUATION or ch == ' '
    ]
    cleaned = re.sub(r' {2,}', ' ', ''.join(cleaned_chars)).strip()

    if len(cleaned) < 3:
        return None
    if has_dominant_char(cleaned):
        return None
    if has_long_digit_sequence(cleaned, min_length=6):
        return None
    if has_oversized_token(cleaned):
        return None

    devanagari_letters = sum(
        1 for ch in cleaned if is_devanagari_char(ch) and not is_devanagari_digit(ch)
    )
    total_meaningful = len([
        ch for ch in cleaned.replace(' ', '')
        if ch not in NEPALI_PUNCTUATION and not is_devanagari_digit(ch)
    ])
    if total_meaningful == 0:
        return None
    if devanagari_letters / total_meaningful < DEVANAGARI_THRESHOLD:
        return None

    cleaned = cleaned.replace('।।', '।').replace('॥', '।').strip(' ।,-')
    return cleaned if len(cleaned) >= 3 else None


def process_file(input_path: str, output_path: str) -> Counter:
    stats = Counter()
    # Chunks already come from our own utf-8 pipeline, no encoding sniffing needed.
    with open(input_path, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:
        for line in fin:
            stats['total'] += 1
            cleaned = clean_line(line)
            if cleaned is None:
                stats['dropped'] += 1
                continue
            fout.write(cleaned + '\n')
            stats['kept'] += 1
    return stats


def process_range(input_dir: str, output_dir: str, start_idx: int, end_idx: int,
                   pattern: str = "chunk_*.txt"):
    """
    Process files[start_idx:end_idx] (0-indexed, end exclusive) from the
    sorted file list. Assign non-overlapping ranges to teammates to run
    this in parallel, e.g.:
      person 1: --start 0   --end 40
      person 2: --start 40  --end 80
      person 3: --start 80  --end 120
      person 4: --start 120 --end 160
      person 5: --start 160 --end 200
    """
    os.makedirs(output_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(input_dir, pattern)))
    my_files = files[start_idx:end_idx]

    print(f"Processing files [{start_idx}:{end_idx}] -> {len(my_files)} files")

    for i, path in enumerate(my_files, 1):
        basename = os.path.basename(path)
        out_path = os.path.join(output_dir, basename.replace('.txt', '_cleaned.txt'))
        stats = process_file(path, out_path)
        print(f"  [{i}/{len(my_files)}] {basename}: kept {stats['kept']:,} / "
              f"dropped {stats['dropped']:,} of {stats['total']:,}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Phase 1: OCR cleaning (per-teammate file range)")
    parser.add_argument('--input_dir', default='chunks')
    parser.add_argument('--output_dir', default='chunks_cleaned')
    parser.add_argument('--start', type=int, required=True, help='start file index (inclusive)')
    parser.add_argument('--end', type=int, required=True, help='end file index (exclusive)')
    args = parser.parse_args()

    process_range(args.input_dir, args.output_dir, args.start, args.end)