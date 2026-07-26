"""
Diagnostic: inspect what's actually being flagged as duplicate in a single
cleaned file, before trusting the dedup rate across the full 200-file corpus.

Uses an exact in-memory dict - safe for spot-checking ONE file (~750K
lines). Do not run this across all 200 files at once; it's a diagnostic,
not a replacement for dedup_global.py's Bloom filter pass.
"""

import argparse
import random
from collections import Counter


def analyze_file(path: str, sample_size: int = 50, top_n: int = 20, seed: int = 42):
    line_counts = Counter()
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')
            if line:
                line_counts[line] += 1

    total_lines = sum(line_counts.values())
    unique_lines = len(line_counts)
    duplicate_lines = [l for l, c in line_counts.items() if c > 1]
    duplicate_occurrences = total_lines - unique_lines

    print(f"Total lines:            {total_lines:,}")
    print(f"Unique lines:           {unique_lines:,}")
    print(f"Distinct strings that repeat: {len(duplicate_lines):,}")
    print(f"Duplicate occurrences removed by dedup: {duplicate_occurrences:,}")
    print()

    # length distribution of the distinct duplicated strings
    lengths = sorted(len(l) for l in duplicate_lines)
    if lengths:
        n = len(lengths)
        short_count = sum(1 for l in lengths if l <= 5)
        print("Length distribution of duplicated (distinct) strings:")
        print(f"  min: {lengths[0]}, median: {lengths[n // 2]}, max: {lengths[-1]}")
        print(f"  <=5 chars: {short_count:,} ({100 * short_count / n:.1f}%)")
        print()

    print(f"Top {top_n} most frequent lines (highest repeat count):")
    for line, count in line_counts.most_common(top_n):
        print(f"  x{count:<6} len={len(line):<4} {line!r}")
    print()

    random.seed(seed)
    sample = random.sample(duplicate_lines, min(sample_size, len(duplicate_lines)))
    print(f"Random sample of {len(sample)} duplicated lines:")
    for line in sample:
        print(f"  x{line_counts[line]:<6} len={len(line):<4} {line!r}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Diagnose duplicate lines in a single cleaned file")
    parser.add_argument('input_file')
    parser.add_argument('--sample_size', type=int, default=50)
    parser.add_argument('--top_n', type=int, default=20)
    args = parser.parse_args()

    analyze_file(args.input_file, sample_size=args.sample_size, top_n=args.top_n)