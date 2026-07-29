"""
Phase 2: global deduplication.

Must run as a SINGLE sequential pass over ALL cleaned files (not split
across teammates) - a duplicate line can appear in two different
people's Phase 1 output, so only a shared, global view catches it.

Uses a Bloom filter instead of an exact hash set: at ~220M lines, an
exact set of md5 hashes would need ~25-30GB RAM, which doesn't fit
Colab free tier. A Bloom filter at 1% false-positive rate needs only
~250-300MB, at the cost of very rarely dropping a genuinely unique
line as if it were a duplicate - an acceptable trade for training data
at this scale.

Run this only after every teammate's Phase 1 range has finished.
"""

import os
import glob
import math
import shutil
import hashlib
import argparse


class BloomFilter:
    def __init__(self, n_items: int, fp_rate: float = 0.01):
        self.size = self._optimal_size(n_items, fp_rate)
        self.hash_count = self._optimal_hash_count(self.size, n_items)
        self.bit_array = bytearray((self.size + 7) // 8)
        print(f"BloomFilter: {self.size:,} bits (~{self.size / 8 / 1024 / 1024:.1f} MB), "
              f"{self.hash_count} hash functions, target fp_rate={fp_rate}")

    @staticmethod
    def _optimal_size(n: int, p: float) -> int:
        m = -(n * math.log(p)) / (math.log(2) ** 2)
        return int(m)

    @staticmethod
    def _optimal_hash_count(m: int, n: int) -> int:
        k = (m / n) * math.log(2)
        return max(1, int(round(k)))

    def _indices(self, item_bytes: bytes):
        # double hashing (Kirsch-Mitzenmacher): derive k indices from 2 base hashes
        h1 = int.from_bytes(hashlib.md5(item_bytes).digest()[:8], 'little')
        h2 = int.from_bytes(hashlib.sha1(item_bytes).digest()[:8], 'little')
        for i in range(self.hash_count):
            yield (h1 + i * h2) % self.size

    def add_and_check(self, item: str) -> bool:
        """Adds `item` to the filter. Returns True if it was (probably) already present."""
        item_bytes = item.encode('utf-8')
        indices = list(self._indices(item_bytes))
        already_present = all(
            self.bit_array[idx // 8] & (1 << (idx % 8)) for idx in indices
        )
        for idx in indices:
            self.bit_array[idx // 8] |= (1 << (idx % 8))
        return already_present


def dedup_all(cleaned_dir: str, output_dir: str, expected_lines: int,
              fp_rate: float = 0.01, pattern: str = "*_cleaned.txt",
              local_staging_dir: str = None):
    os.makedirs(output_dir, exist_ok=True)
    if local_staging_dir:
        os.makedirs(local_staging_dir, exist_ok=True)

    bf = BloomFilter(n_items=expected_lines, fp_rate=fp_rate)

    files = sorted(glob.glob(os.path.join(cleaned_dir, pattern)))
    print(f"Found {len(files)} cleaned files to deduplicate\n")

    total, kept, dup = 0, 0, 0
    for i, path in enumerate(files, 1):
        basename = os.path.basename(path)
        final_out_path = os.path.join(output_dir, basename)
        write_path = os.path.join(local_staging_dir, basename) if local_staging_dir else final_out_path

        with open(path, 'r', encoding='utf-8') as fin, \
             open(write_path, 'w', encoding='utf-8') as fout:
            for line in fin:
                total += 1
                stripped = line.rstrip('\n')
                if not stripped:
                    continue
                if bf.add_and_check(stripped):
                    dup += 1
                    continue
                fout.write(stripped + '\n')
                kept += 1

        if local_staging_dir:
            shutil.copy(write_path, final_out_path)

        print(f"[{i}/{len(files)}] {basename} done "
              f"(running totals - kept: {kept:,}, duplicates: {dup:,})")

    print(f"\nFinal: total lines {total:,}, kept {kept:,}, duplicates removed {dup:,}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Phase 2: global deduplication (Bloom filter)")
    parser.add_argument('--cleaned_dir', required=True,
                         help='folder containing ALL teammates\' Phase 1 *_cleaned.txt output')
    parser.add_argument('--output_dir', required=True,
                         help='where the final deduplicated files are written')
    parser.add_argument('--expected_lines', type=int, default=220_000_000,
                         help='rough total line count across all cleaned files (safe to overestimate)')
    parser.add_argument('--fp_rate', type=float, default=0.01,
                         help='Bloom filter target false-positive rate')
    parser.add_argument('--local_staging_dir', default=None,
                         help='optional local dir (e.g. /content/staging) to write to before '
                              'copying to output_dir - recommended when output_dir is a Drive mount')
    args = parser.parse_args()

    dedup_all(
        cleaned_dir=args.cleaned_dir,
        output_dir=args.output_dir,
        expected_lines=args.expected_lines,
        fp_rate=args.fp_rate,
        local_staging_dir=args.local_staging_dir,
    )