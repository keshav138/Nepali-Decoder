import unicodedata

import src.paths as paths
from pathlib import Path

INPUT_DIR = paths.CHUNKS

OUTPUT_DIR = paths.PROJECT_ROOT / "Normalized_Data"

OUTPUT_DIR.mkdir(exist_ok=True)

count = 0

for file in sorted(INPUT_DIR.glob('*.txt')):
    print(f'Processing file_no: {file.name}')

    with open(file, 'r', encoding='utf-8') as infile, \
        open(OUTPUT_DIR / file.name, 'w', encoding='utf-8') as outfile:

        for line in infile:
            line = unicodedata.normalize('NFC', line)
            outfile.write(line)

print("Done")
