import ijson


def open_new_chunk(chunk_idx, out_dir):
    outfile = open(f'{out_dir}/chunk_{chunk_idx:04d}.txt', 'w', encoding='utf-8')
    return outfile


def flush_buffer(buffer, outfile):
    if not buffer:
        return
    outfile.write('\n'.join(buffer))
    outfile.write('\n')
    buffer.clear()


def stream_extract_chunks(filepath, out_dir, max_docs=1000):
    '''
    Event-based extraction (ijson.parse) instead of ijson.items.

    ijson.items(inline, 'item') fully materializes each top-level array
    element into a Python dict before we can inspect it. If a document
    has multiple page_* keys, ALL of them get built in memory even though
    we only ever need the single text_data field. For docs with 5M+ char
    text, that's a lot of wasted memory per document.

    ijson.parse() instead streams (prefix, event, value) tokens directly
    off the wire with no intermediate container. Since each document has
    exactly one text_data field, we just grab it wherever it shows up.
    '''
    doc_count = 0
    chunk_idx = 0
    buffer = []
    outfile = None

    BUFFER_SIZE = 10          # flush after this many docs OR immediately if a doc is huge
    LARGE_TEXT_CHARS = 200_000  # flush immediately if a single text is this big
    TARGET_SIZE = 100 * 1024 * 1024  # 100 mb per chunk file

    with open(filepath, 'rb') as inline:
        parser = ijson.parse(inline)

        current_doc_text = None   # this doc's text_data, once we hit it

        for prefix, event, value in parser:

            if prefix == 'item' and event == 'start_map':
                # new document starting
                current_doc_text = None

            elif event == 'string' and prefix.endswith('.text_data'):
                # only the value we actually want ever touches memory here
                current_doc_text = value

            elif prefix == 'item' and event == 'end_map':
                # document finished — same behavior as the original per-doc loop body
                if outfile is None:
                    outfile = open_new_chunk(chunk_idx=chunk_idx, out_dir=out_dir)

                if current_doc_text:
                    buffer.append(current_doc_text.strip())
                    if len(buffer) >= BUFFER_SIZE or len(current_doc_text) >= LARGE_TEXT_CHARS:
                        flush_buffer(buffer, outfile)

                if outfile.tell() >= TARGET_SIZE:
                    flush_buffer(buffer, outfile)
                    outfile.close()
                    chunk_idx += 1
                    outfile = open_new_chunk(chunk_idx=chunk_idx, out_dir=out_dir)

                doc_count += 1
                if doc_count >= max_docs:
                    break

                if doc_count % 1000 == 0:
                    print(f"{doc_count} documents chunked")

    if buffer:
        flush_buffer(buffer, outfile)
    if outfile:
        outfile.close()


stream_extract_chunks(
    filepath=r'Nepali model training\nepal.pustakalaya_books.json',
    out_dir=r'Nepali model training\chunks',
)