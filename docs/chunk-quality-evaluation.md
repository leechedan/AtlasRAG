# Chunk Quality and Evaluation

This document defines how AtlasRAG measures and improves text-chunk quality. It applies to both the character-based chunker and the structured Markdown chunker used during ingestion.

## Quality goals

A useful chunk preserves enough local context to answer a query, is small enough for embedding and reranking, retains document structure, and avoids unnecessary overlap duplication.

For Markdown, headings provide section context; tables and fenced code blocks should remain intact whenever possible; overlap should preserve context across artificial boundaries without duplicating protected table or code-block content.

## Implemented measurements

Run `scripts/eval_chunk_quality.py` to recursively evaluate `.md` and `.txt` files and write a JSON report.

```bash
# Static structure and length checks; no embedding model is loaded.
python3 -m scripts.eval_chunk_quality \
  --input-dir data/uploads \
  --chunker structured \
  --no-semantic \
  --output /tmp/chunk-quality.json

# Add sentence-level semantic coherence; requires the configured embedding model.
python3 -m scripts.eval_chunk_quality \
  --input-dir data/uploads \
  --chunker structured \
  --output /tmp/chunk-quality-semantic.json
```

| Metric | Definition | Interpretation |
| --- | --- | --- |
| `length_min_chars`, `length_max_chars`, `length_mean_chars`, `length_median_chars`, `length_stdev_chars` | Distribution of chunk character lengths | Diagnose chunks that are too short, too large, or highly variable. |
| `length_out_of_range_ratio` | Share outside `min_chars..max_chars` | Lower is normally better; inspect intentionally oversize protected blocks separately. |
| `truncated_region_ratio` | Markdown tables and code blocks not fully contained by any chunk | Zero is the target for structured Markdown chunking. |
| `mean_coherence` | Mean cosine similarity of adjacent sentences within scoreable chunks | Diagnostic only: high similarity can also mean redundancy. |
| `recall_at_k`, `mrr` | Retrieval recall and rank quality over labeled questions | Primary evidence for selecting a configuration. |

By default, `min_chars` is 20% of `chunk_size` and `max_chars` equals `chunk_size`. Service defaults are `CHUNK_SIZE=1000` and `CHUNK_OVERLAP=150`; experiments can override these values.

## Retrieval evaluation

Pass `--qa-file` to calculate `Recall@5`, `Recall@10`, and MRR. The file is a JSON array:

```json
[
  {
    "query": "How is a renewal notice handled?",
    "expected_chunk_ids": ["<chunk-id>"]
  }
]
```

`expected_chunk_ids` contains Qdrant chunk IDs computed from `doc_id` and `chunk_index`.

```bash
python3 -m scripts.eval_chunk_quality \
  --input-dir data/uploads \
  --chunker structured \
  --qa-file qa-gold.json \
  --output /tmp/chunk-quality-retrieval.json
```

**Important:** The current CLI queries the already indexed Qdrant collection. Changing `--chunk-size`, `--chunk-overlap`, or `--chunker` alone does **not** rebuild that index. A valid retrieval A/B test must re-ingest the same corpus into a separate index for each candidate configuration and query that matching index. IDs also change after re-chunking, so durable gold data should record a stable source location or answer passage in addition to the current chunk ID.

## Recommended protocol

1. Define a representative, versioned corpus and 50–100 or more real user questions. Record expected source passages, not only current chunk IDs.
2. Select candidate configurations: for example, character versus structured chunking and several size/overlap pairs.
3. Re-ingest the same corpus into isolated indexes, one index per candidate.
4. Apply static gates: full source coverage, no table/code truncation, acceptable short-chunk rate, and bounded overlap duplication.
5. Measure `Recall@5`, `Recall@10`, and MRR on the same gold questions. Compare latency, index size, and token cost as operational constraints.
6. Manually inspect failed questions and boundary cases before promoting a configuration.

## Current limits and follow-up metrics

- **Heading context:** The evaluator currently asks whether heading text appears in a chunk body. Structured chunks instead retain `heading_path` metadata, so `missing_heading_ratio` can be a false positive for structured chunking. Evaluate `heading_path` directly before using this ratio as a release gate.
- **Span location:** Structure checks locate chunks through sequential text search. Repeated text or injected overlap can make spans ambiguous; exact source offsets from the chunker would be more reliable.
- **Characters and tokens:** Length is measured in characters. Production limits should also use the actual embedding/reranker tokenizer.
- **Semantic coherence:** Sentence splitting currently targets Chinese terminal punctuation and coherence is not a universal measure of retrieval usefulness.

Recommended additions are source-coverage ratio, overlap/duplicate ratio, sentence-or-paragraph boundary break rate, heading-metadata completeness, and token-length percentiles. These are diagnostic or safety metrics; retrieval quality on a stable gold set remains the decision metric.

## Relevant implementation

- CLI: `scripts/eval_chunk_quality.py`
- Evaluation orchestration: `app/eval/runner.py`
- Structure, length, coherence, and retrieval metrics: `app/eval/`
- Structured Markdown chunker: `app/ingest/markdown_chunker.py`
