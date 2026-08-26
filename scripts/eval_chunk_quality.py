"""切片质量量化评估 CLI。

用法示例：
    python -m scripts.eval_chunk_quality --input-dir data/uploads --output report.json
    python -m scripts.eval_chunk_quality --input-dir data/uploads --chunk-size 1500 --output report_1500.json

多次跑不同 --chunk-size/--chunk-overlap，对比生成的 report.json，就是最小可行的
参数消融对比。
"""

import argparse
import sys

from app.config import settings
from app.eval.report import format_summary_table, write_report
from app.eval.runner import run


def _build_retrieve_fn():
    from app.models.embeddings import get_text_embedder
    from app.store.qdrant_client import compute_chunk_id, get_default_client, search_text_hybrid

    client = get_default_client()
    embedder = get_text_embedder()

    def retrieve_fn(query: str, top_k: int) -> list[str]:
        vector = embedder.embed([query])[0]
        hits = search_text_hybrid(client, vector, top_k=top_k)
        return [
            compute_chunk_id(h.payload["doc_id"], h.payload.get("chunk_index", 0)) for h in hits
        ]

    return retrieve_fn


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="切片质量量化评估工具")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--chunk-size", type=int, default=settings.chunk_size)
    parser.add_argument("--chunk-overlap", type=int, default=settings.chunk_overlap)
    parser.add_argument("--min-chars", type=int, default=None)
    parser.add_argument("--max-chars", type=int, default=None)
    parser.add_argument("--no-semantic", action="store_true")
    parser.add_argument("--chunker", choices=["char", "structured"], default="char")
    parser.add_argument("--qa-file", default=None)
    parser.add_argument("--output", default="report.json")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    min_chars = args.min_chars if args.min_chars is not None else int(args.chunk_size * 0.2)
    max_chars = args.max_chars if args.max_chars is not None else args.chunk_size

    embedder = None
    if not args.no_semantic:
        from app.models.embeddings import get_text_embedder

        embedder = get_text_embedder()

    retrieve_fn = _build_retrieve_fn() if args.qa_file else None

    report = run(
        input_dir=args.input_dir,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        min_chars=min_chars,
        max_chars=max_chars,
        run_semantic=not args.no_semantic,
        embedder=embedder,
        qa_file=args.qa_file,
        retrieve_fn=retrieve_fn,
        chunker=args.chunker,
    )

    write_report(report, args.output)
    print(format_summary_table(report["aggregate"]))
    print(f"\n完整报告已写入 {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
