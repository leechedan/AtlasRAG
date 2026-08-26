"""编排：读取语料库文档 -> 用现有 chunk_text() 切片 -> 跑各项质量指标 ->
汇总成一份 JSON 兼容的报告 dict。不改动 app.ingest.chunking 里的切片实现
本身，只是拿它的输出做测量。"""

from dataclasses import asdict
from pathlib import Path

from app.eval.coherence_metrics import Embedder, compute_coherence
from app.eval.length_metrics import compute_length_metrics
from app.eval.retrieval_metrics import evaluate_retrieval, load_qa_file
from app.eval.structure_metrics import detect_structure_violations
from app.ingest.chunking import chunk_text
from app.ingest.markdown_chunker import chunk_markdown_structured

SUPPORTED_EXTS = {".md", ".txt"}


def _iter_documents(input_dir: str):
    for path in sorted(Path(input_dir).rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTS:
            yield path


def run(
    input_dir: str,
    chunk_size: int,
    chunk_overlap: int,
    min_chars: int,
    max_chars: int,
    run_semantic: bool = True,
    embedder: Embedder | None = None,
    qa_file: str | None = None,
    retrieve_fn=None,
    chunker: str = "char",
) -> dict:
    if chunker not in ("char", "structured"):
        raise ValueError(f"未知 chunker: {chunker!r}，可选 'char' 或 'structured'")
    if run_semantic and embedder is None:
        raise ValueError("run_semantic=True 时必须提供 embedder")
    if qa_file and retrieve_fn is None:
        raise ValueError("qa_file 不为空时必须提供 retrieve_fn")

    document_reports = []
    all_chunks: list[str] = []
    total_truncated = 0
    total_regions = 0
    total_missing_heading = 0
    total_heading_expected = 0
    coherence_weighted_sum = 0.0
    coherence_weight = 0

    for path in _iter_documents(input_dir):
        text = path.read_text(encoding="utf-8")
        if chunker == "structured":
            chunks = [
                c.content
                for c in chunk_markdown_structured(
                    text, chunk_size=chunk_size, chunk_overlap=chunk_overlap
                )
            ]
        else:
            chunks = chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        all_chunks.extend(chunks)

        structure_report = detect_structure_violations(text, chunks, chunk_overlap)
        total_truncated += structure_report.truncated_regions
        total_regions += structure_report.total_protected_regions
        total_missing_heading += structure_report.chunks_missing_heading
        total_heading_expected += structure_report.chunks_with_heading_expected

        doc_entry = {
            "path": str(path),
            "chunk_count": len(chunks),
            "structure": asdict(structure_report),
        }

        if run_semantic and chunks:
            coherence_report = compute_coherence(chunks, embedder)
            doc_entry["coherence"] = asdict(coherence_report)
            coherence_weighted_sum += (
                coherence_report.mean_coherence * coherence_report.scored_chunk_count
            )
            coherence_weight += coherence_report.scored_chunk_count

        document_reports.append(doc_entry)

    length_report = compute_length_metrics(all_chunks, min_chars, max_chars)

    aggregate = {
        "document_count": len(document_reports),
        "chunk_count": len(all_chunks),
        "truncated_region_ratio": (total_truncated / total_regions) if total_regions else 0.0,
        "missing_heading_ratio": (
            total_missing_heading / total_heading_expected if total_heading_expected else 0.0
        ),
        **{f"length_{k}": v for k, v in asdict(length_report).items()},
    }

    if run_semantic:
        aggregate["mean_coherence"] = (
            coherence_weighted_sum / coherence_weight if coherence_weight else 0.0
        )
    else:
        aggregate["mean_coherence"] = None

    if qa_file:
        queries = load_qa_file(qa_file)
        retrieval_report = evaluate_retrieval(queries, retrieve_fn)
        aggregate["retrieval"] = asdict(retrieval_report)
    else:
        aggregate["retrieval"] = None

    return {"documents": document_reports, "aggregate": aggregate}
