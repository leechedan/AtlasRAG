"""Whole-document -> single Markdown string conversion, no chunking.

`convert_file_to_markdown` backs `POST /documents/convert-to-markdown`
(app.api.routes.documents). Unlike app.ingest.pipeline.ingest_file, this
does no hashing, no Qdrant writes, mints no doc_id, and produces no
chunks -- it exists purely to hand the caller back the whole document as
one undivided Markdown string, for callers that want the converted
document itself rather than an indexed-for-retrieval copy of it.

Reuses the exact same per-extension parsers as app.ingest.pipeline (and
its SUPPORTED_*_EXTS constants, so the two never drift apart on which
extensions are accepted) -- pdf/docx/pptx/xlsx go through MinerU via
parse_pdf_to_markdown (sub/sup tag stripping already applied there), csv
gets reshaped into a real Markdown table (chunk_csv_rows' "key=val" line
format isn't Markdown table syntax and is unsuitable here), and images
use run_ocr directly rather than parse_image -- parse_image also does CLIP
embedding and persists a thumbnail to disk, both of which are ingest-time
side effects this no-persistence-of-the-*input* conversion endpoint must
not trigger.

Embedded images are referenced from the returned Markdown by relative
path (`images/<filename>`), the same scheme MinerU's own md_content
already uses -- not inlined as base64 and not rewritten to a
`/documents/{doc_id}/image` URL (there is no doc_id here). The route
persists ConvertedDocument.images to disk under a convert_id and exposes
them via a batch-download zip endpoint (see
app.api.routes.documents.convert_document_to_markdown and
download_converted_images), so a plain relative-path Markdown file plus
that zip extracted alongside it opens correctly in any ordinary
filesystem-based Markdown reader (VS Code, Typora, Obsidian, ...) without
network access back to this service.
"""

from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from app.ingest.parsers.csv_parser import parse_csv_file
from app.ingest.parsers.image_parser import run_ocr
from app.ingest.parsers.pdf_parser import parse_pdf_to_markdown
from app.ingest.parsers.text_parser import parse_text_file
from app.ingest.pipeline import SUPPORTED_IMAGE_EXTS, SUPPORTED_OFFICE_EXTS, SUPPORTED_TEXT_EXTS

SUPPORTED_CONVERT_EXTS = SUPPORTED_TEXT_EXTS | {".csv"} | SUPPORTED_IMAGE_EXTS | SUPPORTED_OFFICE_EXTS | {".pdf"}

# Relative path prefix used both in the Markdown image references this
# module emits and in the zip entries the route builds from
# ConvertedDocument.images -- the two must stay in lock-step, or the
# links in the returned .md won't resolve once the images zip is
# extracted next to it.
IMAGES_SUBDIR = "images"


@dataclass
class ConvertedImage:
    filename: str
    image: Image.Image


@dataclass
class ConvertedDocument:
    markdown_text: str
    images: list[ConvertedImage] = field(default_factory=list)


def _rows_to_markdown_table(rows: list[dict], header: list[str]) -> str:
    if not header:
        return ""
    # Escape "|" (would otherwise break out of the cell) and newlines
    # (Markdown table rows are single-line) in cell content.
    def _cell(value: str) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    lines.extend("| " + " | ".join(_cell(row.get(col, "")) for col in header) + " |" for row in rows)
    return "\n".join(lines)


def convert_file_to_markdown(path: Path) -> ConvertedDocument:
    """Parse `path` with the same parser the ingest pipeline would use for
    its extension, and return the whole document as one Markdown string
    plus any images it embeds (referenced from the Markdown by
    `images/<filename>`, not yet written to disk -- the caller decides
    where/whether to persist them).

    Raises ValueError for an unsupported extension, and lets any parser
    error (e.g. parse_pdf_to_markdown's RuntimeError on a MinerU failure)
    propagate to the caller -- this is a single synchronous conversion,
    not a batch job, so a failure should surface immediately rather than
    being swallowed the way app.ingest.pipeline.ingest_file swallows
    per-file errors for a batch.
    """
    ext = path.suffix.lower()

    if ext in SUPPORTED_TEXT_EXTS:
        return ConvertedDocument(markdown_text=parse_text_file(path))

    if ext == ".csv":
        rows, header = parse_csv_file(path)
        return ConvertedDocument(markdown_text=_rows_to_markdown_table(rows, header))

    if ext in SUPPORTED_IMAGE_EXTS:
        image = Image.open(path).convert("RGB")
        ocr_text = run_ocr(image)
        parts = [f"![{path.name}]({IMAGES_SUBDIR}/{path.name})"]
        if ocr_text:
            parts.append(ocr_text)
        return ConvertedDocument(
            markdown_text="\n\n".join(parts), images=[ConvertedImage(filename=path.name, image=image)]
        )

    if ext == ".pdf" or ext in SUPPORTED_OFFICE_EXTS:
        parsed = parse_pdf_to_markdown(path)
        # parsed.markdown_text already references each image as
        # "images/<filename>" (that's MinerU's own convention) -- so
        # nothing needs rewriting here, the filenames just need to match
        # what gets zipped up.
        images = [ConvertedImage(filename=img.filename, image=img.image) for img in parsed.images]
        return ConvertedDocument(markdown_text=parsed.markdown_text, images=images)

    raise ValueError(f"unsupported file extension: {ext}")
