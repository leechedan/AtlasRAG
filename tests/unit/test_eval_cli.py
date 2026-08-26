from scripts.eval_chunk_quality import main


def test_main_passes_parsed_args_to_run_and_writes_report(tmp_path, monkeypatch, capsys):
    captured_kwargs = {}

    def fake_run(**kwargs):
        captured_kwargs.update(kwargs)
        return {"aggregate": {"chunk_count": 1}}

    written = {}

    def fake_write_report(report, output_path):
        written["report"] = report
        written["output_path"] = output_path

    monkeypatch.setattr("scripts.eval_chunk_quality.run", fake_run)
    monkeypatch.setattr("scripts.eval_chunk_quality.write_report", fake_write_report)

    output_path = tmp_path / "report.json"
    exit_code = main(
        [
            "--input-dir",
            str(tmp_path),
            "--chunk-size",
            "500",
            "--chunk-overlap",
            "50",
            "--no-semantic",
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 0
    assert captured_kwargs["input_dir"] == str(tmp_path)
    assert captured_kwargs["chunk_size"] == 500
    assert captured_kwargs["chunk_overlap"] == 50
    assert captured_kwargs["run_semantic"] is False
    assert captured_kwargs["embedder"] is None
    assert captured_kwargs["qa_file"] is None
    assert captured_kwargs["retrieve_fn"] is None
    # --no-semantic 时 min/max 未显式指定，按 chunk_size 派生默认区间
    assert captured_kwargs["min_chars"] == 100  # 0.2 * 500
    assert captured_kwargs["max_chars"] == 500
    assert written["output_path"] == str(output_path)
    assert written["report"] == {"aggregate": {"chunk_count": 1}}

    out = capsys.readouterr().out
    assert "chunk_count" in out
    assert str(output_path) in out


def test_main_uses_explicit_min_max_chars_when_given(monkeypatch, tmp_path):
    captured_kwargs = {}
    monkeypatch.setattr(
        "scripts.eval_chunk_quality.run",
        lambda **kwargs: captured_kwargs.update(kwargs) or {"aggregate": {}},
    )
    monkeypatch.setattr("scripts.eval_chunk_quality.write_report", lambda *a, **k: None)

    main(
        [
            "--input-dir",
            str(tmp_path),
            "--min-chars",
            "10",
            "--max-chars",
            "20",
            "--no-semantic",
        ]
    )

    assert captured_kwargs["min_chars"] == 10
    assert captured_kwargs["max_chars"] == 20


def test_main_requires_retrieve_fn_wiring_when_qa_file_given(monkeypatch, tmp_path):
    """--qa-file 传了就必须给 run() 传一个可调用的 retrieve_fn（不是 None），
    具体是否真的打了 Qdrant 由 fake run 的调用方断言，这里只验证不是 None。"""
    captured_kwargs = {}
    monkeypatch.setattr(
        "scripts.eval_chunk_quality.run",
        lambda **kwargs: captured_kwargs.update(kwargs) or {"aggregate": {}},
    )
    monkeypatch.setattr("scripts.eval_chunk_quality.write_report", lambda *a, **k: None)
    monkeypatch.setattr(
        "scripts.eval_chunk_quality._build_retrieve_fn", lambda: (lambda q, k: [])
    )

    qa_path = tmp_path / "qa.json"
    qa_path.write_text("[]", encoding="utf-8")

    main(["--input-dir", str(tmp_path), "--no-semantic", "--qa-file", str(qa_path)])

    assert captured_kwargs["qa_file"] == str(qa_path)
    assert captured_kwargs["retrieve_fn"] is not None


def test_main_passes_chunker_arg_to_run(monkeypatch, tmp_path):
    captured_kwargs = {}
    monkeypatch.setattr(
        "scripts.eval_chunk_quality.run",
        lambda **kwargs: captured_kwargs.update(kwargs) or {"aggregate": {}},
    )
    monkeypatch.setattr("scripts.eval_chunk_quality.write_report", lambda *a, **k: None)

    main(
        [
            "--input-dir",
            str(tmp_path),
            "--no-semantic",
            "--chunker",
            "structured",
        ]
    )

    assert captured_kwargs["chunker"] == "structured"


def test_main_defaults_chunker_to_char(monkeypatch, tmp_path):
    captured_kwargs = {}
    monkeypatch.setattr(
        "scripts.eval_chunk_quality.run",
        lambda **kwargs: captured_kwargs.update(kwargs) or {"aggregate": {}},
    )
    monkeypatch.setattr("scripts.eval_chunk_quality.write_report", lambda *a, **k: None)

    main(["--input-dir", str(tmp_path), "--no-semantic"])

    assert captured_kwargs["chunker"] == "char"
