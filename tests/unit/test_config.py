from app.config import settings


def test_reranker_settings_have_expected_defaults():
    assert settings.reranker_enabled is True
    assert settings.reranker_model_name == "BAAI/bge-reranker-v2-m3"
    assert settings.reranker_candidate_k == 10
    assert settings.reranker_dtype == "bf16"


def test_mineru_settings_have_expected_defaults():
    assert settings.mineru_backend == "pipeline"
    assert settings.mineru_timeout_seconds == 600


def test_extracted_images_dir_is_under_data_dir_and_created():
    assert settings.extracted_images_dir == settings.data_dir / "extracted_images"
    assert settings.extracted_images_dir.is_dir()
