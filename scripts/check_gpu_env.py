"""验证 DGX Spark (GB10) 上 PyTorch CUDA + BGE-M3 + Chinese-CLIP 是否可用。"""
import sys


def check_cuda() -> None:
    import torch

    assert torch.cuda.is_available(), "CUDA 不可用，检查 PyTorch 是否装了 cu130 aarch64 wheel"
    print(f"[OK] CUDA available. device={torch.cuda.get_device_name(0)}")


def check_bge_m3() -> None:
    from FlagEmbedding import BGEM3FlagModel

    model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True, device="cuda")
    output = model.encode(["你好，世界"], return_dense=True, return_sparse=True, return_colbert_vecs=False)
    dense = output["dense_vecs"][0]
    sparse = output["lexical_weights"][0]
    assert len(dense) == 1024, f"期望 dense 向量维度 1024，实际 {len(dense)}"
    assert isinstance(sparse, dict) and len(sparse) > 0
    print(f"[OK] BGE-M3 dense dim={len(dense)}, sparse terms={len(sparse)}")


def check_chinese_clip() -> None:
    import torch
    from PIL import Image
    from transformers import ChineseCLIPModel, ChineseCLIPProcessor

    model_name = "OFA-Sys/chinese-clip-vit-base-patch16"
    model = ChineseCLIPModel.from_pretrained(model_name).to("cuda").eval()
    processor = ChineseCLIPProcessor.from_pretrained(model_name)

    image = Image.new("RGB", (224, 224), color="white")
    with torch.no_grad():
        image_inputs = processor(images=image, return_tensors="pt").to("cuda")
        image_features = model.get_image_features(**image_inputs)
        text_inputs = processor(text=["一张白色的图片"], return_tensors="pt", padding=True).to("cuda")
        text_features = model.get_text_features(**text_inputs)

    print(f"[OK] Chinese-CLIP image dim={image_features.shape[-1]}, text dim={text_features.shape[-1]}")


if __name__ == "__main__":
    try:
        check_cuda()
        check_bge_m3()
        check_chinese_clip()
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] {exc}", file=sys.stderr)
        sys.exit(1)
    print("[OK] 环境验证全部通过")
