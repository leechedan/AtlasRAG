#!/usr/bin/env bash
# 删除 data/ 下除 dataset、hf_cache 之外的所有子文件夹（不动 data/ 下的普通文件）。
set -euo pipefail

DATA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/data"
KEEP=("dataset" "hf_cache")

if [[ ! -d "$DATA_DIR" ]]; then
    echo "找不到目录：$DATA_DIR" >&2
    exit 1
fi

targets=()
for entry in "$DATA_DIR"/*/; do
    [[ -d "$entry" ]] || continue
    name="$(basename "$entry")"
    skip=false
    for keep in "${KEEP[@]}"; do
        if [[ "$name" == "$keep" ]]; then
            skip=true
            break
        fi
    done
    [[ "$skip" == true ]] || targets+=("$name")
done

if [[ ${#targets[@]} -eq 0 ]]; then
    echo "没有需要删除的文件夹。"
    exit 0
fi

echo "将删除以下 $DATA_DIR 下的文件夹："
for t in "${targets[@]}"; do
    echo "  - $t"
done

read -r -p "确认删除？(y/N) " reply
if [[ ! "$reply" =~ ^[Yy]$ ]]; then
    echo "已取消。"
    exit 0
fi

for t in "${targets[@]}"; do
    rm -rf -- "${DATA_DIR:?}/$t"
    echo "已删除：$t"
done
