"""并发压测 /search/text，按并发梯度测吞吐/延迟分位数/GPU 利用率。

方法论和结果解读见 docs/concurrency-benchmark.md。

用法:
    python scripts/bench_search_concurrency.py
    python scripts/bench_search_concurrency.py --url http://localhost:8000/search/text \\
        --levels 1,2,4,8,16,32,64 --requests-per-level 40 --query "检索服务的并发能力"
"""
import argparse
import asyncio
import json
import statistics
import subprocess
import threading
import time

import httpx


class GpuSampler:
    """每 0.3s 采样一次 nvidia-smi 的 GPU 利用率，跑在后台线程里，
    这样不会跟被测请求抢事件循环。采不到（没有 nvidia-smi，或非 NVIDIA
    环境）时静默跳过，不影响压测本身。"""

    def __init__(self):
        self._stop = threading.Event()
        self.samples: list[float] = []
        self._thread: threading.Thread | None = None

    def _run(self):
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=2,
                )
                val = out.stdout.strip().splitlines()[0]
                self.samples.append(float(val))
            except Exception:
                pass
            time.sleep(0.3)

    def start(self):
        self.samples = []
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)


class CpuSampler:
    """每 0.3s 采样一次系统 CPU 利用率（读 /proc/stat 前后差值），跑在后台线程里，
    这样不会跟被测请求抢事件循环。采不到（非 Linux 环境，没有 /proc/stat）时
    静默跳过，不影响压测本身。"""

    def __init__(self):
        self._stop = threading.Event()
        self.samples: list[float] = []
        self._thread: threading.Thread | None = None

    @staticmethod
    def _read_cpu_times() -> tuple[float, float] | None:
        try:
            with open("/proc/stat") as f:
                fields = [float(x) for x in f.readline().split()[1:]]
            idle = fields[3] + fields[4]  # idle + iowait
            return sum(fields), idle
        except Exception:
            return None

    def _run(self):
        prev = self._read_cpu_times()
        while not self._stop.is_set():
            time.sleep(0.3)
            cur = self._read_cpu_times()
            if prev is not None and cur is not None:
                total_delta = cur[0] - prev[0]
                idle_delta = cur[1] - prev[1]
                if total_delta > 0:
                    self.samples.append((1 - idle_delta / total_delta) * 100)
            prev = cur

    def start(self):
        self.samples = []
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)


async def _one_request(client: httpx.AsyncClient, sem: asyncio.Semaphore, url: str, payload: dict) -> tuple[float, int]:
    async with sem:
        start = time.monotonic()
        try:
            resp = await client.post(url, json=payload, timeout=60)
            return time.monotonic() - start, resp.status_code
        except Exception:
            return time.monotonic() - start, -1


async def run_level(url: str, payload: dict, concurrency: int, n: int) -> dict:
    sem = asyncio.Semaphore(concurrency)
    gpu = GpuSampler()
    cpu = CpuSampler()
    async with httpx.AsyncClient() as client:
        gpu.start()
        cpu.start()
        wall_start = time.monotonic()
        results = await asyncio.gather(*[_one_request(client, sem, url, payload) for _ in range(n)])
        wall_elapsed = time.monotonic() - wall_start
        gpu.stop()
        cpu.stop()

    latencies = sorted(r[0] for r in results)
    errors = sum(1 for r in results if r[1] != 200)

    def pct(p):
        idx = min(len(latencies) - 1, int(len(latencies) * p))
        return latencies[idx]

    return {
        "concurrency": concurrency,
        "n": n,
        "wall_s": round(wall_elapsed, 3),
        "throughput_rps": round(n / wall_elapsed, 2),
        "p50_ms": round(pct(0.50) * 1000, 1),
        "p95_ms": round(pct(0.95) * 1000, 1),
        "p99_ms": round(pct(0.99) * 1000, 1),
        "max_ms": round(latencies[-1] * 1000, 1),
        "errors": errors,
        "gpu_util_avg": round(statistics.mean(gpu.samples), 1) if gpu.samples else None,
        "gpu_util_max": round(max(gpu.samples), 1) if gpu.samples else None,
        "cpu_util_avg": round(statistics.mean(cpu.samples), 1) if cpu.samples else None,
        "cpu_util_max": round(max(cpu.samples), 1) if cpu.samples else None,
    }


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="/search/text 并发压测工具")
    parser.add_argument("--url", default="http://localhost:8000/search/text")
    parser.add_argument("--query", default="检索服务的并发能力")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--modalities", default="text", help="逗号分隔，如 text,image")
    parser.add_argument("--levels", default="1,2,4,6,8,16,32", help="并发梯度，逗号分隔")
    parser.add_argument("--requests-per-level", type=int, default=40)
    return parser.parse_args(argv)


async def main_async(args: argparse.Namespace) -> None:
    payload = {"query": args.query, "top_k": args.top_k, "modalities": args.modalities.split(",")}
    levels = [int(x) for x in args.levels.split(",")]

    results = []
    for level in levels:
        r = await run_level(args.url, payload, level, args.requests_per_level)
        results.append(r)
        print(json.dumps(r, ensure_ascii=False))

    print("---SUMMARY---")
    for r in results:
        print(
            f"c={r['concurrency']:>3} rps={r['throughput_rps']:>7} "
            f"p50={r['p50_ms']:>7}ms p95={r['p95_ms']:>7}ms p99={r['p99_ms']:>7}ms "
            f"max={r['max_ms']:>8}ms err={r['errors']} gpu_avg={r['gpu_util_avg']} gpu_max={r['gpu_util_max']} "
            f"cpu_avg={r['cpu_util_avg']} cpu_max={r['cpu_util_max']}"
        )


def main(argv: list[str] | None = None) -> None:
    asyncio.run(main_async(_parse_args(argv)))


if __name__ == "__main__":
    main()
