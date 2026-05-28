from benchpilot.benchmarks.gpu.ollama_bench import (
    OllamaInference,
    _bench_one,
    _pick_small,
)


def test_pick_small_prefers_2b_and_4b_over_larger():
    avail = ["qwen3.5:9b", "qwen3.5:2b", "qwen3.5:4b", "llama:70b"]
    chosen = _pick_small(avail, 2)
    assert chosen == ["qwen3.5:2b", "qwen3.5:4b"]


def test_pick_small_drops_cloud_tags():
    avail = ["deepseek-v4-flash:cloud", "qwen3.5:2b"]
    assert _pick_small(avail, 2) == ["qwen3.5:2b"]


def test_pick_small_returns_up_to_n():
    avail = ["qwen3.5:2b"]
    assert _pick_small(avail, 2) == ["qwen3.5:2b"]


def test_pick_small_unknown_sizes_sorted_after_known(monkeypatch):
    avail = ["mystery-model:latest", "qwen3.5:2b"]
    out = _pick_small(avail, 2)
    assert out[0] == "qwen3.5:2b"
    assert "mystery-model:latest" in out


def test_bench_one_computes_tokens_per_second(monkeypatch):
    """Mock httpx.post and verify the parser does the ns→s conversion."""
    import httpx

    class FakeResp:
        def __init__(self, payload):
            self._p = payload
        def raise_for_status(self):
            pass
        def json(self):
            return self._p

    def fake_post(url, json=None, timeout=None):
        assert json["model"] == "qwen3.5:2b"
        assert json["stream"] is False
        assert json["keep_alive"] == 0
        return FakeResp({
            "eval_count": 100,
            "eval_duration": 2_000_000_000,         # 2s -> 50 tok/s
            "prompt_eval_count": 40,
            "prompt_eval_duration": 200_000_000,    # 0.2s -> 200 tok/s
        })

    monkeypatch.setattr(httpx, "post", fake_post)
    r = _bench_one("http://h", "qwen3.5:2b", "p", 100)
    assert r["tokens_generated"] == 100
    assert r["gen_tokens_per_second"] == 50.0
    assert r["prompt_tokens"] == 40
    assert r["prompt_tokens_per_second"] == 200.0


def test_bench_one_handles_missing_timings(monkeypatch):
    import httpx

    class FakeResp:
        def raise_for_status(self): pass
        def json(self): return {"eval_count": None, "eval_duration": None}

    monkeypatch.setattr(httpx, "post", lambda *a, **k: FakeResp())
    r = _bench_one("http://h", "m", "p", 10)
    assert r["gen_tokens_per_second"] is None


def test_ollama_skipped_when_server_unreachable(monkeypatch):
    import httpx
    def boom(*a, **k):
        raise httpx.ConnectError("no server")
    monkeypatch.setattr(httpx, "get", boom)
    r = OllamaInference().run().results
    assert "skipped" in r


def test_ollama_skipped_when_no_models(monkeypatch):
    import httpx

    class R:
        def raise_for_status(self): pass
        def json(self): return {"models": []}
    monkeypatch.setattr(httpx, "get", lambda *a, **k: R())
    r = OllamaInference().run().results
    assert "skipped" in r
