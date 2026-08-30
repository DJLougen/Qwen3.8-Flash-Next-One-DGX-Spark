"""Evidence harness for the sglang recipe. Phases run independently.

usage: verify.py <concurrency|depth|needle|gsm8k|oom> [...]
Results append to results.json in this directory.
"""
import json, os, re, statistics, subprocess, sys, threading, time, urllib.error, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
URL = "http://127.0.0.1:30000/v1/chat/completions"
OUT = os.path.join(HERE, "results.json")

PAT = ("The quick brown fox jumps over the lazy dog. "
       "Databases use B-tree indexes for ordered range scans. "
       "Write-ahead logging guarantees durability across crashes. ")


def salt():
    """Unique prefix so the radix cache cannot serve a prior prefill."""
    return f"Run marker {time.time_ns()}-{os.getpid()}. "


def call(prompt, max_tokens, temperature=0.0, timeout=3600):
    body = {"model": "/model", "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": temperature, "stream": True,
            "stream_options": {"include_usage": True}}
    req = urllib.request.Request(URL, json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.perf_counter(); ttft = None; usage = None
    content = []; reasoning = []
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            for raw in r:
                line = raw.decode().strip()
                if not line.startswith("data: "):
                    continue
                d = line[6:]
                if d == "[DONE]":
                    break
                o = json.loads(d)
                if o.get("usage"):
                    usage = o["usage"]
                ch = o.get("choices") or []
                if not ch:
                    continue
                dl = ch[0].get("delta") or {}
                c, rc = dl.get("content"), dl.get("reasoning_content")
                if (c or rc) and ttft is None:
                    ttft = time.perf_counter() - t0
                if c:
                    content.append(c)
                if rc:
                    reasoning.append(rc)
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError) as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}", "total": time.perf_counter() - t0}
    total = time.perf_counter() - t0
    ct = (usage or {}).get("completion_tokens", 0)
    return {"ok": True, "ttft": ttft, "total": total, "ct": ct,
            "pt": (usage or {}).get("prompt_tokens", 0),
            "decode": (ct - 1) / (total - ttft) if ttft and total > ttft and ct > 1 else None,
            "content": "".join(content), "reasoning": "".join(reasoning)}


def server_stat(since, key):
    out = subprocess.run(["journalctl", "-u", "sglang", "--since", since, "--no-pager"],
                         capture_output=True, text=True).stdout
    vals = []
    for line in out.splitlines():
        if key in line:
            try:
                vals.append(float(line.split(key)[1].split(",")[0].strip()))
            except (ValueError, IndexError):
                pass
    return vals


def gpu_free():
    out = subprocess.run(["journalctl", "-u", "sglang", "-n", "400", "--no-pager"],
                         capture_output=True, text=True).stdout
    m = re.findall(r"available_gpu_mem=([0-9.]+) GB", out)
    return float(m[-1]) if m else None


def save(phase, data):
    all_r = {}
    if os.path.exists(OUT):
        all_r = json.load(open(OUT))
    all_r[phase] = data
    json.dump(all_r, open(OUT, "w"), indent=2)
    print(f"\n[saved phase '{phase}' -> {OUT}]")


def pct(v, p):
    if not v:
        return None
    v = sorted(v)
    return v[min(len(v) - 1, int(round(p / 100 * (len(v) - 1))))]


# ---------------------------------------------------------------- concurrency
def phase_concurrency(levels=(1, 2, 4, 8), per_stream=192, prompt_reps=40):
    print("=" * 74)
    print("CONCURRENCY SWEEP -- aggregate vs per-stream, TTFT p50/p95, success rate")
    print("=" * 74)
    call(salt() + "Warm up.", 16)
    rows = []
    for n in levels:
        results = [None] * n
        def worker(i):
            p = salt() + PAT * prompt_reps + "\n\nExplain B-tree indexes in detail."
            results[i] = call(p, per_stream)
        threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
        since = time.strftime("%Y-%m-%d %H:%M:%S"); time.sleep(1)
        t0 = time.perf_counter()
        for t in threads: t.start()
        for t in threads: t.join()
        wall = time.perf_counter() - t0
        ok = [r for r in results if r and r["ok"]]
        toks = sum(r["ct"] for r in ok)
        ttfts = [r["ttft"] for r in ok if r["ttft"]]
        per = [r["decode"] for r in ok if r["decode"]]
        acc = server_stat(since, "accept len:")
        row = {"concurrency": n, "requests": n, "succeeded": len(ok),
               "success_rate": len(ok) / n, "wall_s": round(wall, 2),
               "aggregate_tok_s": round(toks / wall, 1) if wall else None,
               "per_stream_tok_s": round(statistics.median(per), 1) if per else None,
               "ttft_p50_s": round(pct(ttfts, 50), 2) if ttfts else None,
               "ttft_p95_s": round(pct(ttfts, 95), 2) if ttfts else None,
               "accept_len_median": round(statistics.median(acc), 2) if acc else None,
               "gpu_free_gb": gpu_free()}
        rows.append(row)
        print(f"n={n}: agg {row['aggregate_tok_s']:>6} tok/s | per-stream "
              f"{row['per_stream_tok_s']:>5} | TTFT p50 {row['ttft_p50_s']:>6}s "
              f"p95 {row['ttft_p95_s']:>6}s | ok {len(ok)}/{n} | accept "
              f"{row['accept_len_median']} | free {row['gpu_free_gb']} GB")
    save("concurrency", rows)


# ---------------------------------------------------------------------- depth
def phase_depth(targets=(8000, 32000, 64000, 128000, 200000)):
    print("=" * 74)
    print("TTFT vs DEPTH -- cold prefill, unique prefix per run (no cache reuse)")
    print("=" * 74)
    tok_per_rep = 30.0   # calibrated below from the first measurement
    rows = []
    for target in targets:
        reps = max(1, int(target / tok_per_rep))
        p = salt() + PAT * reps + "\n\nSummarize the repeated themes in one sentence."
        since = time.strftime("%Y-%m-%d %H:%M:%S"); time.sleep(1)
        r = call(p, 96)
        if not r["ok"]:
            print(f"target {target}: FAILED {r['error']}")
            rows.append({"target": target, "ok": False, "error": r["error"]})
            continue
        tok_per_rep = r["pt"] / reps      # self-calibrate for the next target
        acc = server_stat(since, "accept len:")
        row = {"target": target, "ok": True, "prompt_tokens": r["pt"],
               "ttft_s": round(r["ttft"], 2),
               "prefill_tok_s": round(r["pt"] / r["ttft"]) if r["ttft"] else None,
               "decode_tok_s": round(r["decode"], 1) if r["decode"] else None,
               "accept_len_median": round(statistics.median(acc), 2) if acc else None,
               "gpu_free_gb": gpu_free()}
        rows.append(row)
        print(f"{r['pt']:>7} tok | TTFT {row['ttft_s']:>7}s ({row['prefill_tok_s']:>5} tok/s) "
              f"| decode {row['decode_tok_s']:>5} | accept {row['accept_len_median']} "
              f"| free {row['gpu_free_gb']} GB")
    save("depth", rows)


# --------------------------------------------------------------------- needle
def phase_needle(depths=(0.001, 0.40, 0.90), haystack_tokens=110000):
    print("=" * 74)
    print("NEEDLE RETRIEVAL")
    print("=" * 74)
    secret = "The maintenance passcode for the Halifax relay is 7734-ZULU-19."
    question = "\n\nWhat is the maintenance passcode for the Halifax relay? Answer with the code only."
    reps = int(haystack_tokens / 30)
    rows = []
    for d in depths:
        body = PAT * reps
        cut = int(len(body) * d)
        p = salt() + body[:cut] + " " + secret + " " + body[cut:] + question
        r = call(p, 128)
        if not r["ok"]:
            rows.append({"depth": d, "ok": False, "error": r["error"]}); print(f"depth {d}: FAILED"); continue
        found = "7734" in r["content"] and "ZULU" in r["content"].upper()
        rows.append({"depth": d, "ok": True, "prompt_tokens": r["pt"], "found": found,
                     "answer": r["content"].strip()[:80]})
        print(f"depth {d*100:5.1f}% | {r['pt']:>7} tok | found={found} | {r['content'].strip()[:60]!r}")
    hits = sum(1 for x in rows if x.get("found"))
    print(f"\nneedle: {hits}/{len(rows)}")
    save("needle", rows)


# ---------------------------------------------------------------------- gsm8k
def phase_gsm8k(n=40, max_tokens=2048):
    print("=" * 74)
    print(f"GSM8K -- first {n} of the official test split")
    print("=" * 74)
    path = os.path.join(HERE, "gsm8k_test.jsonl")
    rows = [json.loads(l) for l in open(path)][:n]
    correct = 0; details = []
    for i, item in enumerate(rows):
        gold = item["answer"].split("####")[-1].strip().replace(",", "")
        r = call(item["question"] + "\n\nGive the final numeric answer.", max_tokens)
        if not r["ok"]:
            details.append({"i": i, "ok": False, "error": r["error"]}); continue
        nums = re.findall(r"-?\d[\d,]*\.?\d*", r["content"].replace(",", ""))
        pred = nums[-1].rstrip(".") if nums else None
        hit = pred is not None and abs(float(pred) - float(gold)) < 1e-6
        correct += hit
        details.append({"i": i, "ok": True, "gold": gold, "pred": pred, "hit": hit})
        print(f"  {i+1:>3}/{len(rows)} gold={gold:>8} pred={str(pred):>8} {'ok' if hit else 'MISS'}",
              flush=True)
    print(f"\nGSM8K: {correct}/{len(rows)} = {correct/len(rows)*100:.1f}%")
    save("gsm8k", {"n": len(rows), "correct": correct, "details": details})


# ------------------------------------------------------------------------ oom
def phase_oom(prompt_tokens=32000, max_streams=16):
    print("=" * 74)
    print("OOM BOUNDARY -- ramp concurrent long requests until failure")
    print("=" * 74)
    reps = int(prompt_tokens / 30)
    rows = []
    n = 1
    while n <= max_streams:
        results = [None] * n
        def worker(i):
            results[i] = call(salt() + PAT * reps + "\n\nSummarize.", 64, timeout=1800)
        threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
        for t in threads: t.start()
        for t in threads: t.join()
        ok = sum(1 for r in results if r and r["ok"])
        free = gpu_free()
        errs = [r["error"] for r in results if r and not r["ok"]][:2]
        rows.append({"streams": n, "ok": ok, "failed": n - ok, "gpu_free_gb": free, "errors": errs})
        print(f"streams {n:>2}: {ok}/{n} ok | free {free} GB" + (f" | {errs}" if errs else ""))
        if ok < n:
            print(f"\nboundary: first failure at {n} concurrent {prompt_tokens}-token streams")
            break
        n *= 2
    else:
        print(f"\nno failure up to {max_streams} concurrent streams")
    save("oom", rows)


if __name__ == "__main__":
    phases = sys.argv[1:] or ["concurrency"]
    for ph in phases:
        {"concurrency": phase_concurrency, "depth": phase_depth, "needle": phase_needle,
         "gsm8k": phase_gsm8k, "oom": phase_oom}[ph]()
