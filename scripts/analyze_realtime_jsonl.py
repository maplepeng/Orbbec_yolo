#!/usr/bin/env python3
import argparse
import json
import math
from collections import Counter
from typing import Dict, List, Optional, Tuple


def q_linear(values: List[float], q: float) -> float:
    if not values:
        return float("nan")
    q = max(0.0, min(1.0, q))
    s = sorted(values)
    pos = q * (len(s) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return s[lo]
    frac = pos - lo
    return s[lo] * (1.0 - frac) + s[hi] * frac


def mean(values: List[float]) -> float:
    if not values:
        return float("nan")
    return sum(values) / len(values)


def slope_linear(x: List[float], y: List[float]) -> float:
    """Return dy/dx slope (least squares)."""
    if len(x) != len(y) or len(x) < 2:
        return float("nan")
    mx = mean(x)
    my = mean(y)
    num = 0.0
    den = 0.0
    for xi, yi in zip(x, y):
        dx = xi - mx
        num += dx * (yi - my)
        den += dx * dx
    if den == 0.0:
        return float("nan")
    return num / den


def stats(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"n": 0.0, "mean": float("nan"), "min": float("nan"), "max": float("nan"),
                "p50": float("nan"), "p90": float("nan"), "p99": float("nan")}
    return {
        "n": float(len(values)),
        "mean": mean(values),
        "min": min(values),
        "max": max(values),
        "p50": q_linear(values, 0.50),
        "p90": q_linear(values, 0.90),
        "p99": q_linear(values, 0.99),
    }


def fmt(v: float, digits: int = 3) -> str:
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return "N/A"
    return f"{v:.{digits}f}"


def load_log(path: str) -> Tuple[Optional[Dict], List[Dict]]:
    meta = None
    frames: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at line {lineno}: {e}") from e
            t = obj.get("type")
            if t == "meta":
                meta = obj
            elif t == "frame":
                frames.append(obj)
    return meta, frames


def first_last_delta(values: List[float]) -> Tuple[float, float, float, int]:
    if len(values) < 2:
        return float("nan"), float("nan"), float("nan"), 0
    k = max(10, len(values) // 10)
    k = min(k, len(values) // 2)
    first = values[:k]
    last = values[-k:]
    first_m = mean(first)
    last_m = mean(last)
    return first_m, last_m, (last_m - first_m), k


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze realtime debug JSONL (pre/C/gui/latency decomposition)."
    )
    parser.add_argument("jsonl", help="Path to realtime JSONL log file")
    args = parser.parse_args()

    meta, frames = load_log(args.jsonl)
    if not frames:
        raise SystemExit("No frame records found.")

    fps = float(meta.get("fps")) if meta and meta.get("fps") is not None else float("nan")
    deadline_ms = float(meta.get("deadline_ms")) if meta and meta.get("deadline_ms") is not None else float("nan")
    sync_ref_meta = meta.get("sync_reference_name") if meta else None
    frame_period_ms = (1000.0 / fps) if (not math.isnan(fps) and fps > 0.0) else float("nan")

    pre_ms: List[float] = []
    c_ms: List[float] = []
    gui_ms: List[float] = []
    latency_ms: List[float] = []
    recon_err_ms: List[float] = []
    logged_latency_err_ms: List[float] = []
    source_counts: Counter = Counter()
    sync_ref_counts: Counter = Counter()
    pre_by_source: Dict[str, List[float]] = {}
    system_minus_global_ms: List[float] = []
    start_minus_system_ms: List[float] = []
    start_minus_global_ms: List[float] = []
    exposure_raw: List[float] = []

    for fr in frames:
        t_cap = fr.get("t_capture_for_latency_us")
        t_start = fr.get("t_start_process_us")
        t_end = fr.get("t_end_process_us")
        t_out = fr.get("t_output_us")
        src = fr.get("t_capture_source", "unknown")
        sync_ref_name = fr.get("sync_reference_name", "UNKNOWN")
        system_ts = fr.get("system_ts_us")
        global_ts = fr.get("global_ts_us")
        exposure = fr.get("exposure")
        source_counts[src] += 1
        sync_ref_counts[sync_ref_name] += 1

        if None in (t_cap, t_start, t_end, t_out):
            continue

        pre = (float(t_start) - float(t_cap)) / 1000.0
        c = (float(t_end) - float(t_start)) / 1000.0
        gui = (float(t_out) - float(t_end)) / 1000.0
        lat = (float(t_out) - float(t_cap)) / 1000.0
        recon = pre + c + gui

        pre_ms.append(pre)
        c_ms.append(c)
        gui_ms.append(gui)
        latency_ms.append(lat)
        recon_err_ms.append(lat - recon)
        pre_by_source.setdefault(src, []).append(pre)

        if fr.get("latency_ms") is not None:
            logged_latency_err_ms.append(float(fr["latency_ms"]) - lat)
        if system_ts is not None and global_ts is not None:
            system_minus_global_ms.append((float(system_ts) - float(global_ts)) / 1000.0)
        if system_ts is not None:
            start_minus_system_ms.append((float(t_start) - float(system_ts)) / 1000.0)
        if global_ts is not None:
            start_minus_global_ms.append((float(t_start) - float(global_ts)) / 1000.0)
        if exposure is not None:
            exposure_raw.append(float(exposure))

    n = len(latency_ms)
    if n == 0:
        raise SystemExit("No usable frame records with all timestamps.")

    st_pre = stats(pre_ms)
    st_c = stats(c_ms)
    st_gui = stats(gui_ms)
    st_lat = stats(latency_ms)
    st_recon = stats(recon_err_ms)
    st_logged_err = stats(logged_latency_err_ms) if logged_latency_err_ms else None
    st_sys_minus_glb = stats(system_minus_global_ms)
    st_start_minus_sys = stats(start_minus_system_ms)
    st_start_minus_glb = stats(start_minus_global_ms)
    st_exposure = stats(exposure_raw)

    x = [float(i) for i in range(n)]
    pre_slope_ms_per_frame = slope_linear(x, pre_ms)
    pre_slope_ms_per_1k = pre_slope_ms_per_frame * 1000.0 if not math.isnan(pre_slope_ms_per_frame) else float("nan")
    pre_first_m, pre_last_m, pre_delta_m, k = first_last_delta(pre_ms)

    miss_latency = 0
    miss_c = 0
    if not math.isnan(deadline_ms):
        miss_latency = sum(1 for v in latency_ms if v > deadline_ms)
        miss_c = sum(1 for v in c_ms if v > deadline_ms)

    print("== LOG META ==")
    print(f"path: {args.jsonl}")
    print(f"frames: {n}")
    print(f"fps: {fmt(fps)}")
    print(f"frame_period_ms: {fmt(frame_period_ms)}")
    print(f"deadline_ms: {fmt(deadline_ms)}")
    print(f"sync_reference(meta): {sync_ref_meta if sync_ref_meta is not None else 'N/A'}")
    print(f"capture_source_counts: {dict(source_counts)}")
    print(f"sync_reference_counts(frame): {dict(sync_ref_counts)}")
    print()

    print("== STATS (ms) ==")
    print(f"pre_ms     mean={fmt(st_pre['mean'])} p50={fmt(st_pre['p50'])} p90={fmt(st_pre['p90'])} p99={fmt(st_pre['p99'])}")
    print(f"C_ms       mean={fmt(st_c['mean'])} p50={fmt(st_c['p50'])} p90={fmt(st_c['p90'])} p99={fmt(st_c['p99'])}")
    print(f"gui_ms     mean={fmt(st_gui['mean'])} p50={fmt(st_gui['p50'])} p90={fmt(st_gui['p90'])} p99={fmt(st_gui['p99'])}")
    print(f"latency_ms mean={fmt(st_lat['mean'])} p50={fmt(st_lat['p50'])} p90={fmt(st_lat['p90'])} p99={fmt(st_lat['p99'])}")
    print()

    print("== PRE BY SOURCE (ms) ==")
    for src in sorted(pre_by_source.keys()):
        st = stats(pre_by_source[src])
        print(f"{src:15s} n={int(st['n'])} mean={fmt(st['mean'])} p50={fmt(st['p50'])} p90={fmt(st['p90'])} p99={fmt(st['p99'])}")
    print()

    print("== CONSISTENCY ==")
    print(f"latency - (pre+C+gui) mean={fmt(st_recon['mean'], 6)} ms, p99={fmt(st_recon['p99'], 6)} ms")
    if st_logged_err is not None:
        print(f"logged_latency - recomputed_latency mean={fmt(st_logged_err['mean'], 6)} ms, p99={fmt(st_logged_err['p99'], 6)} ms")
    print()

    print("== TIMESTAMP OFFSETS (ms) ==")
    print(
        f"system-global      n={int(st_sys_minus_glb['n'])} "
        f"mean={fmt(st_sys_minus_glb['mean'])} p50={fmt(st_sys_minus_glb['p50'])} "
        f"p90={fmt(st_sys_minus_glb['p90'])} p99={fmt(st_sys_minus_glb['p99'])}"
    )
    print(
        f"t_start-system     n={int(st_start_minus_sys['n'])} "
        f"mean={fmt(st_start_minus_sys['mean'])} p50={fmt(st_start_minus_sys['p50'])} "
        f"p90={fmt(st_start_minus_sys['p90'])} p99={fmt(st_start_minus_sys['p99'])}"
    )
    print(
        f"t_start-global     n={int(st_start_minus_glb['n'])} "
        f"mean={fmt(st_start_minus_glb['mean'])} p50={fmt(st_start_minus_glb['p50'])} "
        f"p90={fmt(st_start_minus_glb['p90'])} p99={fmt(st_start_minus_glb['p99'])}"
    )
    print()

    if exposure_raw:
        print("== EXPOSURE (raw metadata units) ==")
        print(
            f"exposure           n={int(st_exposure['n'])} mean={fmt(st_exposure['mean'])} "
            f"p50={fmt(st_exposure['p50'])} p90={fmt(st_exposure['p90'])} p99={fmt(st_exposure['p99'])} "
            f"min={fmt(st_exposure['min'])} max={fmt(st_exposure['max'])}"
        )
        print()

    print("== PRE TREND ==")
    print(f"pre slope: {fmt(pre_slope_ms_per_1k)} ms / 1000 frames")
    print(f"pre first({k}) mean={fmt(pre_first_m)} ms, last({k}) mean={fmt(pre_last_m)} ms, delta={fmt(pre_delta_m)} ms")
    print()

    if not math.isnan(deadline_ms):
        print("== DEADLINE MISS ==")
        print(f"latency_miss: {miss_latency}/{n} ({fmt(100.0 * miss_latency / n)}%)")
        print(f"C_only_miss:  {miss_c}/{n} ({fmt(100.0 * miss_c / n)}%)")
        print()

    print("== QUICK DIAGNOSIS ==")
    fixed_offset_hint = (
        not math.isnan(frame_period_ms)
        and abs(st_pre["p50"] - frame_period_ms) < 0.4 * frame_period_ms
        and abs(pre_delta_m) < 3.0
        and abs(pre_slope_ms_per_1k) < 1.0
    )
    backlog_hint = (pre_delta_m > 5.0 and pre_slope_ms_per_1k > 1.0)
    gui_hint = (st_pre["p50"] < 5.0 and st_gui["p50"] > 5.0 and st_gui["p50"] > st_c["p50"])

    if fixed_offset_hint:
        print("- pre_ms is stable near one frame period: likely fixed pipeline phase offset (not queue accumulation).")
    if backlog_hint:
        print("- pre_ms increases over time: possible queue accumulation/backlog.")
    if gui_hint:
        print("- pre_ms is small but gui_ms is large: display/output path may dominate latency.")
    if not (fixed_offset_hint or backlog_hint or gui_hint):
        print("- no single dominant pattern detected; inspect source mix and per-frame timeline.")


if __name__ == "__main__":
    main()
