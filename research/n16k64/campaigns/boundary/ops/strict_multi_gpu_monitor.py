#!/usr/bin/env python3
"""One-query-set fail-closed ownership watchdog for multiple campaign GPUs."""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

from strict_gpu_monitor import (MAX_SAMPLE_GAP_SECONDS, QUERY_TIMEOUT_SECONDS, append,
                                atomic_json, command, container_running, csv_rows,
                                process_record, stop_own_container, utc)


POLL_SECONDS = 30.0


@dataclass
class Target:
    run: Path
    uuid: str
    expected_name: str
    cid: str = ""
    active: bool = True

    @property
    def log(self) -> Path:
        return self.run / "strict_gpu_monitor.jsonl"

    @property
    def marker(self) -> Path:
        return self.run / "STRICT_MONITOR_INVALID.json"


def query_all() -> dict[str, dict]:
    table = csv_rows(command([
        "nvidia-smi", "--query-gpu=uuid,name,memory.used,utilization.gpu,pci.bus_id",
        "--format=csv,noheader,nounits",
    ]))
    apps = csv_rows(command([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ]))
    states = {}
    for row in table:
        if len(row) != 5:
            raise RuntimeError(f"malformed GPU row: {row!r}")
        states[row[0]] = {"uuid": row[0], "name": row[1], "memory_used_mib": int(row[2]),
                          "utilization_gpu_pct": int(row[3]), "pci_bus_id": row[4],
                          "apps": [item for item in apps if len(item) == 4 and item[0] == row[0]]}
    return states


def emit(target: Target, combined: Path, payload: dict) -> None:
    append(target.log, payload)
    append(combined, {"run_id": target.run.name, **payload})


def invalidate(target: Target, combined: Path, exc: Exception) -> None:
    reason = f"strict ownership monitor fail-closed: {exc!r}"
    stop = stop_own_container(target.cid, target.expected_name)
    payload = {"schema": "mixfp4-strict-monitor-invalid/v1", "utc": utc(),
               "reason": reason, "target_uuid": target.uuid, "container_id": target.cid,
               "stop_result": stop, "scientific_outputs_valid": False}
    atomic_json(target.marker, payload)
    emit(target, combined, {"utc": payload["utc"], "event": "invalidated", **payload})
    target.active = False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", action="append", required=True,
                        help="RUN_DIR|GPU_UUID|EXPECTED_CONTAINER_NAME")
    parser.add_argument("--combined-log", required=True)
    parser.add_argument("--wait-seconds", type=float, default=120.0)
    args = parser.parse_args()
    targets = []
    for spec in args.target:
        pieces = spec.split("|")
        if len(pieces) != 3:
            parser.error(f"bad target: {spec!r}")
        targets.append(Target(Path(pieces[0]), pieces[1], pieces[2]))
    combined = Path(args.combined_log)
    deadline = time.monotonic() + args.wait_seconds
    for target in targets:
        cid_path = target.run / "container.cid"
        while time.monotonic() < deadline:
            if cid_path.is_file() and cid_path.read_text().strip():
                break
            time.sleep(0.25)
        if not cid_path.is_file() or not cid_path.read_text().strip():
            raise SystemExit(f"nonempty container.cid did not appear for {target.run}")
        target.cid = cid_path.read_text().strip()
        emit(target, combined, {"utc": utc(), "event": "watchdog_started",
                                "target_uuid": target.uuid, "container_id": target.cid,
                                "poll_seconds": POLL_SECONDS,
                                "query_timeout_seconds": QUERY_TIMEOUT_SECONDS,
                                "maximum_sample_gap_seconds": MAX_SAMPLE_GAP_SECONDS})
    last_success: float | None = None
    invalid_seen = False
    while any(target.active for target in targets):
        for target in targets:
            if not target.active:
                continue
            try:
                running = container_running(target.cid)
            except Exception as exc:
                invalidate(target, combined, exc); invalid_seen = True; continue
            if not running:
                emit(target, combined, {"utc": utc(), "event": "container_terminal",
                                        "container_id": target.cid})
                target.active = False
        active = [target for target in targets if target.active]
        if not active:
            break
        started = time.monotonic()
        try:
            states = query_all()
            now = time.monotonic()
            gap = None if last_success is None else now - last_success
            if gap is not None and gap > MAX_SAMPLE_GAP_SECONDS:
                raise RuntimeError(f"verified ownership sample gap {gap:.6f}s exceeds 60s")
            last_success = now
        except Exception as exc:
            for target in active:
                invalidate(target, combined, exc)
            invalid_seen = True
            break
        for target in active:
            try:
                state = states.get(target.uuid)
                if state is None:
                    raise RuntimeError(f"target GPU absent: {target.uuid}")
                if state["name"] != "NVIDIA RTX A6000":
                    raise RuntimeError(f"unexpected GPU model {state['name']!r}")
                processes = [process_record(row, target.cid) for row in state.pop("apps")]
                foreign = [row for row in processes if not row["in_container"]]
                if foreign:
                    raise RuntimeError(f"foreign or unresolved compute process: {foreign!r}")
                if state["memory_used_mib"] >= 1024 and not processes:
                    raise RuntimeError("substantial GPU memory without a resolvable campaign process")
                emit(target, combined, {"utc": utc(), "event": "sample", "gap_seconds": gap,
                                        "gpu": state, "processes": processes})
            except Exception as exc:
                invalidate(target, combined, exc); invalid_seen = True
        elapsed = time.monotonic() - started
        time.sleep(max(0.0, POLL_SECONDS - elapsed))
    raise SystemExit(6 if invalid_seen else 0)


if __name__ == "__main__":
    main()
