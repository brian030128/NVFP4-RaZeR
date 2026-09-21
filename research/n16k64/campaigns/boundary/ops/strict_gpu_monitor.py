#!/usr/bin/env python3
"""Independent fail-on-first-error GPU ownership monitor for late required retries.

This operational watchdog is intentionally outside the frozen evaluator source tree.  It
adds a stricter fail-closed layer without changing maps, evaluation, or analysis code.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import pwd
import subprocess
import time
from pathlib import Path


POLL_SECONDS = 20.0
QUERY_TIMEOUT_SECONDS = 10.0
MAX_SAMPLE_GAP_SECONDS = 60.0


def utc() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def append(path: Path, payload: dict) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    temporary.replace(path)


def command(args: list[str]) -> str:
    result = subprocess.run(args, capture_output=True, text=True, timeout=QUERY_TIMEOUT_SECONDS)
    if result.returncode:
        raise RuntimeError(f"{args[0]} exit {result.returncode}: {result.stderr.strip()}")
    return result.stdout


def csv_rows(output: str) -> list[list[str]]:
    return [[cell.strip() for cell in row] for row in csv.reader(output.splitlines()) if row]


def gpu_state(uuid: str) -> dict:
    table = csv_rows(command([
        "nvidia-smi", "--query-gpu=uuid,name,memory.used,utilization.gpu,pci.bus_id",
        "--format=csv,noheader,nounits",
    ]))
    matches = [row for row in table if row and row[0] == uuid]
    if len(matches) != 1 or len(matches[0]) != 5:
        raise RuntimeError(f"target GPU resolution failed for {uuid}: {matches!r}")
    row = matches[0]
    apps = csv_rows(command([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ]))
    return {
        "uuid": row[0], "name": row[1], "memory_used_mib": int(row[2]),
        "utilization_gpu_pct": int(row[3]), "pci_bus_id": row[4],
        "apps": [item for item in apps if len(item) == 4 and item[0] == uuid],
    }


def process_record(row: list[str], container_id: str) -> dict:
    pid = int(row[1])
    proc = Path("/proc") / str(pid)
    if not proc.is_dir():
        raise RuntimeError(f"compute PID vanished before ownership resolution: {pid}")
    uid = proc.stat().st_uid
    try:
        owner = pwd.getpwuid(uid).pw_name
    except KeyError as exc:
        raise RuntimeError(f"compute PID owner unresolved: pid={pid} uid={uid}") from exc
    cgroup = (proc / "cgroup").read_text()
    return {"gpu_uuid": row[0], "pid": pid, "process_name": row[2],
            "used_memory_mib": int(row[3]), "uid": uid, "owner": owner,
            "in_container": container_id in cgroup}


def container_running(container_id: str) -> bool:
    try:
        result = subprocess.run(["docker", "inspect", container_id, "--format", "{{.State.Running}}"],
                                capture_output=True, text=True, timeout=QUERY_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("docker inspect timed out; ownership supervision is unresolved") from exc
    if result.returncode == 0:
        return result.stdout.strip() == "true"
    if "No such object" in result.stderr:
        return False
    raise RuntimeError(f"docker inspect exit {result.returncode}: {result.stderr.strip()}")


def stop_own_container(container_id: str, expected_name: str) -> dict:
    try:
        name = command(["docker", "inspect", container_id, "--format", "{{.Name}}"] ).strip().lstrip("/")
        if name != expected_name:
            return {"stopped": False, "error": f"container name mismatch: {name!r}"}
        result = subprocess.run(["docker", "stop", "-t", "30", container_id],
                                capture_output=True, text=True, timeout=45)
        return {"stopped": result.returncode == 0, "returncode": result.returncode,
                "stdout": result.stdout.strip(), "stderr": result.stderr.strip()}
    except Exception as exc:
        return {"stopped": False, "error": repr(exc)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--uuid", required=True)
    parser.add_argument("--expected-name", required=True)
    parser.add_argument("--wait-seconds", type=float, default=120.0)
    args = parser.parse_args()
    run = Path(args.run_dir)
    cid_path = run / "container.cid"
    deadline = time.monotonic() + args.wait_seconds
    while not cid_path.is_file() and time.monotonic() < deadline:
        time.sleep(0.25)
    if not cid_path.is_file():
        raise SystemExit("container.cid did not appear")
    container_id = cid_path.read_text().strip()
    log = run / "strict_gpu_monitor.jsonl"
    marker = run / "STRICT_MONITOR_INVALID.json"
    last_success: float | None = None
    append(log, {"utc": utc(), "event": "watchdog_started", "target_uuid": args.uuid,
                 "container_id": container_id, "poll_seconds": POLL_SECONDS,
                 "query_timeout_seconds": QUERY_TIMEOUT_SECONDS,
                 "maximum_sample_gap_seconds": MAX_SAMPLE_GAP_SECONDS})
    def invalidate(exc: Exception) -> None:
        reason = f"strict ownership monitor fail-closed: {exc!r}"
        stop = stop_own_container(container_id, args.expected_name)
        payload = {"schema": "mixfp4-strict-monitor-invalid/v1", "utc": utc(),
                   "reason": reason, "target_uuid": args.uuid,
                   "container_id": container_id, "stop_result": stop,
                   "scientific_outputs_valid": False}
        atomic_json(marker, payload)
        append(log, {"utc": payload["utc"], "event": "invalidated", **payload})

    while True:
        try:
            running = container_running(container_id)
        except Exception as exc:
            invalidate(exc)
            raise SystemExit(6)
        if not running:
            break
        started = time.monotonic()
        try:
            state = gpu_state(args.uuid)
            if state["name"] != "NVIDIA RTX A6000":
                raise RuntimeError(f"unexpected GPU model {state['name']!r}")
            processes = [process_record(row, container_id) for row in state.pop("apps")]
            foreign = [row for row in processes if not row["in_container"]]
            if foreign:
                raise RuntimeError(f"foreign or unresolved compute process: {foreign!r}")
            if state["memory_used_mib"] >= 1024 and not processes:
                raise RuntimeError("substantial GPU memory without a resolvable campaign process")
            now = time.monotonic()
            gap = None if last_success is None else now - last_success
            if gap is not None and gap > MAX_SAMPLE_GAP_SECONDS:
                raise RuntimeError(f"verified ownership sample gap {gap:.6f}s exceeds 60s")
            last_success = now
            append(log, {"utc": utc(), "event": "sample", "gap_seconds": gap,
                         "gpu": state, "processes": processes})
        except Exception as exc:
            invalidate(exc)
            raise SystemExit(6)
        elapsed = time.monotonic() - started
        time.sleep(max(0.0, POLL_SECONDS - elapsed))
    append(log, {"utc": utc(), "event": "container_terminal", "container_id": container_id})


if __name__ == "__main__":
    main()
