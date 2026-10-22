#!/usr/bin/env python3
"""Wait for a quiet Linux server, then reject sustained outside CPU activity.

Example: benchmark_idle_guard.py --log build/load.jsonl -- python3 benchmark.py
CPU use is sampled once per second. The command and its foreground descendants
are excluded. Two consecutive samples above the threshold stop its process
group and return 75; short-lived external processes can escape sampling.
"""
import argparse
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time


def snapshot():
    result = {}
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():
            continue
        try:
            raw = (path/'stat').read_text()
            fields = raw[raw.rfind(')') + 2:].split()
            result[int(path.name)] = (int(fields[1]), int(fields[11]) + int(fields[12]), int(fields[19]))
        except (OSError, ValueError, IndexError):
            pass
    return result


def external_cpu(before, after, elapsed, root):
    family = {root}
    while True:
        updated = family | {pid for pid, (parent, _, _) in after.items() if parent in family}
        if updated == family:
            break
        family = updated
    ticks = 0
    active = []
    for pid, (parent, total, started) in after.items():
        if pid in family or pid not in before or before[pid][2] != started:
            continue
        delta = max(0, total - before[pid][1])
        ticks += delta
        if delta:
            active.append((delta, pid))
    cores = ticks / os.sysconf('SC_CLK_TCK') / elapsed
    return cores, [pid for _, pid in sorted(active, reverse=True)[:5]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', type=Path, required=True)
    parser.add_argument('--idle-seconds', type=float, default=20)
    parser.add_argument('--threshold', type=float, default=.5)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not Path('/proc/self/stat').is_file():
        parser.error('this guard requires Linux /proc')
    if (not math.isfinite(args.idle_seconds) or not math.isfinite(args.threshold) or
            args.idle_seconds <= 0 or args.threshold < 0):
        parser.error('idle-seconds must be positive and threshold must be nonnegative')
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('missing command')
    args.log.parent.mkdir(parents=True, exist_ok=True)
    before, then = snapshot(), time.monotonic()
    quiet = 0
    process = None
    violations = 0
    try:
        with args.log.open('w') as log:
            while True:
                time.sleep(1)
                after, now = snapshot(), time.monotonic()
                cores, active = external_cpu(before, after, now - then, os.getpid())
                row = {'utc': time.time(), 'external_cores': cores, 'top_pids': active,
                       'phase': 'running' if process else 'waiting'}
                log.write(json.dumps(row) + '\n')
                log.flush()
                if process is None:
                    quiet = quiet + now - then if cores <= args.threshold else 0
                    if quiet >= args.idle_seconds:
                        print('Quiet sampled interval verified; starting experiment.', flush=True)
                        process = subprocess.Popen(command, start_new_session=True)
                else:
                    violations = violations + 1 if cores > args.threshold else 0
                    if violations >= 2:
                        print('Concurrent CPU work detected; rejecting this experiment.', flush=True)
                        return 75
                    if process.poll() is not None:
                        print('Experiment finished; no sustained sampled CPU violation.', flush=True)
                        return process.returncode
                before, then = after, now
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()


if __name__ == '__main__':
    def terminate(signum, _frame):
        raise SystemExit(128 + signum)

    # Reap the owned experiment group when an SSH session or caller stops us.
    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGHUP, terminate)
    raise SystemExit(main())
