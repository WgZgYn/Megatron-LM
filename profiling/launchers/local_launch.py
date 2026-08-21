#!/usr/bin/env python3
"""Tiny local multi-process launcher (Windows smoke tests only).

torchrun's elastic rendezvous constructs ``TCPStore`` without ``use_libuv``,
which fails on the local Windows torch build (no libuv). This launcher spawns
N workers directly and lets each worker call
``init_process_group(backend, init_method='env://')``, which on win32 correctly
defaults ``use_libuv=0`` (see ``torch/distributed/rendezvous.py``).

Usage (GLOO CPU smoke, e.g. validating the DP/TP/PP logic without NCCL):
    python launchers/local_launch.py --nproc 2 --port 29500 -- \
        run.py --mode tp --backend gloo --device cpu --steps 3 ...

On a real multi-GPU Linux server use ``torchrun`` instead (see launchers/*.sh).
"""

import argparse
import os
import subprocess
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--nproc", type=int, required=True, help="number of worker processes")
    p.add_argument("--port", type=int, default=29500)
    p.add_argument("--python", default=sys.executable)
    p.add_argument("args", nargs=argparse.REMAINDER,
                   help="everything after `--` is forwarded to run.py")
    a = p.parse_args()

    if not a.args or a.args[0] != "--":
        # argparse.REMAINDER keeps the leading '--'; tolerate it being omitted.
        pass

    # Drop the leading '--' separator if present.
    script_args = a.args[1:] if (a.args and a.args[0] == "--") else a.args
    if not script_args or not script_args[0].endswith(".py"):
        print("expected: local_launch.py --nproc N -- <entrypoint> [args...]", file=sys.stderr)
        sys.exit(2)

    base_env = os.environ.copy()
    base_env["MASTER_ADDR"] = "127.0.0.1"
    base_env["MASTER_PORT"] = str(a.port)
    base_env["WORLD_SIZE"] = str(a.nproc)
    base_env["USE_LIBUV"] = "0"

    procs = []
    for rank in range(a.nproc):
        env = dict(base_env)
        env["RANK"] = str(rank)
        env["LOCAL_RANK"] = str(rank)
        cmd = [a.python, *script_args]
        procs.append(subprocess.Popen(cmd, env=env))

    codes = [proc.wait() for proc in procs]
    sys.exit(0 if all(code == 0 for code in codes) else 1)


if __name__ == "__main__":
    main()
