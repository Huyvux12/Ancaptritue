"""Run one local-SSD training stage with periodic atomic Drive backups."""
from pathlib import Path
import subprocess
import sys
import threading

import maichi_handoff as handoff


def run_training(root, backup_dir, backup_seconds=600, **kwargs):
    if backup_seconds < 1: raise ValueError("Backup interval must be positive")
    root, backup_dir = Path(root).resolve(), Path(backup_dir).resolve()
    stop = threading.Event()
    errors = []

    def backup_loop():
        while not stop.wait(backup_seconds):
            try:
                count = handoff.sync_training(root, backup_dir)
                print(f"Drive backup: {count} files", flush=True)
            except Exception as error:
                errors.append(str(error))
                print(f"Drive backup failed; local training continues: {error}", flush=True)

    cmd = [sys.executable, "-u", str(Path(__file__).with_name("maichi_flow.py")), "train", "--root", str(root)]
    for key, value in kwargs.items():
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value: cmd.append(flag)
        elif value is not None: cmd += [flag, str(value)]
    worker = threading.Thread(target=backup_loop, daemon=True)
    worker.start()
    process = subprocess.Popen(cmd)
    try:
        result = process.wait()
        if result: raise subprocess.CalledProcessError(result, cmd)
    except KeyboardInterrupt:
        # Let the trainer catch KeyboardInterrupt and atomically save last.pt.
        import signal
        process.send_signal(signal.SIGINT)
        try: process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            process.terminate(); process.wait(timeout=10)
        raise
    finally:
        stop.set(); worker.join()  # Never terminate a file copy halfway through.
        count = handoff.sync_training(root, backup_dir)
        print(f"Final Drive backup: {count} files. Previous transient errors: {len(errors)}", flush=True)
