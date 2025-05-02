
### This script manages GPU usage by logging users on and off GPUs
import json
import os
from datetime import datetime
from filelock import FileLock

STATUS_FILE = "gpu_status.json"
LOCK_FILE = STATUS_FILE + ".lock"

def log_on(user_name, est_end_time):
    with FileLock(LOCK_FILE):
        with open(STATUS_FILE, "r") as f:
            status = json.load(f)

        # Find first free GPU
        for gpu_id, info in status.items():
            if info is None:
                # Assign GPU
                status[gpu_id] = {
                    "user": user_name,
                    "start_time": datetime.now().isoformat(),
                    "est_end": est_end_time
                }
                with open(STATUS_FILE, "w") as f:
                    json.dump(status, f, indent=2)
                os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
                print(f"✅ Assigned GPU {gpu_id} to {user_name}")
                return

        print("❌ All GPUs are currently in use:")
        for gpu_id, info in status.items():
            print(f"  GPU {gpu_id} → {info['user']} until {info['est_end']}")

def log_off(user_name):
    with FileLock(LOCK_FILE):
        with open(STATUS_FILE, "r") as f:
            status = json.load(f)

        for gpu_id, info in status.items():
            if info and info["user"] == user_name:
                status[gpu_id] = None
                with open(STATUS_FILE, "w") as f:
                    json.dump(status, f, indent=2)
                print(f"✅ {user_name} has logged off GPU {gpu_id}")
                return

        print(f"⚠️ No active GPU found for {user_name}")

# How you use it.....
log_on("alice", "2025-05-02T15:00:00")
# later...
log_off("alice")


## Here’s a function you could run at the start of any cell or add to a watchdog:
from datetime import datetime, timedelta

def check_expiration(user_name, warn_minutes=10, expire_minutes=15):
    with FileLock(LOCK_FILE):
        with open(STATUS_FILE, "r") as f:
            status = json.load(f)

        for gpu_id, info in status.items():
            if info and info["user"] == user_name:
                est_end = datetime.fromisoformat(info["est_end"])
                now = datetime.now()
                delta = est_end - now
                if delta.total_seconds() < 60 * warn_minutes and delta.total_seconds() > 0:
                    print(f"⚠️ GPU session for {user_name} on GPU {gpu_id} is expiring in {int(delta.total_seconds() // 60)} minutes.")
                elif now > est_end + timedelta(minutes=expire_minutes):
                    print(f"⛔ GPU session for {user_name} on GPU {gpu_id} has expired and will be logged off.")
                    status[gpu_id] = None
                    with open(STATUS_FILE, "w") as f:
                        json.dump(status, f, indent=2)
                    print(f"✅ GPU {gpu_id} is now free.")
                return



