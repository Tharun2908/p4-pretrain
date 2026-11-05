"""Inject a rank-0 SIGKILL before step 12, without editing train.py."""

import os
import signal
import train

original_lr_at = train.lr_at


def crash_at_step(step, *args, **kwargs):
    if step == 12 and int(os.environ["RANK"]) == 0:
        print(
            "[M5 CRASH] Injecting SIGKILL on rank 0 before step 12",
            flush=True,
        )
        os.kill(os.getpid(), signal.SIGKILL)

    return original_lr_at(step, *args, **kwargs)


train.lr_at = crash_at_step

if __name__ == "__main__":
    train.main()
