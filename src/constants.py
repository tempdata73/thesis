from typing import Final

NUM_REPS: Final[int] = 20
TIMEOUT: Final[int] = 5 * 60  # 5 minutes per instance

bb_raw_options = {
    "name": "bb_raw",
    "msg": False,
    "presolve": True,
    "cuts": False,
    "options": None,
    "threads": 1,
    "timeLimit": TIMEOUT,
    "logPath": "logs/inf",
}

bb_full_options = {
    "name": "bb_full",
    "msg": False,
    "presolve": True,
    "cuts": True,
    "options": ["gomory on", "knapsack off", "probing off"],
    "threads": 1,
    "timeLimit": TIMEOUT,
    "logPath": "logs/inf",
}
