from dataclasses import dataclass

SEED = 42

FEATURE_NAMES = [
    "age","sex","bmi","bp","s1","s2","s3","s4","s5","s6"
]

@dataclass(frozen=True)
class Versions:
    V01 = "v0.1"
    V02 = "v0.2"
