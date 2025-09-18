from datasets import load_dataset


ds = load_dataset("RMT-team/babilong", "256k", split="qa1")
i = 45
col = "input"

val = ds[i][col]
with open("256k.txt", "w", encoding="utf-8") as f:
    f.write(val)
