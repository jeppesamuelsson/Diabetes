import argparse, json, os, joblib
from typing import Dict, Any
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from math import sqrt
from data import load_data
from models import make_v01, make_v02
from config import Versions, SEED

def train(version: str, outdir: str) -> Dict[str, Any]:
    X, y = load_data(as_frame=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED)

    if version == Versions.V01:
        pipe = make_v01()
    elif version == Versions.V02:
        pipe = make_v02()
    else:
        raise ValueError(f"Unknown version: {version}")

    pipe.fit(Xtr, ytr)
    preds = pipe.predict(Xte)
    rmse = sqrt(mean_squared_error(yte, preds))

    os.makedirs(outdir, exist_ok=True)
    model_path = os.path.join(outdir, "model.pkl")
    metrics_path = os.path.join(outdir, "metrics.json")

    joblib.dump({"pipeline": pipe, "version": version}, model_path)
    metrics = {"version": version, "rmse": rmse}
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    return {"model_path": model_path, "metrics_path": metrics_path, "metrics": metrics}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", choices=[Versions.V01, Versions.V02], required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()
    result = train(args.version, args.outdir)
    print(json.dumps(result["metrics"], indent=2))

if __name__ == "__main__":
    main()
