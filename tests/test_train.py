import os
from diabetes_service.train import train
from diabetes_service.config import Versions

def test_train_v01(tmp_path):
    res = train(Versions.V01, tmp_path.as_posix())
    assert os.path.exists(res["model_path"])
    assert res["metrics"]["rmse"] > 0

def test_train_v02(tmp_path):
    res = train(Versions.V02, tmp_path.as_posix())
    assert os.path.exists(res["model_path"])
    assert res["metrics"]["rmse"] > 0
