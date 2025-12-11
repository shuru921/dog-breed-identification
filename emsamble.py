import pandas as pd
import numpy as np

#emsamble
effb2 = pd.read_csv("outputs/submission_efficient_B2.csv")
resnet = pd.read_csv("outputs/submission_resnet_fixed.csv")

assert all(effb2.columns == resnet.columns)  # 欄位 品種名稱
assert all(effb2["id"] == resnet["id"]) # 圖片順序要一致

blend = effb2.copy()
blend.iloc[:, 1:] = 0.6 * effb2.iloc[:, 1:] + 0.4 * resnet.iloc[:, 1:]

blend.to_csv("outputs/submission_ensemble_0.6_0.4.csv", index=False)
print("✅ Ensemble submission saved.")