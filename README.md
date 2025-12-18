# Dog Breed Identification (狗的品種辨識)
![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-Latest-orange.svg)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/shuru921921/dog-breed-identification)

這個專案是針對 Kaggle 上的 [Dog Breed Identification](https://www.kaggle.com/c/dog-breed-identification) 競賽所開發的。目標是透過 **卷積神經網絡 (Convolutional Neural Networks, CNN)** 來辨識照片中狗狗的品種。

---

## 小平台 (Demo)
我將訓練好的 **EfficientNet-B2** 和 **ResNet-50** 模型權重部署到了 Hugging Face：
👉 **[點此進入 Dog Breed Identification 平台](https://huggingface.co/spaces/shuru921921/dog-breed-identification)**

---

## 專案概述 (Project Overview)
本專案利用 **深度學習 (Deep Learning)** 技術處理 **多分類問題 (Multi-class Classification)**。數據集中包含 120 種不同的狗狗品種。

## 技術與模型 (Techniques & Models)
在這個專案中，我使用了 **遷移學習 (Transfer Learning)** 技術，並嘗試了以下模型以及技術：

* **ResNet-50**: 採用 **殘差架構 (Residual Architecture)**，解決深層網絡的梯度消失問題。
* **EfficientNet-B2**: 使用 **複合縮放 (Compound Scaling)** 兼顧計算效率與精確度。
* **Ensemble Learning (集成學習)**: 透過 `emsamble.py` 結合多個模型的預測結果，以提升整體的 **穩健性 (Robustness)** 與預測表現。
* **Data Augmentation (數據增強)**: 包含隨機旋轉 (Rotation)、翻轉 (Flipping) 與歸一化 (Normalization)。
* **Learning Rate Scheduler**: 使用 `ReduceLROnPlateau` 根據驗證損失動態調整 **Learning Rate (學習率)**。

## 資料集 (Dataset)
數據來源於 [Kaggle Dog Breed Identification](https://www.kaggle.com/c/dog-breed-identification/data)。
* **訓練集 (Training Set)**: 10,222 張圖片。
* **測試集 (Test Set)**: 10,357 張圖片。
* **類別數量 (Number of Classes)**: 120 種狗的品種。

## 檔案結構 (File Structure)
* `resNet50.ipynb`: 實作 ResNet-50 模型的訓練與評估 (Notebook)。
* `EfficientNetB2.py`: 使用 EfficientNet-B2 模型進行訓練的 Python 腳本。
* `emsamble.py`: 用於整合不同模型預測結果的腳本 (Ensemble logic)。
* `submission/`: 存放預測結果以供上傳至 Kaggle 的資料夾。

## 如何開始 (Getting Started)

1.  **安裝依賴環境 (Install Dependencies)**:
    ```bash
    pip install torch torchvision pandas numpy matplotlib pillow tqdm scikit-learn
    ```
2.  **下載數據 (Download Data)**:
    從 Kaggle 下載數據並解壓縮至專案目錄。(需注意資料夾檔案名稱與存放位置)
3.  **執行訓練 (Running Training)**:
    直接執行 Python 腳本或在 Jupyter Notebook 中執行代碼塊。



