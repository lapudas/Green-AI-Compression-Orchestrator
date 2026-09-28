# 測試使用的剪枝工具- locuslab 提供之剪枝工具
https://github.com/locuslab/wanda

此 github 可做 [magnitude, wanda, sparsegpt] 三種剪枝。

我們使用該 github 提供的工具進行 wanda 的測試。

## 使用方法
1. 在 terminal 輸入 `git clone https://github.com/locuslab/wanda`
2. 依照該 github 中 INSTALL.md 的指示進行環境安裝
3. 在 terminal 輸入 `pip install --upgrade datasets`
4. 到 wanda/lib/data.py 中第 43 行與第 44 行

    將
    ```
        traindata = load_dataset('allenai/c4', 'allenai--c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train')
        valdata = load_dataset('allenai/c4', 'allenai--c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation')
    ```
    改成
    ```
        traindata = load_dataset('allenai/c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train')
        valdata = load_dataset('allenai/c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation')
    ```
5. 即可正常使用，可參考該 github 內的說明使用。

## 補充
該 github 預設使用 gpu ( cuda:0 ) 進行剪枝。

該 github 支援的模型範圍較舊，如 llama 系列僅支援到 llama2，剪枝前建議確認模型是否在支援範圍內。