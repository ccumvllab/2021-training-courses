# 2021-training-courses

## Q1

Deadline: 2021-08-13

1. 設計一個多項式函數類別 `Polynomial` 和變數類別 `Variable`，支援以下功能：
    - 計算多項式函數 
    - 對多項式函數進行微分
2. 實作 `plot_polynomial()`，輸出一張 png 檔：
    - 畫出多項式函數圖形
    - 畫出經過多項式函數的點的切線
   
### Packages

- NumPy
- Matplotlib
- pytest

### Prototype

```python
from typing import List, Union

import numpy as np


class Variable:

    def __init__(self, value=None):
        self.value = value
        self.grad = None


class Polynomial:

    def __init__(self, a: List = None):
        self.a = np.array(a)

    def __call__(self, x: Union[float, int]) -> Variable:
        pass
```

### Test Cases

```python
from libs.functional import Polynomial


class TestPolynomial:

    def test_io(self):
        f = Polynomial(a=[1, 2, 3])  # 1 + 2x + 3x^2
        assert f(4).value == 57
        assert f(5).value == 86

    def test_grad(self):
        f = Polynomial(a=[1, 2, 3])  # 1 + 2x + 3x^2
        # gradient: <2 + 6x>
        assert f(4).grad == 26
        assert f(5).grad == 32
```
