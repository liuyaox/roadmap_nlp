

# Part I Basic and Overview

## 1. Python + ML Basic

### 机器学习全流程笔记

既有理论知识，也有代码实现，包括Github热榜Top3

Github: <https://github.com/machinelearningmindset/machine-learning-course> (Python, sklearn, Tensorflow)

Website: <https://machine-learning-course.readthedocs.io/en/latest/>

### Data Science Cheatsheets

Github: <https://github.com/faviovazquez/ds-cheatsheets>

## Similarity

欧式距离、曼哈顿距离、余弦距离、相关系数这些easy的就不说了

### 直方图 Similarity

#### Bin-by-bin

相同索引的bin要一一对应，要求2个直方图的bin索引和个数完全一样

**Metric**: 

Correlation, Chi-Square, Alternative Chi-Squre, Intersection, Bhattacharyya Distance, Kullback-Leibler Divergence

**Library**:

OpenCV: [compareHist](https://docs.opencv.org/3.0-beta/modules/imgproc/doc/histograms.html#comparehist)

**Article**:

- [OpenCV: Histogram Comparison](https://docs.opencv.org/3.0-beta/doc/tutorials/imgproc/histograms/histogram_comparison/histogram_comparison.html)

**Code**:

```python
import cv2
import numpy as np
from sklearn.preprocessing import minmax_scale

h1 = np.array([1, 2, 3, 4, 5, 6], dtype=np.float32)   # 需要指定为float32类型，否则报错
h2 = np.array([2, 3, 0, 5, 6, 7], dtype=np.float32)
h1_n = minmax_scale(h1.reshape(1, -1), axis=1).squeeze()   # 需要进行MinMax归一化 参考Article
h2_n = minmax_scale(h2.reshape(1, -1), axis=1).squeeze()

methods = [cv2.HISTCMP_CORREL, cv2.HISTCMP_CHISQR, cv2.HISTCMP_INTERSECT,
           cv2.HISTCMP_BHATTACHARYYA, cv2.HISTCMP_CHISQR_ALT, cv2.HISTCMP_KL_DIV]
for method in methods:
    print('method-', method, ': ', cv2.compareHist(h1_n, h2_n, method))
```
输出
```
method- 0 :  0.7898016794287489
method- 1 :  0.6870749373487413
method- 2 :  2.600000038743019
method- 3 :  0.34054646141665407
method- 4 :  1.5614791418969856
method- 5 :  8.531590503517315
```

#### Cross-bin

2个直方图的bin索引和个数都可以不一样，当直方图有偏移时，也能识别出其相似性

**Metric**: 

[Earth Mover's Distance (EMD)](https://en.wikipedia.org/wiki/Earth_mover%27s_distance)

**Library**:

OpenCV: [EMD](https://docs.opencv.org/3.0-beta/modules/imgproc/doc/histograms.html#emd)

**Article**:

- [向量相似度度量（一）：EMD](https://blog.csdn.net/wangdonggg/article/details/32329879)
- [向量相似度度量（二）：EMD的MATLAB对照实现](https://blog.csdn.net/wangdonggg/article/details/32691445)
- [向量相似度度量（三）：科普-为什么度量方式很重要](https://blog.csdn.net/wangdonggg/article/details/35280735)