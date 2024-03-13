# -*- coding: utf-8 -*-
"""
@Time ： 2024/3/12 13:57
@Auth ： Richard
@File ：topsis_test.py
@IDE ：PyCharm
"""

from sklearn.preprocessing import normalize
from sklearn.metrics import euclidean_distances

import numpy as np

def topsis_sklearn(decision_matrix, weights):
    """
    使用 scikit-learn 实现 TOPSIS 算法

    参数:
    - decision_matrix: numpy array, 决策矩阵
    - weights: numpy array, 每个准则的权重

    返回值:
    - topsis_score: numpy array, 每个备选方案的 TOPSIS 分数
    """
    # 归一化决策矩阵
    normalized_matrix = normalize(decision_matrix, axis=0)

    # 将每列乘以其权重
    weighted_matrix = normalized_matrix * weights

    # 计算理想解和负理想解
    ideal_solution = np.max(weighted_matrix, axis=0)
    negative_ideal_solution = np.min(weighted_matrix, axis=0)

    # 计算到理想解和负理想解之间的欧氏距离
    d_pos = euclidean_distances(weighted_matrix, [ideal_solution])
    d_neg = euclidean_distances(weighted_matrix, [negative_ideal_solution])

    # 计算 TOPSIS 分数
    topsis_score = d_neg / (d_pos + d_neg)

    return topsis_score


if __name__ == "__main__":
    # 示例决策矩阵和权重
    decision_matrix = np.array([[250, 16, 12, 5],
                                [200, 16, 8, 3],
                                [300, 32, 16, 4],
                                [275, 32, 8, 4]])

    weights = np.array([0.25, 0.25, 0.25, 0.25])

    # 计算 TOPSIS 分数
    topsis_scores = topsis_sklearn(decision_matrix, weights)
    print("TOPSIS 分数:", topsis_scores)

