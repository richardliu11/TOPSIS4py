# -*- coding: utf-8 -*-
"""
@Time ： 2024/3/13 16:15
@Auth ： Richard
@File ：entropy.py
@IDE ：PyCharm
"""
import numpy as np

def entropy_weight(matrix):
    """
    使用熵权法计算各个准则的权重

    参数:
    - matrix: numpy array, 决策矩阵

    返回值:
    - weights: numpy array, 各个准则的权重
    """
    # 计算熵值
    p = matrix / np.sum(matrix, axis=0)
#    entropy = -np.sum(p * np.log(p), axis=0) / np.log(len(matrix))
    # Laplace 平滑
    epsilon = 1e-9
    p_smooth = (matrix + epsilon) / (np.sum(matrix, axis=0) + epsilon)
    entropy = -np.sum(p_smooth * np.log(p_smooth), axis=0) / np.log(len(matrix))

    # 计算权重
    weights = (1 - entropy) / np.sum(1 - entropy)

    return weights

if __name__ == "__main__":
    # 示例决策矩阵
    decision_matrix = np.array([[1, 4, 5],
                                [1, 2, 3],
                                [5, 5, 5]])

    # 计算权重
    weights = entropy_weight(decision_matrix)
    print("各个准则的权重:", weights)
