def knapsack_Dynamic_Programming(weights, values, capacity, precision=4):
    """
    支持小数容量的 0-1 背包问题，返回最大价值、选中物品索引和总重量。
    :param weights: List[float]，每个物品的重量
    :param values: List[int]，每个物品的价值
    :param capacity: float，背包容量
    :param precision: int，小数精度（用于缩放处理）
    :return: (最大总价值, 被选中的物品索引列表, 被选中物品总重量)
    """
    scale = 10 ** precision
    int_weights = [int(w * scale + 0.5) for w in weights]
    int_capacity = int(capacity * scale + 0.5)
    n = len(weights)

    dp = [[0] * (int_capacity + 1) for _ in range(n + 1)]
    keep = [[False] * (int_capacity + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        w = int_weights[i - 1]
        v = values[i - 1]
        for j in range(int_capacity + 1):
            if w > j:
                dp[i][j] = dp[i - 1][j]
            else:
                if dp[i - 1][j] < dp[i - 1][j - w] + v:
                    dp[i][j] = dp[i - 1][j - w] + v
                    keep[i][j] = True
                else:
                    dp[i][j] = dp[i - 1][j]

    res_value = dp[n][int_capacity]
    res_items = []
    total_weight_int = 0
    j = int_capacity
    for i in range(n, 0, -1):
        if keep[i][j]:
            res_items.append(i - 1)
            total_weight_int += int_weights[i - 1]
            j -= int_weights[i - 1]

    res_items.reverse()
    total_weight = total_weight_int / scale
    return res_value, res_items, total_weight
