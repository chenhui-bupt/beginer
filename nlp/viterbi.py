from numpy import *
def viterbi(obs, states, start_p, trans_p, emit_p):
    """

    :param obs: 观测序列
    :param states: 隐状态
    :param start_p: 初始概率（隐状态）
    :param trans_p: 转移概率（隐状态）
    :param emit_p: 发射概率（隐状态表现为显状态的概率）
    :return:
    """
    V = [{}]  # 路径概率表 V[时间][隐状态] = 概率
    for y in states:  # 初始化初始状态
        V[0][y] = start_p[y] * emit_p[y][obs[0]]

    for t in range(1, len(obs)):
        V.append({})
        for yj in states:
            V[t][yj] = max([V[t-1][yi] * trans_p[yi][yj] * emit_p[yj][obs[t]] for yi in states])

    result = []
    for vector in V:
        temp = {}
        temp[list(vector.keys())[argmax(list(vector.values()))]] = max(vector.values())
        result.append(temp)
    return result

states = ["Sunny", "Cloudy", "Rainy"]
obs = ["dry", "dryish", "soggy"]
start_p = {"Sunny": 0.63, "Cloudy": 0.17, "Rainy": 0.20}
trans_p = {
    "Sunny": {"Sunny": 0.5, "Cloudy": 0.375, "Rainy": 0.125},
    "Cloudy": {"Sunny": 0.25, "Cloudy": 0.125, "Rainy": 0.625},
    "Rainy": {"Sunny": 0.25, "Cloudy": 0.375, "Rainy": 0.375}
}
emit_p = {
    "Sunny": {"dry": 0.60, "dryish": 0.20, "soggy": 0.05},
    "Cloudy": {"dry": 0.25, "dryish": 0.25, "soggy": 0.25},
    "Rainy": {"dry": 0.05, "dryish": 0.10, "soggy": 0.5},
}
print(viterbi(obs, states, start_p, trans_p, emit_p))



