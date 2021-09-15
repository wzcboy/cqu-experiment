import networkx as nx
import random
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.animation as animation


color_dict = {'S': '#ffc512', 'I': 'red', 'R': 'green'}

# macos中文显示
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']


# 根据 SIR 模型，更新节点的状态
def updateNodeState(G, node, beta, gamma):
    if G.nodes[node]["state"] == "I":               # 感染者
        p = random.random()                         # 生成一个0到1的随机数
        if p < gamma:                               # gamma的概率恢复
            G.nodes[node]["state"] = "R"            # 将节点状态设置成“R”
    elif G.nodes[node]["state"] == "S":             # 易感者
        p = random.random()                         # 生成一个0到1的随机数
        k = 0                                       # 计算邻居中的感染者数量
        for neibor in G.adj[node]:                  # 查看所有邻居状态，遍历邻居用 G.adj[node]
            if G.nodes[neibor]["state"] == "I":     # 如果这个邻居是感染者，则k加1
                k = k + 1
        if p < 1 - (1 - beta)**k:                   # 易感者被感染
            G.nodes[node]["state"] = "I"


def updateNetworkState(G, beta, gamma):
    # 遍历图中节点，每一个节点状态进行更新
    for node in G:
        updateNodeState(G, node, beta, gamma)


# 返回每一个节点的颜色组成的列表
def get_node_color(G):
    color_list = []
    for node in G:
        # 使用我们前面创建的状态到颜色的映射字典 color_dict
        color_list.append(color_dict[G.nodes[node]["state"]])
    return color_list


# 计算三类人群的数量
def countSIR(G):
    S = 0
    I = 0
    for node in G:
        if G.nodes[node]["state"] == "S":
            S = S + 1
        elif G.nodes[node]["state"] == "I":
            I = I + 1
    return S, I, len(G.nodes) - S - I


def initalize(G):
    # 首先将全部人群初始化为S
    for node in G:
        G.nodes[node]['state'] = 'S'
    # 将度中心性最高的五个结点作为I
    max_degree_list = sorted(nx.degree_centrality(G).items(), key=lambda x: x[1], reverse=True)[:5]
    for node, degree in max_degree_list:
        G.nodes[node]['state'] = 'I'


# 实现动画中每一帧的绘制函数，i为第几帧
def graph_draw(i, G, pos, ax, beta, gamma):
    print(i)
    ax.axis("off")
    ax.set_title("day " + str(i) + " 黄色（易感者），红色（感染者），绿色（康复者）")
    plt.box(False)
    # 第一帧，直接绘制网络
    if i == 0:
        nx.draw(G, with_labels=True, font_color="white", node_color=get_node_color(G), edge_color="#D8D8D8", pos=pos, ax=ax)
    # 其余帧，先更新网络状态，再绘制网络
    else:
        updateNetworkState(G, beta, gamma)
        nx.draw_networkx(G, with_labels=True, font_color="white", node_color=get_node_color(G), edge_color="#D8D8D8", pos=pos, ax=ax)
    plt.close()


def drawDynamicNet(G, days, beta, gamma):
    fig, ax = plt.subplots(figsize=(9, 6))  # 将图的大小设置为 9X6
    pos = nx.spring_layout(G, seed=1)  # 设置网络布局，将 seed 固定为 1
    ax.axis("off")  # 关闭坐标轴
    plt.box(False)  # 不显示框

    animator = animation.FuncAnimation(fig, graph_draw, frames=range(0, days),
                                       fargs=(random_network, pos, ax, beta, gamma), interval=200)

    animator.save('SIR_animation.gif', writer='pillow')


if __name__ == '__main__':
    # 结点数
    N = 200
    # 连接概率
    p = 0.03
    # 要模拟的天数
    days = 90
    # 被感染的概率
    beta = 0.5
    # 康复的概率
    gamma = 0.1

    # 生成无标度网络，节点数和每个节点边数分别为100和2，随机数种子为0
    random_network = nx.erdos_renyi_graph(N, p, 0)

    # 初始化
    initalize(random_network)

    # 模拟SIR90天变化
    # SIR_list = []
    # for t in range(days+1):
    #     SIR_list.append(list(countSIR(random_network)))
    #     updateNetworkState(random_network, beta, gamma)
    #
    # # 绘制出SIR变化曲线图
    # df = pd.DataFrame(SIR_list, columns=["S", "I", "R"])
    # df.plot(figsize=(9, 6), color=[color_dict.get(x) for x in df.columns], title='p='+str(p))
    #
    # plt.show()

    drawDynamicNet(random_network, days, beta, gamma)
