import math


class Node:
    end_x = 9
    end_y = 9

    def __init__(self, pos, parent):
        self.x = pos[0]
        self.y = pos[1]
        self.parent = parent
        if parent is not None:
            g_c2p = self.getG(self, parent)  # 缺的参数默认为0
            self.g = g_c2p + parent.g
            self.h = self.getH(Node.end_x, Node.end_y)
            self.f = self.g + self.h
        else:
            self.g = 0
            self.h = 0
            self.f = 0

    def reset_parent(self, parent, new_G):
        if parent is not None:
            self.g = new_G
            self.f = self.g + self.h
        self.parent = parent

    def getG(self, node1, node2):
        x1 = abs(node1.x - node2.x)
        y1 = abs(node1.y - node2.y)
        if x1 == 0 or y1 == 0:
            return 10
        else:
            return 14

    def getH(self, x, y):
        return (abs(x - self.x) + abs(y - self.y)) * 10


class AStar:
    def __init__(self, start, end, danger, step=0.5, inflation=0.3):
        self.inflation = inflation
        self.start = Node(start, None)
        self.end = Node(end, None)
        self.danger = danger
        self.open_list = {}
        self.close_list = {}
        self.path_node = []
        self.path_smooth = []

        self.search_offset = [[step, -step],
                              [step, 0],
                              [step, step],
                              [0, -step],
                              [0, step],
                              [-step, -step],
                              [-step, 0],
                              [-step, step]]

        if self.find_path(self.start, self.end):
            self.mark_path(self.end)
            print('over')
            self.smooth()

    def limit_area(self, node):  # return node if is  area of security
        for i in range(len(self.danger)):
            if (node.x - self.danger[i].x) ** 2 + (node.y - self.danger[i].y) ** 2 <= (
                    self.danger[i].r + self.inflation) ** 2:
                return None
        return node

    def add2open(self, node):
        self.open_list.pop((node.x, node.y))  # 将当前节点移出open_list,要么已经确定在规划里面要么是障碍物区域
        self.close_list[(node.x, node.y)] = node  # 将当前节点移入close_list，下次迭代将不再考虑该点

        _adjacent = []  # 空列表，存储相邻的且在安全区域的八个节点
        for offset in self.search_offset:
            NodeX = node.x + offset[0]
            NodeY = node.y + offset[1]
            nolim_node = self.limit_area(Node([NodeX, NodeY], node))  # 与当前父节点相邻的节点，判断是否在安全区域
            if nolim_node: _adjacent.append(nolim_node)  # 是在安全区域，加入列表_adjacent中

        for a in _adjacent:  # 相对于障碍物来说是安全的区域
            if (a.x - self.end.x) ** 2 + (a.y - self.end.y) ** 2 < 1:
                new_G = Node.getG(self, a, node) + node.g
                self.end.reset_parent(node, new_G)
                print('find path finish')
                return True
            if (a.x, a.y) in self.close_list:  # 已经在close_list中，说明已经不可用于迭代
                continue
            if (a.x, a.y) not in self.open_list:  # 还没加入在close_list中，但可以用于迭代
                self.open_list[(a.x, a.y)] = a
                # print(a.x, a.y, a.f, a.g, a.h)
            else:  # 当把新节点都加入open_list里面之后，进行最优的迭代
                exist_node = self.open_list[(a.x, a.y)]  # 将a节点暂存
                new_G = Node.getG(self, a, node) + node.g  # node是当前的父节点，a是下一个可能用于迭代的节点，g=10或g=14
                if new_G < exist_node.g:
                    exist_node.reset_parent(node, new_G)  # 迭代的节点当前的g值比当前父节点移动到其位置的g值大，更改并设为下一个父节点
                    # print("node", node.x, node.y, node.f, node.g, node.h)
                    # print("exist_node", exist_node.x, exist_node.y, exist_node.f, exist_node.g, exist_node.h)
        # print(node.x, node.y, node.f, node.g, node.h)
        return False

    def minf_node(self):
        if len(self.open_list) == 0:
            raise Exception('not exist path')

        _min = 99999999999
        _k = (self.start.x, self.start.y)  # _k代表字典元素，可理解为地址，即坐标值做字典
        for k, v in self.open_list.items():  # v代表字典对应的值，包括属性f,g,h
            if _min > v.f:  # 找出最小代价的节点，即坐标值
                _min = v.f
                _k = k
        return self.open_list[_k]

    def find_path(self, start, end):
        self.open_list[(start.x, start.y)] = start  # 传入节点对象的值和地址
        the_node = start  # 传入节点对象的地址
        # ii = 0
        try:
            while not self.add2open(the_node):  # 只要路径规划没结束
                # ii += 1
                the_node = self.minf_node()  # 得到当前最小代价的节点，继续循环判断是否到终点
                # print(the_node.x, the_node.y, the_node.f, the_node.g, the_node.h)
        except:
            return False

        # print("迭代次数", ii)
        return True

    def mark_path(self, node):
        self.path_node.append((node.x, node.y))
        # print("debug", self.path_node)
        if node.parent is None:
            return
        self.mark_path(node.parent)

    def isAvoidDanger(self, x1, y1, x2, y2):
        for i in range(len(self.danger)):
            # 计算垂足坐标
            a = (x2 - x1) * (self.danger[i].x - x1) + (y2 - y1) * (self.danger[i].y - y1)
            b = (x2 - x1) ** 2 + (y2 - y1) ** 2
            u = a / b
            x = x1 + u * (x2 - x1)
            y = y1 + u * (y2 - y1)

            dis = math.sqrt((x - self.danger[i].x) ** 2 + (y - self.danger[i].y) ** 2)
            dis1 = math.sqrt((x1 - self.danger[i].x) ** 2 + (y1 - self.danger[i].y) ** 2)
            dis2 = math.sqrt((x2 - self.danger[i].x) ** 2 + (y2 - self.danger[i].y) ** 2)

            if dis < self.danger[i].r + self.inflation:
                if dis1 < self.danger[i].r + self.inflation or dis2 < self.danger[i].r + self.inflation:
                    return False
                else:
                    if (x1 <= x <= x2) or (x2 <= x <= x1):
                        return False
        return True

    def smooth(self):
        route = self.path_node[:]
        self.path_smooth.append((self.end.x, self.end.y))
        while True:
            n = len(route)
            if n == 1:
                break

            for i in range(n - 1, 0, -1):
                if self.isAvoidDanger(route[0][0], route[0][1], route[i][0], route[i][1]):
                    # print(i, route[0][0], route[0][1], route[i][0], route[i][1])
                    self.path_smooth.append((route[i][0], route[i][1]))
                    for j in range(i):
                        route.pop(0)
                        # print(route)
                    break

    def path(self):
        return self.path_smooth
