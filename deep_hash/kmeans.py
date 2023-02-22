import torch
import random
import copy
import json

"""
K-means算法，在hash的offline stage时为每个向量进行无监督分类
"""

class KMeans():
    def __init__(self,data,k):
        self.data = data
        self.k = k

    def distance(self, p1, p2):
        return torch.sum((p1-p2)**2).sqrt()


    # 随机初始化聚类中心，并返回初始聚类中心的列表
    def generate_center(self)->list:
        # 随机初始化聚类中心
        n = self.data.size(0)
        # data的数据量大小为n，从这n个样本中随机挑选出k个样本作为初始的聚类中心
        rand_id = random.sample(range(n), self.k)
        center = []
        for _id in rand_id:
            center.append(self.data[_id])
        return center

    def converge(self, old_center, new_center):
        # 判断是否收敛
        for i in range(len(old_center)):
            if not old_center[i].equal(new_center[i]):
                return False
        return True

    # 计算所有样本点的聚类中心，并返回每个样本点的标签
    def forward(self):
        center = self.generate_center()
        n = self.data.size(0)
        labels = torch.zeros(n).long()
        flag = False

        #当不收敛时就会不断更新聚类中心直至收敛
        while not flag:
            old_center = copy.deepcopy(center)

            for i in range(n):
                cur = self.data[i]
                min_dis = 10**9

                # 为每个点计算与k个聚类中心的距离，如果这个距离小于与之前聚类中心的最小距离，则认为该点应当被归为当前中心的类
                # j为聚类中心的索引下标
                for j in range(self.k):
                    dis = self.distance(cur, center[j])
                    if dis < min_dis:
                        min_dis = dis
                        labels[i] = j

            # 更新聚类中心
            for j in range(self.k):
                # 所有被归为该类的点计算各个方向的均值，所得的点为该类的新聚类中心
                center[j] = torch.mean(self.data[labels == j], dim=0)

            flag = self.converge(old_center, center)

        return labels, center


# 把vec文件中存放的内容仅仅把每个向量读取出来
def read_vec_as_data(vec_file):
    data = []
    with open(vec_file,'r') as vf:
        for line in vf.readlines():
            js = json.loads(line)
            vec = js["code_vec"]
            data.append(vec)
    return torch.tensor(data)

if __name__ == "__main__":
    data = read_vec_as_data("../CodeSearchNet/code_vec/java_test_part_vec.jsonl")
    kmeans = KMeans(data,1)
    labels,_ = kmeans.forward()
    print(labels)