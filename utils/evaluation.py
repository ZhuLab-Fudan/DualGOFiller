import math
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy import sparse as sp
from joblib import Parallel, delayed
from numba import jit
import heapq


@jit(nopython=True)
def find_k_largest(K, candidates):
    # 初始化一个列表来存储前K个候选元素
    n_candidates = []
    for iid, score in enumerate(candidates[:K]):
        n_candidates.append((score, iid))
    # 将列表转换为最小堆
    heapq.heapify(n_candidates)
    for iid, score in enumerate(candidates[K:]):
        # 如果当前分数大于堆顶元素的分数，则替换堆顶元素
        if score > n_candidates[0][0]:
            heapq.heapreplace(n_candidates, (score, iid + K))
    # 将堆中的元素按分数从大到小排序
    n_candidates.sort(key=lambda d: d[0], reverse=True)
    # 提取排序后的元素的索引
    ids = [term[1] for term in n_candidates]
    # 提取排序后的元素的分数
    k_largest_scores = [term[0] for term in n_candidates]
    return ids, k_largest_scores


class Metric(object):
    def __init__(self):
        pass

    @staticmethod
    def hits(origin, res):
        hit_count = {}
        for protein in origin:
            terms = list(origin[protein].keys())
            predicted = [term[0] for term in res[protein]]
            hit_count[protein] = len(set(terms).intersection(set(predicted)))
        return hit_count

    @staticmethod
    def hit_ratio(origin, hits):
        """
        Note: This type of hit ratio calculates the fraction:
         (# retrieved interactions in the test set / #all the interactions in the test set)
        """
        total_num = 0
        for protein in origin:
            terms = list(origin[protein].keys())
            total_num += len(terms)
        hit_num = 0
        for protein in hits:
            hit_num += hits[protein]
        return round(hit_num/total_num,5)

    @staticmethod
    def precision(hits, N):
        # In topN evaluation, we only consider the topN terms for each protein.
        # So, the precision is the fraction of retrieved terms that are relevant to the protein.

        # The formula is:
        # Precision = (# of relevant terms retrieved) / (N * # of proteins)
        # where N is the number of terms to be retrieved. (macro average)
        prec = sum([hits[protein] for protein in hits])
        return round(prec / (len(hits) * N),5)

    @staticmethod
    def recall(hits, origin):
        # The formula is:
        # Recall = (# of relevant terms retrieved) / (# of relevant terms in the test set)
        # where # of relevant terms in the test set is the total number of terms in the test set. (macro average)
        recall_list = [hits[protein]/len(origin[protein]) for protein in hits]
        recall = round(sum(recall_list) / len(recall_list),5)
        return recall

    @staticmethod
    def F1(prec, recall):
        if (prec + recall) != 0:
            return round(2 * prec * recall / (prec + recall),5)
        else:
            return 0

    @staticmethod
    def NDCG(origin,res,N):
        sum_NDCG = 0
        for protein in res:
            DCG = 0
            IDCG = 0
            #1 = related, 0 = unrelated
            for n, term in enumerate(res[protein]):
                if term[0] in origin[protein]:
                    DCG+= 1.0/math.log(n+2,2)
            for n, term in enumerate(list(origin[protein].keys())[:N]):
                IDCG+=1.0/math.log(n+2,2)
            sum_NDCG += DCG / IDCG
        return round(sum_NDCG / len(res),5)


def ranking_evaluation(origin, res, N):
    measure = []
    for n in N:
        predicted = {}
        for protein in res:
            predicted[protein] = res[protein][:n]
        indicators = []
        if len(origin) != len(predicted):
            print('The Lengths of test set and predicted set do not match!')
            exit(-1)
        hits = Metric.hits(origin, predicted)
        hr = Metric.hit_ratio(origin, hits)
        indicators.append('Hit Ratio:' + str(hr) + '\n')
        prec = Metric.precision(hits, n)
        indicators.append('Precision:' + str(prec) + '\n')
        recall = Metric.recall(hits, origin)
        indicators.append('Recall:' + str(recall) + '\n')
        F1 = Metric.F1(prec, recall)
        indicators.append('F1:' + str(F1) + '\n')
        NDCG = Metric.NDCG(origin, predicted, n)
        indicators.append('NDCG:' + str(NDCG) + '\n')
        measure.append('Top ' + str(n) + '\n')
        measure += indicators
    return measure
