"""
evaluate model

Referenece : ZiyaoGeng/Recommender-System-with-TF2.0
"""
import numpy as np

def evaluate_model(model, test, K):
    """[evaluate_model]

        Args:
            model : model
            test : test data set
            K (int): top K
        Return:
            hit rate, ndcg
    """
    pred_y = -model.predict(test)
    rank = pred_y.argsort().argsort()[:, 0]
    hr, ndcg = 0.0, 0.0
    for r in rank:
        if r < K:
            hr += 1
            ndcg += 1 / np.log2(r + 2)
    
    return hr / len(rank), ndcg / len(rank)