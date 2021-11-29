"""
train NCF model

Referenece : ZiyaoGeng/Recommender-System-with-TF2.0
"""
import os
import pandas as pd
import tensorflow as tf
from time import time
from tensorflow.keras.optimizers import Adam

from model import NCF
from evaluate import *
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from data_process.ml_1m import create_ml_1m_dataset

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if __name__ == "__main__":
    # =============================== GPU ==============================
    gpu = tf.config.experimental.list_physical_devices(device_type='GPU')
    print(gpu)
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    # ========================= Hyper Parameters =======================
    file = '../dataset/ml-1m/ratings.dat'
    trans_score = 1
    test_neg_num = 100
    
    embed_dim = 32
    embed_reg = 1e-6
    hidden_units = [256, 128, 64]
    activation = 'relu'
    dropout = 0.2
    K = 10
    
    learning_rate = 0.001
    batch_size = 512
    epochs = 10
    
    # ========================== Create dataset =======================
    feature_columns, train, val, test = create_ml_1m_dataset(file=file,
                                                        trans_score = trans_score,
                                                        embed_dim = embed_dim,
                                                        test_neg_num = test_neg_num)
    # ============================Build Model==========================
    model = NCF(feature_columns, hidden_units, dropout, activation, embed_dim)
    model.summary()
    # ============================Compile==============================
    model.compile(optimizer=Adam(learning_rate=learning_rate))    
    
    results = []
    for epoch in range(1, epochs+1):
        # ==============================Fit============================
        t1 = time()
        model.fit(
            train,
            None,
            validation_data=(val, None),
            epochs=1,
            batch_size=batch_size,
        )    
        # ===========================Test================================
        t2 = time()
        if epoch % 2 == 0:
            hit_rate, ndcg = evaluate_model(model, test, K)
            print(f'Iteration {epoch} Fit [{t2-t1:.1f} s], Evaluate [{time() - t2:.1f} s]: HR = {hit_rate:.4f}, NDCG = {ndcg:.4f}')
            results.append([epoch, t2 - t1, time() - t2, hit_rate, ndcg])

    # ========================== Write Log ===========================
    pd.DataFrame(results, columns=['Iteration', 'fit_time', 'evaluate_time', 'hit_rate', 'ndcg'])\
        .to_csv('log/NCF_log_dim_{}__K_{}.csv'.format(embed_dim, K), index=False)    
        