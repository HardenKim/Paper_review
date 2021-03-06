"""
train FM model

Referenece : ZiyaoGeng/Recommender-System-with-TF2.0
"""
import tensorflow as tf
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC

from model import FM
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from data_process.criteo import create_criteo_dataset

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if __name__ == "__main__":
    # =============================== GPU ==============================
    gpu = tf.config.experimental.list_physical_devices(device_type='GPU')
    print(gpu)
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    # ========================= Hyper Parameters =======================
    file = '../dataset/Criteo/train.txt'
    read_part = True
    sample_num = 5_000_000 # get_chunk(sample_num)
    test_size = 0.2
    
    k = 8
    
    learning_rate = 0.001
    batch_size = 4096
    epochs = 10
    
    # ========================== Create dataset =======================
    feature_columns, train, test = create_criteo_dataset(file=file,
                                           read_part=read_part,
                                           sample_num=sample_num,
                                           test_size=test_size)
    train_X, train_y = train
    test_X, test_y = test
    
    # ============================Build Model==========================
    model = FM(feature_columns=feature_columns, k=k)
    model.summary()
    # ============================Compile============================
    model.compile(loss=binary_crossentropy, optimizer=Adam(learning_rate=learning_rate),
                    metrics=[AUC()])
    
    # ============================model checkpoint======================
    # check_path = '../save/fm_weights.epoch_{epoch:04d}.val_loss_{val_loss:.4f}.ckpt'
    # checkpoint = tf.keras.callbacks.ModelCheckpoint(check_path, save_weights_only=True,
    #                                                 verbose=1, period=5)
    
    # ==============================Fit==============================
    model.fit(
        train_X,
        train_y,
        epochs=epochs,
        callbacks=[EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)],  # checkpoint
        batch_size=batch_size,
        validation_split=0.1
    )
    
    # ===========================Test==============================
    print('test AUC: %f' % model.evaluate(test_X, test_y, batch_size=batch_size)[1])

    
    