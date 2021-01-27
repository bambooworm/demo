# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 09:58:41 2021

@author: shenchen
"""

# Train FSDKaggle2018 model
#
import sys
sys.path.append('../..')
from lib_train import *
import shutil
from glob2 import glob
from plot_keras_history import plot_history


def gen_train_data():
    fname = [fn for fn in glob('audio/pos/*.wav')]
    label = ['car']*len(fname)
    data1 = pd.DataFrame(list(zip(fname,label)))
    fname = [fn for fn in glob('audio/neg/*.wav')]
    label = ['none']*len(fname)
    data2 = pd.DataFrame(list(zip(fname,label)))
    data = pd.concat((data1,data2), ignore_index=True)
    data.columns = ['fname','label']
    data.to_csv('train.csv', index=None)

def gen_test_data():
    fname = [fn for fn in glob('audio/test_pos/*.wav')]
    label = ['car']*len(fname)
    data1 = pd.DataFrame(list(zip(fname,label)))
    fname = list(set([fn for fn in glob('audio/test_neg/*.wav')]).difference(set(fname)))
    label = ['none']*len(fname)
    data2 = pd.DataFrame(list(zip(fname,label)))
    data = pd.concat((data1,data2), ignore_index=True)
    data.columns = ['fname','label']
    data.to_csv('test.csv', index=None)

# # 2. Preprocess data if it's not ready
# def preprocessed_train_data():
#     # Load Meta data
#     df_train = pd.read_csv('train.csv')
#     # Plain y_train label
#     plain_y_train = np.array([conf.label2int[l] for l in df_train.label])
#     conf.folder.mkdir(parents=True, exist_ok=True)
#     if not os.path.exists(conf.X_train):
#         XX = mels_build_multiplexed_X(conf, [fname for fname in df_train.fname])
#         X_train, y_train, X_test, y_test = \
#             train_valid_split_multiplexed(conf, XX, plain_y_train, demux=True)
#         np.save(conf.X_train, X_train)
#         np.save(conf.y_train, y_train)
#         np.save(conf.X_test, X_test)
#         np.save(conf.y_test, y_test)

# 2. Preprocess data if it's not ready
def preprocessed_data():
    # Load Meta data
    print('preprocessed train data')
    df_train = pd.read_csv('train.csv')
    # Plain y_train label
    plain_y_train = np.array([conf.label2int[l] for l in df_train.label])
    conf.folder.mkdir(parents=True, exist_ok=True)
    if not os.path.exists(conf.X_train):
        XX = mels_build_multiplexed_X(conf, [fname for fname in df_train.fname])
        X_train, y_train, X, y = \
            train_valid_split_multiplexed(conf, XX, plain_y_train, demux=True)
        X_train = np.concatenate((X_train, X), axis=0)
        y_train = np.concatenate((y_train, y), axis=0)
        np.save(conf.X_train, X_train)
        np.save(conf.y_train, y_train)

    # Load Meta data
    print('preprocessed test data')
    df_test = pd.read_csv('test.csv')
    # Plain y_test labelX
    plain_y_test = np.array([conf.label2int[l] for l in df_test.label])
    conf.folder.mkdir(parents=True, exist_ok=True)
    if not os.path.exists(conf.X_test):
        XX = mels_build_multiplexed_X(conf, [fname for fname in df_test.fname])
        X_test, y_test, X, y = \
            train_valid_split_multiplexed(conf, XX, plain_y_test, demux=True)
        X_test = np.concatenate((X_test, X), axis=0)
        y_test = np.concatenate((y_test, y), axis=0)
        np.save(conf.X_test, X_test)
        np.save(conf.y_test, y_test)

gen_train_data()
gen_test_data()
preprocessed_data()

# 3. Load all dataset & normalize
X_train, y_train = load_audio_datafiles(conf, conf.X_train, conf.y_train, normalize=True)
X_test, y_test = load_audio_datafiles(conf, conf.X_test, conf.y_test, normalize=True)
print('Loaded train:test = {}:{} samples.'.format(len(X_train), len(X_test)))

# 4. Train folds
history, model = train_classifier(conf, fold=0,
                                  dataset=[X_train, y_train, X_test, y_test],
                                  model=None,
                                  show_detail=False,
                                  # init_weights=None, # from scratch
                                  init_weights=conf.best_weight_file, # from scratch
)

# 5. Evaluate
evaluate_model(conf, model, X_test, y_test)

print('___ training finished ___')

fn = glob('weights/*.h5')[-1]
shutil.copy(fn, conf.best_weight_file)
os.system("python convert_keras_to_tf.py --model_type {} --keras_weight {} --out_prefix {}".format(conf.model, conf.best_weight_file, os.path.splitext(conf.best_weight_file)[0]))
print('___ trasfer xxx.h5 to xxx.pb finished ___')


plot_history(history.history, path="history.png")
plt.close()
