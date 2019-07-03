import tensorflow as tf
import numpy as np
import os
from collections import Counter
import librosa
import time

wav_path = 'd://data/wav/train'
label_file = 'd://data/doc/trans/train.word.txt'


def get_wav_files(_wav_path=wav_path):
    _wav_files = []
    for(dirpath, dirnames, filenames) in os.walk(_wav_path):
        for filename in filenames:
            if filename.endswith('.wav') or filename.endswith('.WAV'):
                filename_path = os.sep.join([dirpath,filename])
                if os.stat(filename_path).st_size < 240000:
                    continue
                _wav_files.append(filename_path)
    return _wav_files


wav_files_list = get_wav_files()


def get_wav_label(_wav_files=wav_files_list, _label_file=label_file):
    labels_dict = {}
    with open(_label_file, encoding='utf-8') as f:
        for _label in f:
            _label = _label.strip('\n')
            _label_id = _label.split(' ', 1)[0]
            _label_text = _label.split(' ', 1)[1]
            labels_dict[_label_id] = _label_text
    _labels = []
    new_wav_files = []
    for wav_file in _wav_files:
        wav_id = os.path.basename(wav_file).split('.')[0]
        if wav_id in labels_dict:
            _labels.append(labels_dict[wav_id])
            new_wav_files.append(wav_file)

    return new_wav_files, _labels


wav_files, labels = get_wav_label()
print("加载训练样本:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
print("样本数:", len(wav_files))

all_words = []
for label in labels:
    all_words += [word for word in label]
counter = Counter(all_words)
counter_pairs = sorted(counter.items(), key=lambda x: -x[1])

words, _ = zip(*counter_pairs)
words_size = len(words)
print('词汇表大小:', words_size)

word_num_map = dict(zip(words,range(len(words))))
to_num = lambda word: word_num_map.get(word, len(words))
labels_vector = [list(map(to_num, label)) for label in labels]

label_max_len = np.max([len(label) for label in labels_vector])
print('最长句子的字数:', label_max_len)

wav_max_len = 0
for wav in wav_files:
    wav, sr = librosa.load(wav, mono = True)
    mfcc = np.transpose(librosa.feature.mfcc(wav, sr),[1,0])
    if len(mfcc) > wav_max_len:
        wav_max_len = len(mfcc)
print('最长的语音:', wav_max_len)


