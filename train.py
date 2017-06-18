import numpy as np
import chainer
from chainer import cuda, Variable, optimizers, serializers
import chainer.functions as F
import chainer.links as L
import cupy
import sys, time
import math

from music2melspec import READ_DATASET
from sounds2melspec import READ_DATASET2

'''
入力音楽のサンプリングレートを16000Hzにあげてみた
'''


class MUSIC_NET(chainer.Chain):
    def __init__(self, ):
        super(MUSIC_NET, self).__init__(
            conv1=L.Convolution2D(in_channels=1, out_channels=16,
                                  ksize=(16, 9), stride=4, pad=0,
                                  wscale=0.02 * math.sqrt(16 * 9)),
            conv2=L.Convolution2D(in_channels=16, out_channels=32,
                                  ksize=(5, 3), stride=2, pad=0,
                                  wscale=0.02 * math.sqrt(16 * 5 * 3)),
            conv3=L.Convolution2D(in_channels=32, out_channels=64,
                                  ksize=(3, 3), stride=2, pad=0,
                                  wscale=0.02 * math.sqrt(32 * 3 * 3)),
            fc4=L.Linear(in_size=64 * 14 * 19, out_size=4096,
                         wscale=0.02 * math.sqrt(64 * 14 * 19)),
            fc5=L.Linear(in_size=4096, out_size=7, wscale=0.02 * math.sqrt(4096)),
        )

    def __call__(self, x, t):
        y = self.forward(x)
        loss = F.softmax_cross_entropy(y, t)
        accuracy = F.accuracy(y, t)
        return loss, accuracy

    def forward(self, x):
        conv1 = F.relu(self.conv1(x))
        conv2 = F.relu(self.conv2(conv1))
        conv3 = F.relu(self.conv3(conv2))
        # print(x.shape, conv1.shape, conv2.shape, conv3.shape)
        reshape3 = F.dropout(F.reshape(conv3, (-1, 64 * 14 * 19)), ratio=0.5)
        fc4 = F.dropout(F.relu(self.fc4(reshape3)), ratio=0.5)
        fc5 = self.fc5(fc4)
        # print(reshape3.shape, fc4.shape, fc5.shape)
        return fc5

    def predict(self, x):
        y = self.forward(x)
        return F.softmax(y)


def main():
    CHUNK = 1024
    length = 160
    expected_fs = 16000

    model = MUSIC_NET()

    cuda.get_device(0).use()  # Make a specified GPU current
    cuda.check_cuda_available()
    model.to_gpu()  # Copy the model to the GPU
    xp = cupy

    optimizer = optimizers.Adam(alpha=0.0001)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(0.0001))

    music_data = dict()
    music_data['LiSA1'] = READ_DATASET(wavfile='music/8bit-16000Hz/LiSA/l-01.wav',
                                       chunk=CHUNK, length=length, expected_fs=expected_fs)
    music_data['LiSA2'] = READ_DATASET(wavfile='music/8bit-16000Hz/LiSA/l-02.wav',
                                       chunk=CHUNK, length=length, expected_fs=expected_fs)
    music_data['LiSA3'] = READ_DATASET(wavfile='music/8bit-16000Hz/LiSA/l-03.wav',
                                       chunk=CHUNK, length=length, expected_fs=expected_fs)
    music_data['LiSA4'] = READ_DATASET(wavfile='music/8bit-16000Hz/LiSA/l-04.wav',
                                       chunk=CHUNK, length=length, expected_fs=expected_fs)

    music_data['BUMP1'] = READ_DATASET(wavfile='music/8bit-16000Hz/BUMP/bump-01.wav',
                                       chunk=CHUNK, length=length, expected_fs=expected_fs)
    music_data['BUMP2'] = READ_DATASET(wavfile='music/8bit-16000Hz/BUMP/bump-02.wav',
                                       chunk=CHUNK, length=length, expected_fs=expected_fs)
    music_data['BUMP3'] = READ_DATASET(wavfile='music/8bit-16000Hz/BUMP/bump-03.wav',
                                       chunk=CHUNK, length=length, expected_fs=expected_fs)
    music_data['BUMP4'] = READ_DATASET(wavfile='music/8bit-16000Hz/BUMP/bump-04.wav',
                                       chunk=CHUNK, length=length, expected_fs=expected_fs)

    music_data['Prono1'] = READ_DATASET(wavfile='music/8bit-16000Hz/prono/p-01.wav',
                                        chunk=CHUNK, length=length, expected_fs=expected_fs)
    music_data['Prono2'] = READ_DATASET(wavfile='music/8bit-16000Hz/prono/p-02.wav',
                                        chunk=CHUNK, length=length, expected_fs=expected_fs)
    music_data['Prono3'] = READ_DATASET(wavfile='music/8bit-16000Hz/prono/p-03.wav',
                                        chunk=CHUNK, length=length, expected_fs=expected_fs)
    music_data['Prono4'] = READ_DATASET(wavfile='music/8bit-16000Hz/prono/p-04.wav',
                                        chunk=CHUNK, length=length, expected_fs=expected_fs)

    music_data['Bz1'] = READ_DATASET(wavfile='music/8bit-16000Hz/Bz/bz-01.wav',
                                     chunk=CHUNK, length=length, expected_fs=expected_fs)
    music_data['Bz2'] = READ_DATASET(wavfile='music/8bit-16000Hz/Bz/bz-02.wav',
                                     chunk=CHUNK, length=length, expected_fs=expected_fs)
    music_data['Bz3'] = READ_DATASET(wavfile='music/8bit-16000Hz/Bz/bz-03.wav',
                                     chunk=CHUNK, length=length, expected_fs=expected_fs)
    music_data['Bz4'] = READ_DATASET(wavfile='music/8bit-16000Hz/Bz/bz-04.wav',
                                     chunk=CHUNK, length=length, expected_fs=expected_fs)

    music_data['MrChildren1'] = READ_DATASET(wavfile='music/8bit-16000Hz/MrChildren/mch-01.wav',
                                             chunk=CHUNK, length=length, expected_fs=expected_fs)
    music_data['MrChildren2'] = READ_DATASET(wavfile='music/8bit-16000Hz/MrChildren/mch-02.wav',
                                             chunk=CHUNK, length=length, expected_fs=expected_fs)
    music_data['MrChildren3'] = READ_DATASET(wavfile='music/8bit-16000Hz/MrChildren/mch-03.wav',
                                             chunk=CHUNK, length=length, expected_fs=expected_fs)
    music_data['MrChildren4'] = READ_DATASET(wavfile='music/8bit-16000Hz/MrChildren/mch-04.wav',
                                             chunk=CHUNK, length=length, expected_fs=expected_fs)

    music_data['Utada1'] = READ_DATASET(wavfile='music/8bit-16000Hz/utada/u-01.wav',
                                        chunk=CHUNK, length=length, expected_fs=expected_fs)
    music_data['Utada2'] = READ_DATASET(wavfile='music/8bit-16000Hz/utada/u-02.wav',
                                        chunk=CHUNK, length=length, expected_fs=expected_fs)
    music_data['Utada3'] = READ_DATASET(wavfile='music/8bit-16000Hz/utada/u-03.wav',
                                        chunk=CHUNK, length=length, expected_fs=expected_fs)
    music_data['Utada4'] = READ_DATASET(wavfile='music/8bit-16000Hz/utada/u-04.wav',
                                        chunk=CHUNK, length=length, expected_fs=expected_fs)

    music_data['Other'] = READ_DATASET2(wavdir='wav_16000/',
                                        chunk=CHUNK, length=length, expected_fs=expected_fs)

    artist2label = {'LiSA': 1, 'BUMP': 2, 'Prono': 3, 'Bz': 4, 'MrChildren': 5, 'Utada': 6}

    n_epoch = 1
    n_batch_per_oneclass = 10
    for epoch in np.arange(n_epoch):
        for key in music_data.keys():
            if key == 'Other':
                continue
            print key, artist2label[key[:-1]], music_data[key].n_blocks
            music_data[key].shuffle_indexes()

        n_blocks = music_data['LiSA1'].n_blocks
        for key in music_data.keys():
            if key == 'Other':
                continue
            n_blocks = min(n_blocks, music_data[key].n_blocks)

        end_flag = False
        for i in np.arange(0, n_blocks, n_batch_per_oneclass):
            xs, ts = None, None
            for key in music_data.keys():
                if key == 'Other':
                    continue
                if i + n_batch_per_oneclass < music_data[key].n_blocks:
                    indexes = np.arange(i, i + n_batch_per_oneclass)
                    features = music_data[key].get_batch_melspec(indexes)
                    label_num = artist2label[key[:-1]]
                    answers = np.ones(n_batch_per_oneclass, dtype=np.int32) * label_num
                else:
                    end_flag = True
                    break

                if xs is None:
                    xs = features.copy()
                    ts = answers.copy()
                else:
                    xs = np.concatenate((xs, features.copy()), axis=0)
                    ts = np.r_[ts, answers.copy()]

            if end_flag:
                break

            Other_feats = music_data['Other'].get_batch_melspec()
            Other_ts = np.zeros(music_data['Other'].n_data, dtype=np.int32)

            # 連結
            xs = np.concatenate((xs, Other_feats.copy()), axis=0).astype('float32')
            # xs = np.log(xs + 1e-04)
            xs = np.log(xs + 1.0)

            # 教師信号
            ts = np.r_[ts, Other_ts.copy()]

            optimizer.zero_grads()
            loss, accuracy = model(cuda.to_gpu(xs), cuda.to_gpu(ts))  # 入力はログスケール

            loss.backward()
            optimizer.update()

            if i % (n_batch_per_oneclass * 10) == 0:
                print 'epoch:', epoch + 1, ', index:', i, '(/ %d)' % n_blocks
                print 'value:', cuda.to_cpu(loss.data), cuda.to_cpu(accuracy.data)
                print '--------------------'

                if i % (n_batch_per_oneclass * 100) == 0 and i != 0:
                    # モデルパラメータの保存
                    serializers.save_hdf5("trainedmodel/model3-%02d-%05d.model" % (epoch + 1, i), model)
                    print("model saved")


if __name__ == "__main__":
    main()
