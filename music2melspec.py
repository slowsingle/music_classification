import numpy as np
import librosa
import sys, time
from scipy.io.wavfile import read


class READ_DATASET(object):
    def __init__(self, wavfile, chunk, length, expected_fs=None):
        fs, all_data = read(wavfile)
        if expected_fs != None and expected_fs != fs:
            print("It has difference between expected_fs and fs")
            raise AssertionError

        all_data = all_data.astype('float64') - 128.0
        all_data /= 128.0

        self.all_data = all_data
        self.sampling_rate = fs
        self.CHUNK = chunk
        self.length = length

        # リアルノイズを読み込む
        self.noise = np.load('noise/noise.npy')  # 8bit 16000Hz
        self.noise = self.noise.astype('float32') / 128.0

        # インデックス。初期化時は昇順にしておく
        n_bolcks_all = len(self.all_data) - self.CHUNK * self.length - 1
        self.indexes = np.linspace(100, n_bolcks_all, int(n_bolcks_all / 3.0)).astype(np.int64)
        # 忙しい人用
        #self.indexes = np.linspace(1024, n_bolcks_all, int(n_bolcks_all / 100.0)).astype(np.int64)
        self.n_blocks = len(self.indexes)

        print("sampling rate is {}".format(fs))

    def shuffle_indexes(self):
        self.indexes = np.random.permutation(len(self.indexes))

    # リアルなノイズの追加
    def _add_noise(self, data, scale=None):
        if scale is None:
            scale = np.random.uniform(low=0.001, high=3.0)
        start_i = np.random.randint(low=0, high=len(self.noise) - len(data))
        noise = self.noise[start_i:(start_i + len(data))]
        data_with_noise = data + noise * scale
        return data_with_noise

    # 音量調整
    def _change_volume(self, data, volume=None):
        if volume is None:
            volume = np.random.uniform(low=0.1, high=1.0)
        data_changed_vol = data * volume
        return data_changed_vol


    # 1個データを取り出す(mel-spec)
    def get_one_melspec(self, index):
        start_i = self.indexes[index]
        data = self.all_data[start_i:(start_i + self.CHUNK * self.length)].copy()

        # データの変形
        data = self._add_noise(data)  # ノイズ追加
        data = self._change_volume(data)  # 音量調節

        melspecs = librosa.feature.melspectrogram(y=data, sr=self.sampling_rate,
                                                  n_fft=2048, n_mels=256)
        return melspecs

    # 複数個データを取り出す(mel-spec)
    def get_batch_melspec(self, indexes):
        melspecs_dataset = list()
        for index in indexes:
            melspecs = self.get_one_melspec(index)
            melspecs_dataset.append(melspecs[np.newaxis, :])
        return np.array(melspecs_dataset)


def main():
    read_dataset = READ_DATASET(wavfile='music/8bit-16000Hz/CatchTheMoment.wav',
                                chunk=1024, length=160, expected_fs=16000)
    read_dataset.shuffle_indexes()

    melspecs_s = read_dataset.get_batch_melspec(np.arange(10))
    print(read_dataset.indexes[:100])
    print(read_dataset.all_data.max(), read_dataset.all_data.min())
    print(melspecs_s.shape, read_dataset.n_blocks, len(read_dataset.all_data))
    print(read_dataset.all_data.dtype)
    print(melspecs_s.max(), melspecs_s.min())


if __name__ == "__main__":
    main()
