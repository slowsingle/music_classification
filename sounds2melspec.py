import numpy as np
import librosa
import sys, time
from scipy.io.wavfile import read
import glob


class READ_DATASET2(object):
    def __init__(self, wavdir, chunk, length, expected_fs=None):
        wavfiles = glob.glob(wavdir + '*.wav')
        self.wavs = list()
        for wavfile in wavfiles:
            if wavfile.split('.')[-1] != 'wav':
                continue

            fs, all_data = read(wavfile)
            if expected_fs != None and expected_fs != fs:
                print("It has difference between expected_fs and fs")
                raise AssertionError

            if len(all_data.shape) == 2:
                if all_data.shape[1] == 2:
                    # ステレオをモノラルにする
                    all_data = all_data[:, 0]

            all_data = all_data.astype('float64') - 128.0
            all_data /= 128.0

            self.wavs.append(all_data)

        # 無音の追加
        self.wavs.append(np.zeros((chunk * length,), dtype=np.float64))

        # ノイズを読み込む
        self.noise = np.load('noise/noise.npy')  # 8bit 16000Hz
        self.noise = self.noise.astype('float32') / 128.0

        self.sampling_rate = expected_fs
        self.CHUNK = chunk
        self.length = length
        self.n_data = len(self.wavs)

        print("sampling rate is {}".format(self.sampling_rate))
        print("%d sounds was read from %s" % (len(self.wavs), wavdir))


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
            volume = np.random.uniform(low=0.05, high=1.0)
        data_changed_vol = data * volume
        return data_changed_vol

    # 1個データを取り出す(mel-spec)
    def get_one_melspec(self, wav_no):
        all_data = self.wavs[wav_no]

        # データの長さがネットワークの入力より長いか、短いかで処理を変える
        # 長い場合はランダムで切り出し、短い場合はゼロで埋める
        if len(all_data) > self.CHUNK * self.length:
            start_i = np.random.randint(low=0,
                                        high=len(all_data) - self.CHUNK * self.length)
            data = all_data[start_i:(start_i + self.CHUNK * self.length)].copy()
        elif len(all_data) == self.CHUNK * self.length:
            data = all_data.copy()
        else:
            start_i = np.random.randint(low=0,
                                        high=self.CHUNK * self.length - len(all_data))
            data = np.zeros(self.CHUNK * self.length, dtype=np.float64)
            data[start_i:(start_i + len(all_data))] = all_data.copy()

        # データの変形
        data = self._add_noise(data)  # ノイズ追加
        data = self._change_volume(data)  # 音量調節

        melspecs = librosa.feature.melspectrogram(y=data, sr=self.sampling_rate,
                                                  n_fft=2048, n_mels=256)
        return melspecs

    # 複数個データを取り出す(mel-spec)
    def get_batch_melspec(self, wav_nos=None):
        if wav_nos is None:
            wav_nos = np.arange(self.n_data)
        melspecs_dataset = list()
        for n in wav_nos:
            melspecs = self.get_one_melspec(n)
            melspecs_dataset.append(melspecs[np.newaxis, :])
        return np.array(melspecs_dataset)


def main():
    read_dataset = READ_DATASET2(wavdir='wav_16000/',
                                 chunk=1024, length=160, expected_fs=16000)

    for i in range(read_dataset.n_data):
        print(read_dataset.wavs[i].max(), read_dataset.wavs[i].min())

    melspecs_s = read_dataset.get_batch_melspec()
    print(melspecs_s.shape)


if __name__ == "__main__":
    main()
