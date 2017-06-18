import pyaudio
import numpy as np
import chainer
from chainer import Variable, serializers
import chainer.functions as F
import chainer.links as L
import sys, time
import librosa


class MUSIC_NET(chainer.Chain):
    def __init__(self, ):
        super(MUSIC_NET, self).__init__(
            conv1=L.Convolution2D(in_channels=1, out_channels=16,
                                  ksize=(16, 9), stride=4, pad=0),
            conv2=L.Convolution2D(in_channels=16, out_channels=32,
                                  ksize=(5, 3), stride=2, pad=0),
            conv3=L.Convolution2D(in_channels=32, out_channels=64,
                                  ksize=(3, 3), stride=2, pad=0),
            fc4=L.Linear(in_size=64 * 14 * 19, out_size=4096),
            fc5=L.Linear(in_size=4096, out_size=7),
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
        reshape3 = F.dropout(F.reshape(conv3, (-1, 64 * 14 * 19)), ratio=0.5, train=False)
        fc4 = F.dropout(F.relu(self.fc4(reshape3)), ratio=0.5, train=False)
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

    raw_data = np.zeros((CHUNK * length,), dtype='float32')

    model = MUSIC_NET()
    serializers.load_hdf5("hoge.model", model)

    p = pyaudio.PyAudio()

    stream = p.open(format=pyaudio.paInt8,
                    channels=1,
                    rate=expected_fs,
                    frames_per_buffer=CHUNK,
                    input=True)

    while stream.is_active():
        try:
            input_data = stream.read(CHUNK)
            data = np.frombuffer(input_data, dtype='int8')
            data = data.astype('float64') / 128.0

            tmp = raw_data.copy()
            raw_data[:-CHUNK] = tmp[CHUNK:]
            raw_data[-CHUNK:] = data.copy()

            melspecs = librosa.feature.melspectrogram(y=raw_data, sr=expected_fs,
                                                      n_fft=2048, n_mels=256)
            melspecs = melspecs[np.newaxis, np.newaxis, :]

            pred = model.predict(melspecs.astype('float32'))
            print(pred.data)

        except KeyboardInterrupt:
            break

    stream.stop_stream()
    stream.close()
    p.terminate()


if __name__ == "__main__":
    main()
