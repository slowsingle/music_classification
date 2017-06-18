# music_classification
classify some musics (ex. J-POP) (you can use other music)

train.pyを見ると、B'zとかBUMPとかのwavファイルを読み込んで学習させているのがわかるかと思います。\
曲に関しては個人で用意していただきますようお願いいたします。

music2melspec.py : wavファイル1曲を読み込む \
sound2melspec.py : 音楽以外の音、足音や機械音などのwavファイルを読み込む

・実行 \
$ python train.py

・推論は以下のどちらかを実行
predict_by_pyaudio.py : 音声入力にpyaudioを使う人向け（Mac, Windows） \
predict_by_alsaaudio.py : 音声入力にalsaaudioを使う人向け（Linux）

Mac, Ubuntu16.04(Linux)でのみ動作を確認
