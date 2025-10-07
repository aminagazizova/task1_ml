import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import csv
import sys
import soundfile as sf

model = hub.load('https://tfhub.dev/google/yamnet/1')

def classify_audio(file_path):
    wav_data, sr = sf.read(file_path, dtype=np.int16)

    if len(wav_data.shape) > 1:
        wav_data = np.mean(wav_data, axis=1)

    waveform = wav_data / 32768.0

    scores, embeddings, spectrogram = model(waveform)
    scores_np = scores.numpy().mean(axis=0)

    class_map_path = tf.keras.utils.get_file(
        'yamnet_class_map.csv',
        'https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv'
    )
    class_names = [line.strip().split(',')[2] for line in open(class_map_path).readlines()[1:]]

    top_indices = np.argsort(scores_np)[::-1][:5]
    results = [(class_names[i], float(scores_np[i])) for i in top_indices]

    print(f"\n Результаты для файла: {file_path}")
    for label, score in results:
        print(f"{label}: {score:.3f}")

    with open('results.csv', mode='a', newline='') as f:
        writer = csv.writer(f)
        for label, score in results:
            writer.writerow([file_path, label, score])

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Использование: python audio_classification_yamnet.py <audio_file.wav>")
    else:
        classify_audio(sys.argv[1])
