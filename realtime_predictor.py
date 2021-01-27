#-*-coding:utf-8-*-
#!/usr/bin/python
#
# Run sound classifier in realtime.
#
from config import conf
from common import audio_to_melspectrogram
from common import samplewise_normalize_audio_X
from common import geometric_mean_preds
from common import read_audio
from common import print_pyaudio_devices
from common import KerasTFGraph

import pyaudio
import sys
import time
import array
import numpy as np
import queue
from collections import deque
import argparse
import librosa
import pylab as plt

parser = argparse.ArgumentParser(description='Run sound classifier')
parser.add_argument('--input', '-i', default='0', type=int,
                    help='Audio input device index. Set -1 to list devices')
parser.add_argument('--input-file', '-f', default='', type=str,
                    help='If set, predict this audio file.')
#parser.add_argument('--save_file', default='recorded.wav', type=str,
#                    help='File to save samples captured while running.')
parser.add_argument('--model-pb-graph', '-pb', default='', type=str,
                    help='Feed model you want to run, or conf.runtime_weight_file will be used.')
args = parser.parse_args()

# # Capture & pridiction jobs
raw_frames = queue.Queue(maxsize=100)
def callback(in_data, frame_count, time_info, status):
    wave = array.array('h', in_data)
    raw_frames.put(wave, True)
    return (None, pyaudio.paContinue)

def on_predicted(ensembled_pred):
    result = np.argmax(ensembled_pred)
    print(conf.labels[result],ensembled_pred[result], end='')
    if ensembled_pred[result]<0.75:
        print('\t maybe')
    else:
        print()
    return conf.labels[result], ensembled_pred[result]

raw_audio_buffer = []
pred_queue = deque(maxlen=conf.pred_ensembles)
def main_process(model, on_predicted):
    # Pool audio data
    global raw_audio_buffer
    fail = []
    hard = []
    preds = []
    while not raw_frames.empty():
        raw_audio_buffer.extend(raw_frames.get())
        if len(raw_audio_buffer) >= conf.mels_convert_samples: break
    if len(raw_audio_buffer) < conf.mels_convert_samples: return [],[],[]
    # Convert to log mel-spectrogram
    audio_to_convert = np.array(raw_audio_buffer[:conf.mels_convert_samples]) / 32767
    raw_audio_buffer = raw_audio_buffer[conf.mels_onestep_samples:]
    mels = audio_to_melspectrogram(conf, audio_to_convert)
    # Predict, ensemble
    X = []
    for i in range(conf.rt_process_count):
        cur = int(i * conf.dims[1] / conf.rt_oversamples)
        X.append(mels[:, cur:cur+conf.dims[1], np.newaxis])
    X = np.array(X)
    samplewise_normalize_audio_X(X)
    raw_preds = model.predict(X)
    for raw_pred in raw_preds:
        pred_queue.append(raw_pred)
        ensembled_pred = geometric_mean_preds(np.array([pred for pred in pred_queue]))
        label, pred = on_predicted(ensembled_pred)
        if label == 'car':
            fail.extend(audio_to_convert)
        elif label == 'none' and pred<0.75:
            hard.extend(audio_to_convert)
        preds.append(list(ensembled_pred))
    return fail, hard, preds


# # Main controller
def process_file(model, filename, on_predicted=on_predicted):
    # Feed audio data as if it was recorded in realtime
    fail_audio = []
    hard_audio = []
    Audio = read_audio(conf, filename, trim_long_data=False)
    audio = Audio * 32767
    Preds = []
    ss = time.time()
    if conf.DEBUG:plt.ion()
    while len(audio) > conf.rt_chunk_samples:
        raw_frames.put(audio[:conf.rt_chunk_samples])
        audio = audio[conf.rt_chunk_samples:]
        fail, hard, pred = main_process(model, on_predicted)
        fail_audio.extend(fail)
        hard_audio.extend(hard)
        Preds.extend(pred)
        preds = list(zip(*Preds))
        ee = time.time()
        if conf.DEBUG and  ee-ss>1:
            plt.clf()
            plt.subplot(211)
            y = np.arange(0,len(Audio)/conf.sampling_rate,1/conf.sampling_rate)
            plt.plot(y[:len(Audio)],Audio[:len(y)], label='audio')
            plt.subplot(212)
            y = np.arange(0,len(Audio)/conf.sampling_rate,len(Audio)/len(preds[0])/conf.sampling_rate)
            plt.plot(y[:len(preds[0])], preds[0][:len(y)], label=conf.labels[0])
            plt.pause(0.001)
            plt.legend()
            plt.ioff()
            ss = time.time()
    print('librosing...')
    try:
        if len(fail_audio)>0:librosa.output.write_wav('audio_fail.wav', np.array(fail_audio), conf.sampling_rate)
        if len(hard_audio)>0:librosa.output.write_wav('audio_hard.wav', np.array(hard_audio), conf.sampling_rate)
    except:
        print('fail')
        pass

def my_exit(model):
    model.close()
    sys.exit(0)

def get_model(graph_file):
    model_node = {
        'alexnet': ['import/conv2d_1_input',
                    'import/batch_normalization_1/keras_learning_phase',
                    'import/output0'],
        'mobilenetv2': ['import/input_1',
                        'import/bn_Conv1/keras_learning_phase',
                        'import/output0']
    }
    return KerasTFGraph(
        conf.runtime_model_file if graph_file == '' else graph_file,
        input_name=model_node[conf.model][0],
        keras_learning_phase_name=model_node[conf.model][1],
        output_name=model_node[conf.model][2])


fail_audio = []
hard_audio = []
def run_predictor():
    global fail_audio
    global hard_audio
    model = get_model(args.model_pb_graph)
    # file mode
    if args.input_file != '':
        process_file(model, args.input_file)
        my_exit(model)
    # device list display mode
    if args.input < 0:
        print_pyaudio_devices()
        my_exit(model)
    # normal: realtime mode
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    audio = pyaudio.PyAudio()
    stream = audio.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=conf.sampling_rate,
                input=True,
                input_device_index=args.input,
                frames_per_buffer=conf.rt_chunk_samples,
                start=False,
                stream_callback=callback # uncomment for non_blocking
            )
    # main loop
    stream.start_stream()
    Preds = []
    if conf.DEBUG:plt.ion()
    s = time.time()
    ss = time.time()
    while stream.is_active():
        fail, hard, pred = main_process(model, on_predicted)
        Preds.extend(pred)
        preds = list(zip(*Preds))
        e = time.time()
        if conf.DEBUG and e-s>1:
            plt.clf()
            plt.plot(preds[0], label=conf.labels[0])
            plt.pause(0.001)
            plt.legend()
            plt.ioff()
            s = time.time()
        fail_audio.extend(fail)
        hard_audio.extend(hard)
        time.sleep(0.001)
        ee = time.time()
        if not conf.DEBUG and ee-ss>10:
            print('librosing...')
            try:
                if len(fail_audio)>0:librosa.output.write_wav('audio_fail.wav', np.array(fail_audio), conf.sampling_rate)
                if len(hard_audio)>0:librosa.output.write_wav('audio_hard.wav', np.array(hard_audio), conf.sampling_rate)
            except:
                print('fail')
                pass

            ss = time.time()
    stream.stop_stream()
    stream.close()
    # finish
    audio.terminate()
    my_exit(model)

if __name__ == '__main__':
    # args.input_file = 'audio/neg.wav'
    conf.DEBUG = False
    run_predictor()
