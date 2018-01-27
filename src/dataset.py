import os
import glob
import librosa
import librosa.display
import numpy as np
import logging
from random import shuffle
import matplotlib.pyplot as plt
import common


class DataSet:
    x = None
    y = None
    Labels = []

    def Split(self, train_div, val_div):

        assert train_div + val_div < 1.0
        total_sample_cnt = len(self.x)
        assert total_sample_cnt == len(self.y)

        val_start = total_sample_cnt * train_div
        test_start = total_sample_cnt * (train_div + val_div)

        train_set = {'x': self.x[0: val_start],
                     'y': self.y[0: val_start]}
        val_set = {'x': self.x[val_start, test_start],
                   'y': self.y[val_start, test_start]}
        test_set = {'x': self.x[test_start, total_sample_cnt],
                    'y': self.y[test_start, total_sample_cnt]}
        return test_set, val_set, train_set

    def Set(self, samples, classes):
        self.lables = []
        self.x = samples
        self.y = classes

    def Load(self, path):
        load_dir = os.path.join(path, "*")
        logging.info("loading from directory %s", load_dir)
        files = glob.glob(load_dir)
        if len(files) == 0:
            logging.warn("Cannot find any data in path %s", path)
            return

        # @todo is this needed?
        shuffle(files)

        x_as_list = []
        y_as_list = []

        expected_shape = None
        correct_classes = []
        for f in files:
            logging.debug("Loading file %s", f)

            # correct class is not one-hot encoding at this point. will be
            # converted later
            correct_classes.append(self.__CalculateLabel(f))
            x = self.LoadImpl(f)

            # check if shape is consistent
            if expected_shape is None:
                expected_shape = x.shape
            else:
                if expected_shape != x.shape:
                    logging.warn("incosistent loaded data.  was: %d, now: %d",
                                 expected_shape, x.shape)
                assert x.shape == expected_shape

            # numpy array occupied a continuous block of memory
            # so it is actually faster to convert at once
            x_as_list.append(x)

        # covert y to one hot encoding
        for correct_class in correct_classes:
            y_as_list.append(self.__OneHotEncoding(correct_class))

        assert len(x_as_list) == len(y_as_list)
        assert expected_shape[0] == len(x_as_list[0])
        logging.info("%d files loaded, total classes %d",
                     len(x_as_list), len(y_as_list))

        if len(x_as_list) > 0:
            self.x = np.array(x_as_list)
            self.y = np.array(y_as_list)

    # split the labels, put in labels, and calculate the correct class
    def __CalculateLabel(self, file_name):
        base_name = os.path.basename(file_name)
        token = base_name.split('_')[0]
        logging.debug("assign label %s for file %s", token, base_name)

        idx = -1
        try:
            idx = self.labels.index(token)
        except ValueError:
            pass

        if idx == -1:
            correct_class = len(self.labels)
            self.labels.append(token)
            logging.debug("assign existing label %s for file %s, class: %d",
                          token, base_name, correct_class)
        else:
            correct_class = idx
            logging.debug("adding a new label %s for file %s, class: %d",
                          token, base_name, correct_class)

    def LoadImpl(self, path):
        raise NotImplementedError()
        return None

    def __OneHotEncoding(self, correct_class):
        num_classes = len(self.labels)
        one_hot = np.zeros(num_classes)
        one_hot[correct_class] = 1
        # print("total: ", num_classes, "c: ", correct, "one hot: ", one_hot)

        return one_hot


class WaveDataSet(DataSet):

    def LoadImpl(self, path):
        y, sr = librosa.load(path, 8000)
        logging.debug("wave file %s loaded, sample rate:%d", path, sr)
        return y


class NpyDataSet(DataSet):

    def LoadImpl(self, path):
        y = np.load(path)
        logging.debug("npy data file %s loaded, dimemsion %s",
                      path, str(y.shape))
        return y
#
# conversion
#


# generate spectrogram
def GenerateSpectrogram(src_filename, target_filename, sampling_rate,
                        num_fft=320, hop_len=160, num_mels=80, flatten=True):
    logging.debug("GenerateSpectrogram(), src file: %s, dst file: %s",
                  src_filename, target_filename)
    y, _ = librosa.core.load(src_filename, sr=sampling_rate)
    spectro = librosa.feature.melspectrogram(y, sr=sampling_rate,
                                             n_fft=num_fft,
                                             hop_length=hop_len,
                                             n_mels=num_mels)
    if flatten is True:
        np.save(target_filename, spectro.flatten())
    else:
        np.save(target_filename, spectro)


def NormalizeWav(src_dir, target_dir, save_as_wav):
    normalize_sr = 8000
    if os.path.exists(target_dir):
        assert os.path.isdir(target_dir)
        old_files = glob.glob(os.path.join(target_dir, "*"))
        for f in old_files:
            os.remove(f)
    else:
        os.mkdir(target_dir)

    files = glob.glob(os.path.join(src_dir, "*.wav"))
    for f in files:
        logging.debug("resample %s to 8000 HZ and 3 sec", f)
        y, _ = librosa.load(f, sr=normalize_sr, duration=3.0)
        dst_filename = os.path.join(target_dir,
                                    os.path.splitext(os.path.basename(f))[0])

        if save_as_wav:
            librosa.output.write_wav(dst_filename + ".wav", y, sr=normalize_sr)
        else:
            np.save(dst_filename, y)


def ConvertToSpectrum(src_dir, target_dir):
    normalize_sr = 8000
    if os.path.exists(target_dir):
        assert os.path.isdir(target_dir)
        old_files = glob.glob(os.path.join(target_dir, "*"))
        for f in old_files:
            os.remove(f)
    else:
        os.mkdir(target_dir)

    files = glob.glob(os.path.join(src_dir, "*.wav"))
    for f in files:
        dst_filename = os.path.join(target_dir,
                                    os.path.splitext(os.path.basename(f))[0])
        logging.debug("converting %s to %s", f, dst_filename)
        GenerateSpectrogram(f, dst_filename, normalize_sr, 320, 160, 80, False)


def PlotSpectrumWav(wave_file):
    y, _ = librosa.core.load(wave_file, sr=8000)
    PlotSpectrum(y)


def PlotSpectrumNpy(npy_file):
    y = np.load(npy_file)
    PlotSpectrum(y)


def PlotSpectrum(data, title="no titie", flatten=True):
    specto = librosa.feature.melspectrogram(data, sr=8000, n_fft=320,
                                            hop_length=160, n_mels=80)
    log_specto = librosa.core.logamplitude(specto)
    plt.figure(figsize=(12, 4))
    plt.title(title)
    librosa.display.specshow(log_specto, sr=8000, x_axis='time', y_axis='mel',
                             hop_length=160)
    plt.show()


#
# Tests
#


def TestLoadSample():
    dir_path = os.path.join(common.DataDir(), "systest-prototype-small")
    wave_data_set = WaveDataSet()
    wave_data_set.Load(dir_path)
    PlotSpectrum(wave_data_set.x[0])

    dir_path = os.path.join(common.DataDir(), "systest-prototype-small-npy")
    print(dir_path)
    npy_data_set = NpyDataSet()
    npy_data_set.Load(dir_path)
    PlotSpectrum(npy_data_set.x[0])


def TestConvertWav():
    rootdir = common.RootDir()
    src_path = os.path.join(rootdir, "data-src/systest-prototype-small")
    dst_path = os.path.join(rootdir, "systest-prototype-small")
    NormalizeWav(src_path, dst_path, True)


def TestConvertNpy():
    rootdir = common.RootDir()
    src_path = os.path.join(rootdir, "data-src/systest-prototype-small")
    dst_path = os.path.join(rootdir, "systest-prototype-small-npy")
    NormalizeWav(src_path, dst_path, False)


def TestPlotSpectrumWav():
    dir_path = os.path.join(common.DataDir(), "systest-prototype-small/*.wav")
    files = glob.glob(dir_path)
    logging.debug("plot spectrom wav: %s", files[0])
    PlotSpectrumWav(files[0])


def TestPlotSpectrumNpy():
    datadir = common.DataDir()
    dir_path = os.path.join(datadir, "systest-prototype-small-npy/*.npy")
    files = glob.glob(dir_path)
    logging.debug("plot spectrom npy: %s", files[0])
    PlotSpectrumNpy(files[0])


def TestDataSets():
    TestConvertWav()
    TestConvertNpy()
    TestLoadSample()
