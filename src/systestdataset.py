import logging
import os
import glob
import np
from random import shuffle
import librosa


class DataSet:

    def Split(self, train_div, val_div):
        assert train_div + val_div < 1.0
        total_sample_cnt = len(self.x)
        assert total_sample_cnt == len(self.y_)

        val_start = total_sample_cnt * train_div
        test_start = total_sample_cnt * (train_div + val_div)
        logging.debug("val_start: %d, test_start: %d, total_sample_cnt: %d ",
                      val_start, test_start, total_sample_cnt)

        train_set = {'x': self.x[0: val_start],
                     'y_': self.y_[0: val_start]}
        val_set = {'x': self.x[val_start, test_start],
                   'y_': self.y_[val_start, test_start]}
        test_set = {'x': self.x[test_start, total_sample_cnt],
                    'y_': self.y_[test_start, total_sample_cnt]}
        return test_set, val_set, train_set

    def __init__(self, train=0.7, val=0.1):
        self.labels = []
        self.x = None
        self.y_ = None
        self.filename = None
        self.wav = None
        self.spectrum = None
        self.sample_rate = 8000
        self.duration = 3000
        self.train_percentage = train
        self.validate_percentage = val
        self.test_percentage = 1.0 - train - val
        self.train_idx = 0
        self.train_max = 0
        self.validate_idx = 0
        self.validate_max = 0
        self.test_idx = 0
        self.test_max = 0
        assert self.train_percentage > 0.0 and self.train_percentage < 1.0
        assert self.validate_percentage > 0.0 and \
               self.validate_percentage < 1.0
        assert self.test_percentage > 0.0 and self.test_percentage < 1.0
        assert self.train_percentage + \
               self.validate_percentage + \
               self.test_percentage == 1.0

#    def Set(self, samples, classes, audio, spectrum):
#        self.lables = []
#        self.x = samples
#        self.y_ = classes
#        self.file
#        self.wav = audio
#        self.spectrum = spectrum

    def Load(self, wav_dir, npy_dir):
        # use wav dir as input
        load_dir = os.path.join(wav_dir, "*")
        logging.info("loading from directory %s", load_dir)
        files = glob.glob(load_dir)
        if len(files) == 0:
            logging.warn("Cannot find any file in %s", wav_dir)
            return

        shuffle(files)

        x_as_list = []
        y_as_list = []
        wav_as_list = []
        spectrum_as_list = []

        expected_shape = None
        correct_classes = []
        for f in files:
            logging.debug("Loading file %s", f)

            # correct class is not one-hot encoding at this point. will be
            # converted later
            correct_classes.append(self.__CalculateLabel(f))

            base_name = os.path.basename(f).split('.')[0]
            logging.debug("file name: %s", base_name)
            wav, _ = librosa.load(base_name + ".npy",
                                  sr=self.sample_rate,
                                  duration=self.duration)
            spectrum = np.load(os.path.join(np.dir, base_name + '.npy'))

            # @todo check what we should use as x
            x = spectrum.flatten()

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
            wav_as_list.append(wav)
            spectrum_as_list.append(spectrum)
            self.filename.append(base_name)

        # covert y to one hot encoding
        for correct_class in correct_classes:
            y_as_list.append(self.__OneHotEncoding(correct_class))

        assert len(x_as_list) == len(y_as_list)
        assert len(y_as_list) == len(wav_as_list)
        assert len(wav_as_list) == len(spectrum_as_list)
        assert expected_shape[0] == len(x_as_list[0])
        logging.info("%d files loaded, total classes %d",
                     len(x_as_list), len(y_as_list))

        if len(x_as_list) > 0:
            self.x = np.array(x_as_list)
            self.y = np.array(y_as_list)
            self.wav = np.array(wav_as_list)
            self.spectrum = np.array(spectrum_as_list)

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

    def __OneHotEncoding(self, correct_class):
        num_classes = len(self.labels)
        one_hot = np.zeros(num_classes)
        one_hot[correct_class] = 1
        # print("total: ", num_classes, "c: ", correct, "one hot: ", one_hot)

        return one_hot

    def Dump(self):
        logging.debug("x shape: %s", str(self.x.shape))
        logging.debug("y shape: %s", str(self.y.shape))
        logging.debug("wav shape: %s", str(self.wav.shape))
        logging.debug("spectrum shape: %s", str(self.npy.shape))
        logging.debug("filename: %s", str(self.filename))
