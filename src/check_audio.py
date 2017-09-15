import os


class SampleType:
    Wav, Npy = range(2)


class Sample:
    """ Sample class """
    def __init__(self):
        self.path = ""
        self.name = ""
        self.wav_data = None
        self.npy_data = None
        self.label = -1
        self.sample_rate = -1
        self.labels = {}

    def Load(path):
        basename = os.path.basename(path)
        tokens = basename.split('.')
        print(tokens)
        assert len(tokens) == 2
        if tokens[1] == 'npy':
            LoadNpySample(tokens[0])
        else:
            LoadWavSample(tokens[0])
        return

    def __CalculateLabel(file_name):
        token = os.path.basename(file_name).split('_')[0]
        correct_class = -1
        if token in self.labels:
            correct_class = self.LoadNpySample(file_name)
        else:
            correct_class = len(labels)
            labels.append(token)
        return token

    def LoadNpySample(name, label):
        return

    def LoadWavSample(name, label):
        return

def main():
    print("test\n")
    return


if __name__ == "__main__":
    main()
