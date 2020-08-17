from abc import ABC, abstractmethod

from ..pipeline import TfCorpus
import tensorflow as tf

class Net(ABC):
    '''A wrapper class for simple, sequential keras models intended
        to serve as the base class for all snnasp models
    '''
    basePath = "./Networks/"
    def __init__(self, name, exists = False):
        self.name = name
        self.path = self.basePath+name
        self.model = tf.keras.Sequential()

    def fit(self, pipeline, **kwargs):
        if isinstance(pipeline, tf.data.Dataset):
            return self.model.fit(pipeline,**kwargs)
        elif isinstance(pipeline, TfCorpus):
            return self.model.fit(pipeline.dataset,**kwargs)
        else:
            raise TypeError(f"Expected a type {type(tf.data.Dataset)} or type {type(pipeline.TfCorpus)} but instead received {type(dataset)}")
    
    def evaluate(self, dataset, **kwargs):
        if isinstance(pipeline, tf.data.Dataset):
            return self.model.fit(pipeline,**kwargs)
        elif isinstance(pipeline, TfCorpus):
            return self.model.fit(pipeline.dataset,**kwargs)
        else:
            raise TypeError(f"Expected a type {type(tf.data.Dataset)} but instead received {type(dataset)}")

    def predict(self, pipeline, **kwargs):
        if isinstance(pipeline, tf.data.Dataset):
            return self.model.fit(pipeline,**kwargs)
        elif isinstance(pipeline, TfCorpus):
            return self.model.fit(pipeline.dataset,**kwargs)
        else:
            raise TypeError(f"Expected a type {type(tf.data.Dataset)} but instead received {type(dataset)}")

    def add(self, layer):
        self.model.add(layer)

    def compile(self, **kwargs):
        self.model.compile(**kwargs)

    @abstractmethod
    def process(self, stream):
        '''applies model inference to an iterable object passed in "stream"
            while also ensuring that the output is 32bit float array bounded
            by -1 and 1.
        '''
        raise NotImplementedError

    @abstractmethod
    def save(self):
        pass

    @abstractmethod
    def load(self):
        pass


class ConvAutoEncoder(Net):
    def __init__(self, name, numLayers, numKernels, kernelSize = 4,
                 activation = tf.keras.activations.sigmoid,
                 pooling = tf.keras.layers.AveragePooling1D,
                 poolSize = 4,
                 exists = False
                 ):
        super().__init__(name)
        if exists:
            self.load()

        else:
            self.encoder = tf.keras.Sequential()
            self.decoder = tf.keras.Sequential()

            for index in range(numLayers):
                self.encoder.add(tf.keras.layers.Conv1D(numKernels, kernelSize, padding="same", activation=activation))
                self.encoder.add(pooling(poolSize, padding="same"))

                self.decoder.add(tf.keras.layers.UpSampling1D(size=poolSize))

            self.model.add(self.encoder)
            self.model.add(self.decoder)

    def process(self, stream):
        return self.model.predict(stream)

    def save(self):
        self.encoder.save(self.path+"-encoder")
        self.decoder.save(self.path+"-decoder")
        self.model.save(self.path)

    def load(self):
        self.model = tf.keras.models.load_model(self.path)

    
if __name__ == "__main__":
    from ..pipeline import LibriSpeech
    
    autoencoder = ConvAutoEncoder("convAE", 2, 8)
    autoencoder.compile(optimizer="adam", loss="mse")

    libriSpeech = LibriSpeech("dev-clean-mix")
    libriSpeech.Map(lambda *dataset: (dataset[0],dataset[0]))
    libriSpeech.Buffer(256)

    # history = autoencoder.fit(libriSpeech, epochs = 15)
    