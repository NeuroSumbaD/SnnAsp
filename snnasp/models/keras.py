from abc import ABC, abstractmethod

from ..pipeline import TfCorpus, Ready
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
        self.inShape = None
        self.outShape = None

    def fit(self, pipeline, **kwargs):
        if isinstance(pipeline, tf.data.Dataset):
            return self.model.fit(self.prepInput(Ready(pipeline)),**kwargs)
        elif isinstance(pipeline, TfCorpus):
            return self.model.fit(self.prepInput(pipeline),**kwargs)
        else:
            raise TypeError("Expected a type %s or type %s but instead received %s" %(type(tf.data.Dataset), type(pipeline.TfCorpus), type(dataset)))
    
    def evaluate(self, pipeline, **kwargs):
        if isinstance(pipeline, tf.data.Dataset):
            return self.model.evaluate(self.prepInput(Ready(pipeline)),**kwargs)
        elif isinstance(pipeline, TfCorpus):
            return self.model.evaluate(self.prepInput(pipeline),**kwargs)
        else:
            raise TypeError("Expected a type %s or type %s but instead received %s" %(type(tf.data.Dataset), type(pipeline.TfCorpus), type(dataset)))

    def predict(self, pipeline, **kwargs):
        if isinstance(pipeline, tf.data.Dataset):
            return self.model.predict(self.prepInput(Ready(pipeline)),**kwargs)
        elif isinstance(pipeline, TfCorpus):
            return self.model.predict(self.prepInput(pipeline),**kwargs)
        else:
            raise TypeError("Expected a type %s or type %s but instead received %s" %(type(tf.data.Dataset), type(pipeline.TfCorpus), type(dataset)))

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

    def prepInput(self, pipeline):
        '''returns a prepared dataset map object so that pipeline is unaffected
            by the operations required to provide input to the model.
        '''
        return pipeline.Buffer(self.inShape[0], copy=True)

    @abstractmethod
    def save(self):
        pass

    @abstractmethod
    def load(self):
        pass


class ConvAutoencoder(Net):
    def __init__(self, name, numBlocks, numKernels, kernelSize = 4,
                 activation = tf.keras.activations.sigmoid,
                 pooling = tf.keras.layers.AveragePooling1D,
                 poolSize = 4,
                 exists = False,
                 inLength = 256
                 ):
        super().__init__(name)
        self.inShape = (inLength, 8)
        self.outShape = (numKernels, 8)
        if exists:
            print(self.path)
            self.load()

        else:
            self.encoder = tf.keras.Sequential()
            self.decoder = tf.keras.Sequential()

            for index in range(numBlocks):
                self.encoder.add(tf.keras.layers.Conv1D(numKernels, kernelSize, padding="same", activation=activation))
                self.encoder.add(pooling(poolSize, padding="same"))

                self.decoder.add(tf.keras.layers.UpSampling1D(size=poolSize))
                self.decoder.add(tf.keras.layers.Conv1D(numKernels, kernelSize, padding="same", activation=activation))

            self.model.add(self.encoder)
            self.model.add(self.decoder)

    def process(self, stream):
        predictions = self.predict(stream)
        return predictions.reshape(stream.numEntries, -1, self.inShape[1])

    def save(self):
        self.encoder.save(self.path+"-encoder")
        self.decoder.save(self.path+"-decoder")
        self.model.save(self.path)

    def load(self):
        self.model = tf.keras.models.load_model(self.path)
        self.encoder= tf.keras.models.load_model(self.path+"-encoder")
        self.decoder= tf.keras.models.load_model(self.path+"-decoder")

    
if __name__ == "__main__":
    from ..pipeline import LibriSpeech
    import snnasp.evaluate as evaluate
    import matplotlib.pyplot as plt
    
    autoencoder1 = ConvAutoencoder("convAE", 2, 8)
    autoencoder1.compile(optimizer="adam", loss="mse")

    autoencoder2 = ConvAutoencoder("convA2", 2, 8, activation = tf.keras.activations.tanh)
    autoencoder2.compile(optimizer="adam", loss="mse")

    libriSpeech = LibriSpeech("dev-clean-mix")
    libriSpeech.Map(lambda *dataset: (dataset[0],dataset[0]))
    
    print("Training one")
    hist1 = autoencoder1.fit(libriSpeech, epochs = 30)
    # plt.plot(hist.history["loss"])
    # plt.show()

    print("Training other")
    hist2 = autoencoder2.fit(libriSpeech, epochs = 30)
    
    evaluator = evaluate.Testbed(pipeline=libriSpeech,
                                 models=[autoencoder1, autoencoder2],
                                 metrics=[evaluate.SiSNR])

    evaluator.run()
    print(evaluator.results)
    