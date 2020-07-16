import tensorflow as tf
import numpy as np

import nengo
import nengo_dl

import matplotlib.pyplot as plt
import glob


fileNames = glob.glob("../Dataset/LibriSpeech/dev-clean-mix/*.tfrecord")
sigLength = 16000*60

sceneFeatureDescription = {
    'Scene': tf.io.FixedLenFeature([], tf.string),
    'sampleRate': tf.io.FixedLenFeature([1], tf.int64),
    'numRecs': tf.io.FixedLenFeature([1], tf.int64),
    'micArrOrigin': tf.io.FixedLenFeature([3], tf.float32),
    'micSteer': tf.io.FixedLenFeature([3], tf.float32),

    'speaker1': tf.io.FixedLenFeature([], tf.string),
    'location1': tf.io.FixedLenFeature([sigLength], tf.float32),
    'signal1': tf.io.FixedLenFeature([3], tf.float32),

    'speaker2': tf.io.FixedLenFeature([], tf.string),
    'location2': tf.io.FixedLenFeature([sigLength], tf.float32),
    'signal2': tf.io.FixedLenFeature([3], tf.float32),
    
    'speaker3': tf.io.FixedLenFeature([], tf.string),
    'location3': tf.io.FixedLenFeature([sigLength], tf.float32),
    'signal3': tf.io.FixedLenFeature([3], tf.float32),

    'traceData': tf.io.FixedLenFeature([sigLength*8*4+19], tf.int64), #A stupid mistake led to this
}

def parseSerialized(serialized):
    return tf.io.parse_single_example(serialized, sceneFeatureDescription)

serializedDataset = tf.data.TFRecordDataset(fileNames)

sceneDataset = serializedDataset.map(parseSerialized)

#Use integrators to simulate a form of memory, and then use fully connected neurons to learn the best pattern
with nengo.Network() as model:
    #input layer limited to one samples from each of the 8 channels
    micInputs = nengo.Node(np.zeros(8))
    #layers of integrating neurons
    integrator1 = nengo.networks.Integrator(0.5, 240, 8)
    nengo.Connection(micInputs, integrator1.input)
    integrator2 = nengo.networks.Integrator(0.5, 240, 8)
    nengo.Connection(integrator1.ensemble, integrator2.input)
    integrator3 = nengo.networks.Integrator(0.5, 240, 8)
    nengo.Connection(integrator2.ensemble, integrator3.input)
    #Fully connected layers
    chan1 = nengo.Ensemble(40, 1)
    nengo.Connection(micInputs[0], chan1)
    nengo.Connection(integrator1.ensemble[0], chan1)
    nengo.Connection(integrator2.ensemble[0], chan1)
    nengo.Connection(integrator3.ensemble[0], chan1)
    chan2 = nengo.Ensemble(40, 1)
    nengo.Connection(micInputs[0], chan2)
    nengo.Connection(integrator1.ensemble[1], chan2)
    nengo.Connection(integrator2.ensemble[1], chan2)
    nengo.Connection(integrator3.ensemble[1], chan2)
    chan3 = nengo.Ensemble(40, 1)
    nengo.Connection(micInputs[0], chan3)
    nengo.Connection(integrator1.ensemble[2], chan3)
    nengo.Connection(integrator2.ensemble[2], chan3)
    nengo.Connection(integrator3.ensemble[2], chan3)
    chan4 = nengo.Ensemble(40, 1)
    nengo.Connection(micInputs[0], chan4)
    nengo.Connection(integrator1.ensemble[3], chan4)
    nengo.Connection(integrator2.ensemble[3], chan4)
    nengo.Connection(integrator3.ensemble[3], chan4)
    chan5 = nengo.Ensemble(40, 1)
    nengo.Connection(micInputs[0], chan5)
    nengo.Connection(integrator1.ensemble[4], chan5)
    nengo.Connection(integrator2.ensemble[4], chan5)
    nengo.Connection(integrator3.ensemble[4], chan5)
    chan6 = nengo.Ensemble(40, 1)
    nengo.Connection(micInputs[0], chan6)
    nengo.Connection(integrator1.ensemble[5], chan6)
    nengo.Connection(integrator2.ensemble[5], chan6)
    nengo.Connection(integrator3.ensemble[5], chan6)
    chan7 = nengo.Ensemble(40, 1)
    nengo.Connection(micInputs[0], chan7)
    nengo.Connection(integrator1.ensemble[6], chan7)
    nengo.Connection(integrator2.ensemble[6], chan7)
    nengo.Connection(integrator3.ensemble[6], chan7)
    chan8 = nengo.Ensemble(40, 1)
    nengo.Connection(micInputs[0], chan8)
    nengo.Connection(integrator1.ensemble[7], chan8)
    nengo.Connection(integrator2.ensemble[7], chan8)
    nengo.Connection(integrator3.ensemble[7], chan8)
    #Fully connected output layer
    outNeur = nengo.Ensemble(360, 1)
    nengo.Connection(chan1, outNeur)
    nengo.Connection(chan2, outNeur)
    nengo.Connection(chan3, outNeur)
    nengo.Connection(chan4, outNeur)
    nengo.Connection(chan5, outNeur)
    nengo.Connection(chan6, outNeur)
    nengo.Connection(chan7, outNeur)
    nengo.Connection(chan8, outNeur)
    
    outProbe = nengo.Probe(outNeur)


import soundfile as sf

mixDataPairs = []
for scene in sceneDataset.take(30):
    mixture = scene["Scene"].numpy().decode("utf8")
    speaker1 = scene["speaker1"].numpy().decode("utf8")
    location1 = scene["signal1"].numpy()
    micArrLoc = scene["micArrOrigin"].numpy()
    targDist = np.sum(np.square(location1 - micArrLoc))
    sampDelay = int(round(targDist/343*16000))
    tup = (mixture, speaker1, str(sampDelay))
    mixDataPairs.append(tup)

def mixGen(*pairs):
    for (mixData, target, sampDelay) in pairs:
        mixture = sf.read(mixData)[0]
        output = sf.read("../Dataset/LibriSpeech/dev-clean-mix/"+target.decode("utf8")+".wav")[0]
        output = np.pad(output, pad_width=(int(sampDelay),0))[:16000*60]
        print(mixture.dtype, output.dtype)
        yield mixture, output

mixDataset = tf.data.Dataset.from_generator(mixGen, args=mixDataPairs, output_types=(tf.float32, tf.float32))

sampleIn, sampleOut = iter(mixDataset).next()
sampleIn = sampleIn.numpy()[None,:,:]
sampleOut = sampleOut.numpy()[None,:,None]
print(sampleIn.shape)
print(sampleOut.shape)

with nengo_dl.Simulator(model, seed=0) as earlyModel:
    earlyModel.compile(optimizer="adam", loss=tf.losses.mean_squared_error)
    earlyModel.fit(sampleIn, sampleOut)


plt.plot(earlyModel.trange(), earlyModel.data[outProbe])
plt.show()
