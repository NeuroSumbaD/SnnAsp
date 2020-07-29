import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt
import glob

import nengo
import nengo_dl

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
    parsed = tf.io.parse_single_example(serialized, sceneFeatureDescription)
    return parsed

serializedDataset = tf.data.TFRecordDataset(fileNames)

sceneDataset = serializedDataset.map(parseSerialized)


inLayer = tf.keras.Input(shape=(32,))
hidLayer1 = tf.keras.layers.Dense(32*4, activation="sigmoid")(inLayer)
hidLayer2 = tf.keras.layers.Dense(64, activation="sigmoid")(hidLayer1)
outLayer = tf.keras.layers.Dense(1, activation="sigmoid")(hidLayer2)

model = tf.keras.Model(inputs=inLayer, outputs=outLayer)
    


import soundfile as sf

mixDataPairs = []
for scene in sceneDataset.take(55):
    mixture = scene["Scene"].numpy().decode("utf8")
    # print(mixture) #Uncomment if the "cannot parse" error is thrown
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
        yield mixture, output

mixDataset = tf.data.Dataset.from_generator(mixGen, args=mixDataPairs, output_types=(tf.float32, tf.float32))


def loadAll(dataset):
    Mixes, Targets = iter(dataset).next()
    Mixes = Mixes.numpy()[None,:,:]
    Targets = Targets.numpy()[None,:,None]
    for mix, targ in dataset.skip(1):
        mix = mix.numpy()[None,:,:]
        targ = targ.numpy()[None,:,None]
        Mixes = np.append(Mixes, mix, axis=0)
        Targets = np.append(Targets, targ, axis=0)
    return Mixes, Targets

trainSet = mixDataset.take(50)
testSet = mixDataset.skip(50)

print("Loading...")
trainingMixes, trainingTargets = loadAll(trainSet)

def convertToAudio(data):
    for frame in data:
        frame -= 0.5
        frame *= 2

def convertToLIF(data):
    for frame in data:
        frame *= 0.5
        frame += 0.5

def bufferData(mixArr, buffSize=4, LIF=True):
    '''Makes each time step corresponding to multiple samples of audio.
        Each time step will be 4 samples long for each channel including
        the "current" input
    '''
    inShape = mixArr.shape
    numChannels = inShape[2]
    outArr = np.zeros((inShape[0], inShape[1], numChannels*4))
    for index, sample in enumerate(mixArr):
        for timeStep, frame in enumerate(sample):
            for buff in range(buffSize):
                if timeStep-buff >= 0:
                    outArr[index, timeStep,buff*numChannels:(buff+1)*numChannels] = mixArr[index, timeStep-buff,:]
        if LIF: convertToLIF(outArr[index])
    return outArr


print("Buffering...")
trainingMixes = bufferData(trainingMixes)
print("Convert targets to LIF")
for sample in trainingTargets:
    convertToLIF(sample)
print("Done.")

history = []
model.compile(loss="mse", optimizer="adam")
if len(glob.glob("./earlyTfModel4.h5"))!=0:
    model = tf.keras.models.load_model("./earlyTfModel4.h5")
else:
    hist = model.fit(np.reshape(trainingMixes, (50*16000*60,32)), np.reshape(trainingTargets, (50*16000*60,1)),
                        epochs=120, batch_size=960000)
    model.save("./earlyTfModel4.h5")
    history.append(hist.history["loss"])
    np.save("./earlyTfLoss4", np.array(hist.history["loss"]))

    plt.figure(figsize=(6,4), dpi=100)
    plt.plot(hist.history["loss"])
    plt.xlabel("Epochs")
    plt.ylabel("MSE")
    plt.title("Loss Curve")
    plt.savefig("./Images/LossCurve")

singleInput = trainingMixes[0]
singleTarget = trainingTargets[0]
sigOut = model.predict(singleInput)

waveforms = {   "ANN Model": sigOut,
                "Mixture": singleInput,
                "Target": singleTarget
            }


snnConverter = nengo_dl.Converter(model)
snnModel = snnConverter.net

with snnModel:
    outProbe = nengo.Probe(snnModel.ensembles[2])
    snnEnsembleProbe = snnModel.probes[0]

snnInLayer = [snnConverter.inputs[key] for key in snnConverter.inputs][0]

with nengo_dl.Simulator(snnModel) as snnNet:
    snnData = snnNet.predict({snnInLayer: singleInput[None,:,:]})
    if len(glob.glob("./earlyTfModel4.h5"))==0:
        snnNet.save_params("./earlyTfModelSpiking")
    snnParams = snnNet.get_nengo_params([*snnModel.ensembles, *snnModel.connections])
snnOut = snnData[snnEnsembleProbe][0]
waveforms["SNN Model"] =  snnOut

#TRUE SPIKING NETWORK
print("ATTEMPTING SPIKING MODEL")
with nengo.Network() as trueSpikeModel:
    process = nengo.processes.PresentInput(singleInput, presentation_time=0.05)
    neuronModel = nengo.LIF(tau_rc=0.2, amplitude=20)
    inNode = nengo.Node(process)
    a = nengo.Ensemble(128, 1, **snnParams[0])
    b = nengo.Ensemble(64, 1, **snnParams[1])
    spikeOut = nengo.Ensemble(1, 1, **snnParams[2])
    d = nengo.Connection(inNode, a.neurons, **snnParams[3])
    e = nengo.Connection(a.neurons, b.neurons, **snnParams[4])
    f = nengo.Connection(b.neurons, spikeOut.neurons, **snnParams[5])
    spikeProbe = nengo.Probe(spikeOut.neurons)

with nengo_dl.Simulator(trueSpikeModel) as spikeNet:
    spikeNet.load_params("./earlyTfModelSpiking")
    spikeData = spikeNet.predict(singleInput[None,:,:])
spikeOutput = spikeData[spikeProbe][0]
waveforms["Spiking Model"] =  spikeOutput



timeAxis = np.arange(16000*60)/16000
for key in waveforms:
    print(f"Saving {key}")
    plt.figure(figsize=(6.5,2), dpi=100)
    currSignal = waveforms[key][:,0]
    plt.plot(timeAxis, currSignal)
    plt.xlabel("time (s)")
    plt.ylabel("Relative Intensity")
    plt.title(key)
    plt.savefig("./Images/"+key+".png", bbox_inches='tight')

print("Done.")


def measureSNR(estimate, rawTarget):
    estimate = (estimate[:,0]-0.5)*2
    rawTarget = (rawTarget[:,0]-0.5)*2
    normalization = np.dot(estimate, rawTarget)/np.dot(rawTarget, rawTarget)
    target = normalization*rawTarget
    noise = estimate-target
    SNR = 10*np.log10(np.dot(target, target)/np.dot(noise, noise))
    return SNR

#ANALYZE ACCURACY
def meanSNR(allEstimates, allTargets):
    SNRs = []
    for index, target in enumerate(allTargets):
        SNRs.append(measureSNR(allEstimates[index], target))
    return np.mean(SNRs), np.std(SNRs)



results = {}
print("Measuring Dataset SNR")
mean, stdDev = meanSNR(trainingMixes, trainingTargets)
results["Training Set Mean SNR"] = mean
results["Training Set StdDev SNR"] = stdDev
print("Results")
print(results)
print("Predicting with ANN...")
estimates = []
for mixture in trainingMixes:
    estimates.append(model.predict(mixture))
print("Measuring ANN SNR")
mean, stdDev = meanSNR(estimates, trainingTargets)
results["ANN Train Mean SNR"] = mean
results["ANN Train StdDev SNR"] = stdDev


print("Loading Test set")
testMixes, testTargets = loadAll(testSet)
print("Buffering...")
testMixes = bufferData(testMixes)
print("Convert targets to LIF")
for sample in testTargets:
    convertToLIF(sample)
print("Done.")

print("Measuring Test set SNR")
mean, stdDev = meanSNR(testMixes, testTargets)
results["Test Set Mean SNR"] = mean
results["Test Set StdDev SNR"] = stdDev
print("Results")
print(results)
print("Predicting with ANN...")
estimates = []
for mixture in testMixes:
    estimates.append(model.predict(mixture))
print("Measuring ANN SNR")
mean, stdDev = meanSNR(estimates, testTargets)
results["ANN Test Mean SNR"] = mean
results["ANN Test StdDev SNR"] = stdDev
