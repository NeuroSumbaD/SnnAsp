'''This demo provides an example of a model trained purely in
    TensorFlow and converted to an equivalent spiking network.
'''
import glob
import argparse

#Parser code to provide options
parser = argparse.ArgumentParser(description='An example feedforward network with spiking and non-spiking equivalents.')
parser.add_argument("-t", '--train', action="store_true", help='specifies if training should be attempted')
parser.add_argument("-e" ,'--evaluate', action="store_true", help='plots output of single example on each version of the model')
parser.add_argument('-d', '--deep',  action="store_true", help='measures SNR over the full dataset')
parser.add_argument('-m', '--model', type=str, default="earlyTfModel4", help='specifies name to load/save model')
parser.add_argument('-E', '--epochs', type=int, default="120", help='provides number of epochs for training (default 120)')
args = parser.parse_args()
modelPath = "./Networks/"+args.model+".h5"


#suppress verbose warnings
import logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)
import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt
import nengo
import nengo_dl

from snnasp import pipeline


#Initialize data pipeline
libriSpeech = pipeline.LibriSpeech("dev-clean-mix")

#Define custom transformation function
def flatBuffer(tensor, buffSize):
    '''Makes each time step correspond to multiple samples of audio.
        Each time step will be 4 samples long for each channel including
        the "current" input
    '''
    inShape = tensor.shape
    numChannels = inShape[1]
    outArr = np.zeros((inShape[0], numChannels*buffSize))
    for timeStep, frame in enumerate(tensor):
        for buff in range(buffSize):
            if timeStep-buff >= 0:
                outArr[timeStep,buff*numChannels:(buff+1)*numChannels] = tensor[timeStep-buff,:]
    del tensor
    return outArr


# This is not the recommended way to handle tensorflow datasets
#   but this approach was used out of relative ease. Converting
#   flatBuffer to use tensorflow operations would be preferred
#   but I was unable to do so efficiently.
def BufferDataset(dataset):
    '''This is a helper function which helps to transform the
        dataset without preloading all data into memory
    '''
    def generateTuples(buffSize = 4):
        for trace, target in dataset:
            yield flatBuffer(trace.numpy(), buffSize), target
            del trace
    return tf.data.Dataset.from_generator(generateTuples, (tf.float32, tf.float32), output_shapes=((960000, 8*4),(960000,)))
    

libriSpeech.dataset = libriSpeech.dataset.apply(BufferDataset)
libriSpeech.Scale(0.5)
libriSpeech.Shift(0.5)
trainSet, testSet = libriSpeech.Split(portion = 5/6)


#If training flag is specified
if args.train:
    history = []
    print("Preparing model...")
    #Define feedforward model
    inLayer = tf.keras.Input(shape=(32,))
    hidLayer1 = tf.keras.layers.Dense(32*4, activation="sigmoid")(inLayer)
    hidLayer2 = tf.keras.layers.Dense(64, activation="sigmoid")(hidLayer1)
    outLayer = tf.keras.layers.Dense(1, activation="sigmoid")(hidLayer2)

    model = tf.keras.Model(inputs=inLayer, outputs=outLayer)
    model.compile(loss="mse", optimizer="adam")

    print("Beginning training...")
    hist = model.fit(trainSet,
                     epochs=args.epochs, batch_size=960000)
    print("Saving model...")
    model.save(modelPath)
    history.append(hist.history["loss"])
    np.save("./"+args.model+"Loss", np.array(hist.history["loss"]))

    print("Plotting training results")
    plt.figure(figsize=(6,4), dpi=100)
    plt.plot(hist.history["loss"])
    plt.xlabel("Epochs")
    plt.ylabel("MSE")
    plt.title("Loss Curve")
    plt.savefig("./Images/LossCurve")
else: #Otherwise load the pre-trained model from memory
    print("Loading existing model...")
    model = tf.keras.models.load_model(modelPath)

print("Creating nengo nengo model...")
snnConverter = nengo_dl.Converter(model)
snnModel = snnConverter.net

with snnModel:
    outProbe = nengo.Probe(snnModel.ensembles[2])
    snnEnsembleProbe = snnModel.probes[0]
    snnInLayer = [snnConverter.inputs[key] for key in snnConverter.inputs][0]

#If evaluation flag is specified
if args.evaluate:
    print("Loading single example input...")
    singleInput, singleTarget = iter(trainSet).next()
    singleInput = singleInput.numpy()
    singleTarget = singleTarget.numpy().reshape((-1, 1))
    print("Calculating ANN output")
    sigOut = model.predict(singleInput)

    waveforms = {"ANN Model": sigOut,
                 "Mixture": singleInput,
                 "Target": singleTarget
                }

    with nengo_dl.Simulator(snnModel) as snnNet:
        print("Calculating nengo model output...")
        snnData = snnNet.predict({snnInLayer: singleInput[None,:,:]})
        if len(glob.glob("./Networks/"+args.model+"Spiking"))==0:
            snnNet.save_params("./Networks/"+args.model+"Spiking")
        # snnParams = snnNet.get_nengo_params([*snnModel.ensembles, *snnModel.connections])
    snnOut = snnData[snnEnsembleProbe][0]
    waveforms["SNN Model"] =  snnOut

    #SPIKING NETWORK
    #This example does not work. Nengo is currently implementing a
    # non-spiking to spiking converter, but for the time-being there
    # is no official spiking version of the model
    # 
    # print("Creating SPIKING model...")
    # with nengo.Network() as trueSpikeModel:
    #     process = nengo.processes.PresentInput(singleInput, presentation_time=0.05)
    #     #Layers
    #     inNode = nengo.Node(process) #possibly redundant outside of nengo.Simulator
    #     a = nengo.Ensemble(128, 1, **snnParams[0])
    #     b = nengo.Ensemble(64, 1, **snnParams[1])
    #     spikeOut = nengo.Ensemble(1, 1, **snnParams[2])
    #     #Connections
    #     d = nengo.Connection(inNode, a.neurons, **snnParams[3])
    #     e = nengo.Connection(a.neurons, b.neurons, **snnParams[4])
    #     f = nengo.Connection(b.neurons, spikeOut.neurons, **snnParams[5])
    #     spikeProbe = nengo.Probe(spikeOut.neurons)

    # print("Calculating SPIKING output")
    # with nengo_dl.Simulator(trueSpikeModel) as spikeNet:
    #     spikeNet.load_params("./Networks/"+args.model+"Spiking")
    #     spikeData = spikeNet.predict(singleInput[None,:,:])
    # spikeOutput = spikeData[spikeProbe][0]
    # waveforms["Spiking Model"] =  spikeOutput


    timeAxis = np.arange(16000*60)/16000
    for key in waveforms:
        print(f"Saving plot: {key}")
        plt.figure(figsize=(6.5,2), dpi=100)
        currSignal = waveforms[key][:,0]
        plt.plot(timeAxis, currSignal)
        plt.xlabel("time (s)")
        plt.ylabel("Relative Amplitude")
        plt.title(key)
        plt.savefig("./Demos/Images/"+key+".png", bbox_inches='tight')

    print("Done.")


if args.deep:
    #Evaluation procedures. Will be replaced soon with official module
    def measureSNR(estimate, rawTarget):
        estimate = (estimate-0.5)*2
        rawTarget = (rawTarget-0.5)*2
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

    #With mean normalization
    def measureSNRnorm(estimate, rawTarget):
        estimate = (estimate-0.5)*2
        estimate -= np.mean(estimate)

        rawTarget = (rawTarget-0.5)*2
        rawTarget -= np.mean(estimate)
        normalization = np.dot(estimate, rawTarget)/np.dot(rawTarget, rawTarget)
        target = normalization*rawTarget
        noise = estimate-target
        SNR = 10*np.log10(np.dot(target, target)/np.dot(noise, noise))
        return SNR
    def meanSNRnorm(allEstimates, allTargets):
        SNRs = []
        for index, target in enumerate(allTargets):
            SNRs.append(measureSNRnorm(allEstimates[index], target))
        return np.mean(SNRs), np.std(SNRs)
    
    results = {}
    
    def Evaluate(dataset, results, name):
        waveforms = {"mixes": [], "ann": [], "nengo": []}
        targets = []
        #possibly not the most efficient but keeps the dataset from being
        # loaded multiple times. It might be worthwhile to see if it's
        # faster doing batch executions but for some reason it was only
        # returning a single output for me
        for mixture, target in dataset:
            targets.append(target.numpy())
            waveforms["mixes"].append(mixture.numpy()[:,0].reshape(-1))
            print("Calulating ann output")
            waveforms["ann"].append(model.predict(mixture).reshape(-1))
            print("Calulating nengo output")
            with nengo_dl.Simulator(snnModel) as snnNet:
                snnData = snnNet.predict({snnInLayer: mixture.numpy()[None,:,:]})
                waveforms["nengo"].append(snnData[snnEnsembleProbe][0].reshape(-1))

        for key in waveforms:
            print(f"Measuring SNR: {key}")
            mean, stdDev = meanSNR(waveforms[key], targets)
            results[f"{name}: {key} Mean SNR"] = mean
            results[f"{name}: {key} StdDev SNR"] = stdDev

            print(f"Measuring norm SNR: {key}")
            mean, stdDev = meanSNRnorm(waveforms[key], targets)
            results[f"{name}: {key} Normed Mean SNR"] = mean
            results[f"{name}: {key} Normed StdDev SNR"] = stdDev

        return results
        

    blah = Evaluate(trainSet,results, "Training data")
    blah2 = Evaluate(testSet,results, "Testing data")
    print(results)


