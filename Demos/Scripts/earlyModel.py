import glob

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


#Define simple feedforward model
inLayer = tf.keras.Input(shape=(32,))
hidLayer1 = tf.keras.layers.Dense(32*4, activation="sigmoid")(inLayer)
hidLayer2 = tf.keras.layers.Dense(64, activation="sigmoid")(hidLayer1)
outLayer = tf.keras.layers.Dense(1, activation="sigmoid")(hidLayer2)

model = tf.keras.Model(inputs=inLayer, outputs=outLayer)

#prepare data pipeline
libriSpeech = pipeline.LibriSpeech("dev-clean-mix")
libriSpeech.Scale(0.5)
libriSpeech.Shift(0.5)

def flatBuffer(tensor):
    '''Makes each time step corresponding to multiple samples of audio.
        Each time step will be 4 samples long for each channel including
        the "current" input
    '''
    buffSize=4
    inShape = tensor.shape
    numChannels = inShape[1]
    outArr = tf.unstack(tf.zeros((inShape[0], buffSize, numChannels), "float32"))
    timeStep = 0
    for frames in outArr:
        frame = tf.unstack(frames)
        buff = 0
        for block in frame:
            if timeStep-buff >= 0:
                frame[buff] = tensor[timeStep-buff]
            buff += 1
        outArr[timeStep] = tf.stack(frame)
        del frame
        timeStep += 1
    return tf.stack(outArr)

libriSpeech.MapChannel(flatBuffer, channels=[True, False])

trainSet, testSet = libriSpeech.Split(portion = 5/6) 







# history = []
# model.compile(loss="mse", optimizer="adam")
# if len(glob.glob("./earlyTfModel4.h5"))!=0:
#     model = tf.keras.models.load_model("./earlyTfModel4.h5")
# else:
#     hist = model.fit(np.reshape(trainingMixes, (50*16000*60,32)), np.reshape(trainingTargets, (50*16000*60,1)),
#                         epochs=120, batch_size=960000)
#     model.save("./earlyTfModel4.h5")
#     history.append(hist.history["loss"])
#     np.save("./earlyTfLoss4", np.array(hist.history["loss"]))

#     plt.figure(figsize=(6,4), dpi=100)
#     plt.plot(hist.history["loss"])
#     plt.xlabel("Epochs")
#     plt.ylabel("MSE")
#     plt.title("Loss Curve")
#     plt.savefig("./Images/LossCurve")

# singleInput = trainingMixes[0]
# singleTarget = trainingTargets[0]
# sigOut = model.predict(singleInput)

# waveforms = {   "ANN Model": sigOut,
#                 "Mixture": singleInput,
#                 "Target": singleTarget
#             }


# snnConverter = nengo_dl.Converter(model)
# snnModel = snnConverter.net

# with snnModel:
#     outProbe = nengo.Probe(snnModel.ensembles[2])
#     snnEnsembleProbe = snnModel.probes[0]

# snnInLayer = [snnConverter.inputs[key] for key in snnConverter.inputs][0]

# with nengo_dl.Simulator(snnModel) as snnNet:
#     snnData = snnNet.predict({snnInLayer: singleInput[None,:,:]})
#     if len(glob.glob("./earlyTfModel4.h5"))==0:
#         snnNet.save_params("./earlyTfModelSpiking")
#     snnParams = snnNet.get_nengo_params([*snnModel.ensembles, *snnModel.connections])
# snnOut = snnData[snnEnsembleProbe][0]
# waveforms["SNN Model"] =  snnOut

# #TRUE SPIKING NETWORK
# print("ATTEMPTING SPIKING MODEL")
# with nengo.Network() as trueSpikeModel:
#     process = nengo.processes.PresentInput(singleInput, presentation_time=0.05)
#     neuronModel = nengo.LIF(tau_rc=0.2, amplitude=20)
#     inNode = nengo.Node(process)
#     a = nengo.Ensemble(128, 1, **snnParams[0])
#     b = nengo.Ensemble(64, 1, **snnParams[1])
#     spikeOut = nengo.Ensemble(1, 1, **snnParams[2])
#     d = nengo.Connection(inNode, a.neurons, **snnParams[3])
#     e = nengo.Connection(a.neurons, b.neurons, **snnParams[4])
#     f = nengo.Connection(b.neurons, spikeOut.neurons, **snnParams[5])
#     spikeProbe = nengo.Probe(spikeOut.neurons)

# with nengo_dl.Simulator(trueSpikeModel) as spikeNet:
#     spikeNet.load_params("./earlyTfModelSpiking")
#     spikeData = spikeNet.predict(singleInput[None,:,:])
# spikeOutput = spikeData[spikeProbe][0]
# waveforms["Spiking Model"] =  spikeOutput



# timeAxis = np.arange(16000*60)/16000
# for key in waveforms:
#     print(f"Saving {key}")
#     plt.figure(figsize=(6.5,2), dpi=100)
#     currSignal = waveforms[key][:,0]
#     plt.plot(timeAxis, currSignal)
#     plt.xlabel("time (s)")
#     plt.ylabel("Relative Intensity")
#     plt.title(key)
#     plt.savefig("./Images/"+key+".png", bbox_inches='tight')

# print("Done.")


# def measureSNR(estimate, rawTarget):
#     estimate = (estimate[:,0]-0.5)*2
#     rawTarget = (rawTarget[:,0]-0.5)*2
#     normalization = np.dot(estimate, rawTarget)/np.dot(rawTarget, rawTarget)
#     target = normalization*rawTarget
#     noise = estimate-target
#     SNR = 10*np.log10(np.dot(target, target)/np.dot(noise, noise))
#     return SNR

# #ANALYZE ACCURACY
# def meanSNR(allEstimates, allTargets):
#     SNRs = []
#     for index, target in enumerate(allTargets):
#         SNRs.append(measureSNR(allEstimates[index], target))
#     return np.mean(SNRs), np.std(SNRs)



# results = {}
# print("Measuring Dataset SNR")
# mean, stdDev = meanSNR(trainingMixes, trainingTargets)
# results["Training Set Mean SNR"] = mean
# results["Training Set StdDev SNR"] = stdDev
# print("Results")
# print(results)
# print("Predicting with ANN...")
# estimates = []
# for mixture in trainingMixes:
#     estimates.append(model.predict(mixture))
# print("Measuring ANN SNR")
# mean, stdDev = meanSNR(estimates, trainingTargets)
# results["ANN Train Mean SNR"] = mean
# results["ANN Train StdDev SNR"] = stdDev


# print("Loading Test set")
# testMixes, testTargets = loadAll(testSet)
# print("Buffering...")
# testMixes = bufferData(testMixes)
# print("Convert targets to LIF")
# for sample in testTargets:
#     convertToLIF(sample)
# print("Done.")

# print("Measuring Test set SNR")
# mean, stdDev = meanSNR(testMixes, testTargets)
# results["Test Set Mean SNR"] = mean
# results["Test Set StdDev SNR"] = stdDev
# print("Results")
# print(results)
# print("Predicting with ANN...")
# estimates = []
# for mixture in testMixes:
#     estimates.append(model.predict(mixture))
# print("Measuring ANN SNR")
# mean, stdDev = meanSNR(estimates, testTargets)
# results["ANN Test Mean SNR"] = mean
# results["ANN Test StdDev SNR"] = stdDev
