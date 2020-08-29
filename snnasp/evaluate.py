'''Provides the standard evaluation metrics for the output of
    any model. The module will also be available from the command
    line to allow evaluation on existing models.
'''
from time import time as currentTime

import numpy as np
np.seterr(invalid='warn')


class Testbed:
    def __init__(self, pipeline=None, models=[], metrics=[]):
        self.pipeline = pipeline
        self.models = {}
        self.metrics = {}
        self.results = {} #for mean and std dev table
        self.data = {} #for storing results in their entirety

        for model in models:
            self.models[model.name] = model

        for metric in metrics:
            self.metrics[metric.name] = metric

    def addModel(self, model):
        self.models[model.name] = model

    def addMetric(self, metric):
        self.metrics[metric.name] = metric

    def run(self, pipeline=None):
        #bring into local scope and save most recently used pipeline
        if pipeline == None:
            pipeline = self.pipeline
            self.pipeline = pipeline if pipeline is not None else self.pipeline
            pipelineName = pipeline.path.split("/")[-1]
        else:
            raise ValueError("A pipeline must be given")

        print("Loading dataset")
        ins = pipeline.dataset.map(lambda first, second: first)
        targets = pipeline.dataset.map(lambda first, second: second)
        
        ins = np.array(list(ins.as_numpy_iterator()))
        targets = np.array(list(targets.as_numpy_iterator()))


        for model in self.models:
            model = self.models[model]

            print(f"Running {model.name}...")
            timeStart = currentTime()
            modelOut = model.process(pipeline)
            inferenceTime = currentTime() - timeStart

            modelResults = {"outs": modelOut}
            modelSummary = {}

            if pipeline.numEntries is not None:
                modelSummary["Inference time"] = f"{inferenceTime/pipeline.numEntries} s/entry"
            else:
                modelSummary["Inference time"] = f"{inferenceTime} s"
            print("Done.")
            for metric in self.metrics:
                metric = self.metrics[metric]
                print(f"Evaluating {model.name}-{metric.name}...")
                mean, stdDev, full = metric.call(modelOut, targets)
                modelSummary[metric.name] = f"{mean}+/-{stdDev}"
                modelResults[metric.name] = full
            self.results[model.name] = modelSummary
            self.data[model.name]  = modelResults

        baselineSummary = {"Inference time": "N/A"}
        baselineResults = {}
        for metric in self.metrics:
            metric = self.metrics[metric]
            if metric.baseline:
                print(f"Running baseline metric: {metric.name}")
                mean, stdDev, full = metric.call(ins, targets)
                baselineSummary[metric.name] = f"{mean} +/- {stdDev}"
                baselineResults[metric.name] = full
            else:
                baselineSummary[metric.name] = "N/A"
                baselineResults[metric.name] = None

        self.results[f"dataset--{pipelineName}"] = baselineSummary
        self.data[f"dataset--{pipelineName}"] = baselineResults

        
        del pipeline, ins, targets
        return self.results

    def save(self):
        title = f"{self.pipeline.path.split('/')[-1]}"
        path = "./Networks/Results/"+title+"--summary.csv"
        print(f"Saving result summary to:", path)
        colHeads = ["Inference time"]+list(self.metrics.keys())
        rowNames = np.array(list(self.results.keys()))
        summary = np.array([[self.results[data][metric] for metric in colHeads] for data in self.results])
        colHeads = ["model names"] + colHeads
        strings = np.append(rowNames.reshape(-1,1), summary.astype('str'), axis=1)
        np.savetxt(path, strings, fmt="%s", delimiter=",", header=",".join(colHeads))

        fullData = [filter(None,[[f"{group}--{metric}"]+list(self.data[group][metric]) if self.data[group][metric] is not None else None for metric in self.metrics]) for group in self.data]
        fullData = [line for sublist in fullData for line in sublist]
        path = "./Networks/Results/"+title+"--full.csv"
        print("Saving full results to:", path)
        np.savetxt(path, np.array(fullData), fmt="%s", delimiter=",", header="Measurement, list of results")

        print("Done saving results.")


class Metric:
    def __init__(self, func, name, baseline=False):
        '''The baseline parameter indicates that the metric should be tested
            on the input output pairs in the dataset as well
        '''
        self.func = func
        self.name = name
        self.baseline = baseline
        self.lastCall = None

    def call(self, ins, outs):
        self.lastCall = []
        for inData, outData in zip(ins,outs):
            self.lastCall.append(self.func(inData, outData))

        self.lastCall = np.array(self.lastCall)
        mean = np.mean(self.lastCall)
        stdDev = np.std(self.lastCall)

        return mean, stdDev, self.lastCall


class SiSNR(Metric):
    '''Scale-invariant signal-to-noise ratio for time-series data. Multiple
        channel inputs are flattened before tested.
    '''
    name = "SiSNR"

    def SiSNRfn(self, entry, target):
        entry = entry.flatten()
        target = target.flatten()
        normFactor = np.dot(entry, target)/np.dot(target, target)
        normedTarget = normFactor*target
        noise = entry-normedTarget
        SNR = 10*np.log10(np.dot(normedTarget, normedTarget)/np.dot(noise, noise))  
        return SNR
    
    def __init__(self, baseline = True):
        self.func = self.SiSNRfn
        self.lastCall = None
        self.baseline = baseline #set false for autoencoders 


#---------------Development Code---------------#
if __name__ == "__main__":
    from .pipeline import LibriSpeech
    import snnasp.models.keras as keras

    autoencoder = keras.ConvAutoencoder("convAE", 2, 8, exists=True)
    autoencoder.compile(optimizer="adam", loss="mse")

    libriSpeech = LibriSpeech("dev-clean-mix")
    libriSpeech.Map(lambda *dataset: (dataset[0],dataset[0]))

    evaluator = Testbed(pipeline=libriSpeech,
                        models=[autoencoder],
                        metrics=[SiSNR()])

    result = evaluator.run()
    evaluator.save()