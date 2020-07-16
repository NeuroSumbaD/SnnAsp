import nengo
import numpy as np

model = nengo.Network()
with model:
    synDelay = nengo.synapses.LinearFilter([0,1],[1])
    stim = nengo.Node(np.arange(8)/7)
    neurChannels = [stim[index] for index in range(8)]
    outNeur = nengo.Ensemble(1,1, label="OUTPUT")
    for layer in range(3):
        for channel in range(8):
            neur = nengo.Ensemble(5, 1, label=f"{layer}-{channel}")
            nengo.Connection(neurChannels[channel], neur, synapse=synDelay)
            nengo.Connection(neurChannels[channel], outNeur)
            neurChannels[channel] = neur
            
    for channel in range(8):
        nengo.Connection(neurChannels[channel], outNeur)
    outProbe = nengo.Probe(outNeur)