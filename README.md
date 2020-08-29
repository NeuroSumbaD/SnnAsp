# SnnAsp

  This project explores applications of spiking neural networks for blind source audio separation. Using [ARTRS](https://github.com/MillerLab-UCDavis/RoomSimulation),
  an eight-channel, multiple-talker dataset has been synthesized to simulate input from an eight-microphone linear array.
  
  ## Minimum Dependencies
  
  * Python 3.7
  * TensorFlow 2.0
  * Nengo 3.0
  * Nengo-dl 3.2
  * Numpy 1.17
  * Matplotlib 3.1
  
  ## Demos
  This folder contains scripts which demonstrate the basic use of this packages. Each should be run from the parent project directory using the following command:
      
      python ./Demos/Scripts/ScriptName.py [options/flags]

  ### earlyTfModel
  
  This demo contains a model which is a naive approach to the audio separation task intended mostly for quick development and easy training. Peeking at the code will provide an example of how models are expected to be trained and defined using the snnasp package. The *pipeline* module is used to define inputs to the model. A model definition package will be streamlined in the coming weeks.

  This demo provides the following flags:
  
      usage: earlyModel.py [-h] [-t] [-e] [-d] [-m MODEL] [-E EPOCHS]
      optional arguments:
        -h, --help            show this help message and exit
        -t, --train           specifies if training should be attempted
        -e, --evaluate        plots output of single example on each version of the model
        -d, --deep            measures SNR over the full dataset
        -m , --model          specifies name to load/save model
        -E, --epochs          provides number of epochs for training (default 120)
  
  The model expects input to be buffered so each time step provides the most current audio sample from each channel in the first eight elements of a vector, with the second most recent set of audio samples in the next eight elements, third in the third, and fourth in the fourth to create a 32 element vector.
  
  Model hyper-parameters include:
  * 32-neuron input layer
  * 128-neuron hidden layer
  * 64-neuron hidden layer
  * 1-neuron output layer
  
  A sigmoid activation function was used and the network was trained for 120 epochs with the "Adam" optimizer provided in TensorFlow. A standard mean-squared error (MSE) loss function was chosen for the optimization task.
  
  ### convModel & convModel2

  These GUI-based demos are provided as an example of how nengo can apply convolution operations. They may be removed if they are found to be lackluster or unnecessary. At the time of this writing, it appears that a convolution will be less efficient to implement for audio processing than an LSTM-style network simply due to the number of neurons required to represent each dimension accurately along with the additional neurons required to perform the convolution operation.

  In convModel, a convolution is applied across each of the neurons in an ensemble representing the input at the current timestep. In convModel2, the inputs are buffered with samples of audio spread across the dimensions represented by the input ensemble array.