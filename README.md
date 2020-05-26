# SNN-ASP

  This project explores applications of spiking neural networks for blind source audio separation. Using [ARTRS](https://github.com/MillerLab-UCDavis/RoomSimulation),
  an eight-channel, multiple-talker dataset has been synthesized to simulate input from an eight-microphone linear array.
  
  ## Minimum Dependencies
  
  * Python 3.7
  * TensorFlow 2.0
  * Nengo 3.0
  * Nengo-dl 3.2
  * Numpy 1.17
  * Matplotlib 3.1
  
  ## earlyTfModel
  
  This model is a naive approach to the audio separation task intended mostly for quick development and easy training. The model expects
  input to be buffered so each time step provides the most current audio sample from each channel in the first eight elements of a vector,
  with the second most recent set of audio samples in the next eight elements, third in the third, and fourth in the fourth to create a
  32 element vector.
  
  Model hyper-parameters include:
  * 32-neuron input layer
  * 128-neuron hidden layer
  * 64-neuron hidden layer
  * 1-neuron output layer
  
  A sigmoid activation function was used and the network was trained for 120 epochs with the "Adam" optimizer provided in TensorFlow. 
  A standard mean-squared error (MSE) loss function was chosen for the optimization task.
  
