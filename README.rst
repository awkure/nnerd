========================================================
nnerd − an audio classification RNN
========================================================
This project shows that deep learning algorithms can be further generalized to classify and generate audio files as well as people do. (Not really)

============
Dependencies
============
- ffmpeg
- python 3

  - Sphinx
  - SciPy
  - Theano
  - NumPy
  - Plotly

=============
More overview
=============
This network is an example of the simple Recurrent Neural Network (RNN) which uses Long Short Term Memory (LSTM) architecture. I used binary output as proceeded training data to speed up the validation. 

=====
Using
=====
Installation
------------

.. code:: shell

    $ git clone https://github.com/awkure/nnerd
    $ cd nnerd
    $ make

Starting
--------
Edit `config/config.py`, then add your audio data inside `data/` directory and then start the network

.. code:: shell

    $ make run

After training the model would be saved inside `models/` folder you can visualize its history errors and weights using plotly.

.. code:: shell
    
    $ make visualization 


TODO:
-----
#. - Multithreading
#. - Audio channels
#. - More continuously refactoring
#. + Visualization
#. - Add optimizers
#. - Documentation
