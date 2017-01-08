========================================================
nnerd − an audio classification RNN
========================================================
This project shows that deep learning algorithms can be further generalized to classify and generate audio files as well as people do.

============
Dependencies
============
- FFmpeg
- Python 3

  - Sphinx
  - SciPy
  - Theano
  - NumPy
  - Plotly

=============
More overview
=============
This network is an example of the simple Recurrent Neural Network (RNN) which uses Long Short Term Memory (LSTM) architecture. I used binary as training data to speed up the validation.

=====
Using
=====
Installation
------------

.. code:: shell

    $ git clone https://github.com/awkure/nnerd
    $ cd nnerd
    $ pip install -r requirements.txt

Starting
--------
Edit `config/config.py`, add your audio data inside `data/` directory and then start the network

.. code:: shell

    $ python3 start.py

After training the model would be saved inside `models/` folder you can visualize its history errors and weights (absolutely useless)

.. code:: shell
    
    $ python3 visualize_model.py


TODO:
-----
#. ⬜ Multithreading
#. ⬜ Channels
#. ⬜ More continuously refactoring
#. ☑ Visualization
#. ⬜ Optimizers
