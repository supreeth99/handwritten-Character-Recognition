# Handwritten Text Recognition with Keras

Video Demo : https://youtu.be/m6soGLvyPoY

Handwritten Digit Recognition system implemented using Keras and CNN. The model takes **images of single digit throught the canvas shown on the browser as input** and **outputs the recognized Digit**.
3/4 of the digits from the validation-set are correctly recognized, and the character error rate is around 10%.


## Run demo

* Clone the repository. 
* If you want to retrain the model and start from the beginning, for the ##Run Training Steps. 
* Go to the `src` directory 
* Run `conda install` to download all the dependencies. 
* Run inference code:
  * Execute `app12.py` to run the application. A flask Application will run on your localhost. 


### Run training

* Delete files from `model_v1.h5` directory if you want to train from scratch
* Go to the `src` directory and execute `Model_generator.py`
* The IAM dataset is split into 90% training data and 10% validation data    
* Training stops after a fixed number of epochs without improvement.


### Fast image loading
Loading and decoding the png image files from the disk is the bottleneck even when using only a small GPU.
The database LMDB is used to speed up image loading:
* Go to the `src` directory and run `create_lmdb.py --data_dir path/to/iam` with the IAM data directory specified
* A subfolder `lmdb` is created in the IAM data directory containing the LMDB files
* When training the model, add the command line option `--fast`

The dataset should be located on an SSD drive.
Using the `--fast` option and a GTX 1050 Ti training on single words takes around 3h with a batch size of 500.
Training on text lines takes a bit longer.


## References
* [Build a Handwritten Text Recognition System using TensorFlow](https://towardsdatascience.com/2326a3487cd5)
* [Scheidl - Handwritten Text Recognition in Historical Documents](https://repositum.tuwien.ac.at/obvutwhs/download/pdf/2874742)
* [Scheidl - Word Beam Search: A Connectionist Temporal Classification Decoding Algorithm](https://repositum.tuwien.ac.at/obvutwoa/download/pdf/2774578)

