# ZeroSpeech 2019: TTS without T
A Pytorch implementation for the [ZeroSpeech 2019 challenge](https://zerospeech.com/2019/), the model we used is based on this [paper](https://arxiv.org/pdf/1804.02812.pdf).

## Quick Start

### Setup
* Clone this repo: `git clone git@github.com:andi611/ZeroSpeech-TTS-without-T.git`
* CD into this repo: `cd ZeroSpeech-TTS-without-T`

### Installing dependencies

1. Install Python 3.

2. Install the latest version of **[Pytorch](https://pytorch.org/get-started/locally/)** according to your platform. For better
	performance, install with GPU support (CUDA) if viable. This code works with Pytorch 0.4 and later.

3. Install [requirements](requirements.txt):
	```
	pip3 install -r requirements.txt
	```
	*Warning: you need to install torch depending on your platform. Here list the Pytorch version used when built this project was built.*

### Prepare data

1. **Download the ZeroSpeech dataset.**
	```
	wget https://download.zerospeech.com/2019/english.tgz
	tar xvfz english.tgz -C data
	rm -f english.tgz
	```

2. **After unpacking the dataset into `~/Tacotron-Pytorch/data`, data tree should look like this:**
	```
	 |- ZeroSpeech-TTS-without-T
		 |- data
			 |- english
				 |- train
				 	|- unit
				 	|- voice
				 |- test
	```

3. **Preprocess the dataset and make model-ready meta files:**
	```
	python3 main.py --preprocess
	```

### Training

1. **Train a model:**
	```
	python3 main.py --train
	```

	Restore training from a previous checkpoint:
	```
	python3 train.py --train --load_model
	```

	Tunable hyperparameters can be found in [hps/zerospeech.json](hps/zerospeech.json). 
	
	You can adjust these parameters and setting by editing the file, the default hyperparameters are recommended for this task.

2. **Monitor with Tensorboard** (OPTIONAL)
	```
	tensorboard --logdir 'path to log_dir'
	```

	The trainer dumps audio and alignments every 2000 steps by default. You can find these in `tacotron/ckpt/`.


### Testing
1. **Test with a pre-trained model:**:
	```
	python3 main.py --test
	```


## Acknowledgement
Credits to Ju-chieh Chou for a wonderful Pytorch [implementation](https://github.com/jjery2243542/voice_conversion) of a voice-conversion model, which this work is mainly based on. 

## TODO
* Add more configurable hparams
