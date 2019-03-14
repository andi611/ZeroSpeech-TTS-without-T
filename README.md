# ZeroSpeech 2019: TTS without T
A Pytorch implementation for the [ZeroSpeech 2019 challenge](https://zerospeech.com/2019/).

## Quick Start

### Setup
* Clone this repo: `git clone git@github.com:andi611/ZeroSpeech-TTS-without-T.git`
* CD into this repo: `cd ZeroSpeech-TTS-without-T`

### Installing dependencies

1. Install Python 3.

2. Install the latest version of **[Pytorch](https://pytorch.org/get-started/locally/)** according to your platform. For better
	performance, install with GPU support (CUDA) if viable. This code works with Pytorch 0.4 and later.

### Prepare data

1. **Download the ZeroSpeech dataset.**
	The English dataset:
	```
	wget https://download.zerospeech.com/2019/english.tgz
	tar xvfz english.tgz -C data
	rm -f english.tgz
	```
	The Surprise dataset:
	```
	wget https://download.zerospeech.com/2019/surprise.zip
	unzip surprise.zip -d data
	>> enter the password when prompted for:  9kneopShevtat]
	rm -f surprise.zip
	```
	
	Dataset Backup Link: [Download](https://drive.google.com/drive/folders/19MwNuGO8WbhR4ujmjf9B5k8bHocctSS_?usp=sharing)

2. **After unpacking the dataset into `~/ZeroSpeech-TTS-without-T/data`, data tree should look like this:**
	```
	 |- ZeroSpeech-TTS-without-T
		 |- data
			 |- english
				 |- train
				 	|- unit
				 	|- voice
				 |- test
			|- surprise
				 |- train
				 	|- unit
				 	|- voice
				 |- test
	```

3. **Preprocess the dataset and sample model-ready index files:**
	```
	python3 main.py --preprocess
	```

### Training

1. **Train stage 1 model:**
	```
	python3 main.py --train_ae
	```

	Tunable hyperparameters can be found in [hps/zerospeech.json](hps/zerospeech.json). 
	You can adjust these parameters and setting by editing the file, the default hyperparameters are recommended for this project.

2. **Train stage 2 patcher:**
	```
	python3 main.py --train_g --load_model --load_train_model_name=model.pth-ae-200000
	```

3. **Monitor with Tensorboard** (OPTIONAL)
	```
	tensorboard --logdir 'path to log_dir'
	```


### Testing
1. **Test on 'synthesis.txt' and generate converted resynthesized audio files:**:
	```
	python3 main.py --test --load_test_model_name=model.pth-ae-200000
	```

2. **Test on all the testing speech under `test/` and generate encoding files:**:
	```
	python3 main.py --test_encode --load_test_model_name=model.pth-ae-200000
	```

### Switching between datasets
1. **Simply use add --dataset=surprise to switch to the default alternative set**, all paths are handled automatically if the data tree structure is placed as suggested.
	For example:
	```
	python3 main.py --train_ae --dataset=surprise
	```

