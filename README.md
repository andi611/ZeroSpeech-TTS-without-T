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
	# Go to https://download.zerospeech.com  and accept the licence agreement 
	# to get the password protecting the archive
	unzip surprise.zip -d data
	rm -f surprise.zip
	```

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

## Usage

### Training

1. **Train stage 1 ASR-TTS model:**
	```
	python3 main.py --train_ae
	```
	Tunable hyperparameters can be found in [hps/zerospeech.json](hps/zerospeech.json). 
	You can adjust these parameters and setting by editing the file, the default hyperparameters are recommended for this project.

2. **Train stage 2 TTS patcher:**
	```
	python3 main.py --train_p --load_model --load_train_model_name=model.pth-ae-400000
	```

3. **Train stage 2 TTS patcher with target guided adversarial training:**
	```
	python3 main.py --train_tgat --load_model --load_train_model_name=model.pth-ae-400000
	```

4. **Monitor with Tensorboard** (OPTIONAL)
	```
	tensorboard --logdir='path to log dir'
	or
	python3 -m tensorboard.main --logdir='path to log dir'
	```


### Testing
1. Test on a single speech::
	```
	python3 main.py --test_single --load_test_model_name=model.pth-ae-200000
	```

2. Test on 'synthesis.txt' and **generate resynthesized audio files:**:
	```
	python3 main.py --test --load_test_model_name=model.pth-ae-200000
	```

3. Test on all the testing speech under `test/` and **generate encoding files:**:
	```
	python3 main.py --test_encode --load_test_model_name=model.pth-ae-200000
	```

4. Add **`--enc_only`** if testing with ASR-TTS autoencoder only:
	```
	python3 main.py --test_single --load_test_model_name=model.pth-ae-200000 --enc_only
	python3 main.py --test --load_test_model_name=model.pth-ae-200000 --enc_only
	python3 main.py --test_encode --load_test_model_name=model.pth-ae-200000 --enc_only
	```

### Switching between datasets
1. Simply use add **`--dataset=surprise`** to switch to the default alternative set, all paths are handled automatically if the data tree structure is placed as suggested.
	For example:
	```
	python3 main.py --train_ae --dataset=surprise
	```

### Trained-Models
1. We provide trained models as ckpt files, Donwload Link: [ZeroSpeech_Ckpts]()
2. Reload model for training:
	```
	--load_train_model_name=model.pth-ae-400000-128-multi-6-english
	```
	(`--ckpt_dir=./ckpt_english` or `--ckpt_dir=./ckpt_surprise` by default).
3. 2 ways to load model for testing:
	```
	--load_test_model_name=model.pth-s2-30000-128-multi-6-tg-r-english
	--ckpt_pth=ckpt/model.pth-s2-30000-128-multi-6-tg-r-english
	```


## Note
This code includes all the settings and methods we've tested for this challenge, some of which did not suceess but we did not remove them from our code. However, the previous instructions and default settings are for the method we proposed. By running them one can easily reproduce our results.

