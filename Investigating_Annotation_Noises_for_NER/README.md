# Investigating Annotation Noises for NER

1. The 'detector' file contains two sub-models based on contrastive learning.
	

	🚀🚀🚀 A TensorFlow implementation of Contrastive Learning model for extracting contextual and mentional features of Named Entities.

	##### Project Features

	* based on Tensorflow api. 
	* highly _scalable_; everything is *configurable*. 
	* modularized with clear structure.
	* very friendly for beginners.
	 

	# Project


	## Function Support

	- configuring all settings
	    - Running Mode: [`train`/`test`]
	    - Datasets(Input/Output): 
	    - Labeling Scheme: 
		- [`PER`|`LOC`|`ORG`|`MISC`]
		- ...
	    - Model Configuration: 
		- encoder: Bi-LSTM, layer, Bi/Uni-directional
		- decoder: softmax,
		- embedding level: char/word,
		- with/without self attention
		- hyperparameters,
		- ...
	    - Training Settings: 
		- optimazers: GD/Adagrad/AdaDelta/RMSprop/Adam
	    - Testing Settings,
	    - Api service Settings,
	    

	## Requirements

	- python >=3.5 
	- tensorflow >=1.8
	- numpy
	- pandas
	- ...

	## Setup

	#### Option A:
	download the repo for directly use.


	#### Option B: _TODO_
	install the BiLSTM-CRF package as a module.

	```
	pip install BiLSTM-CRF
	```

	usage:
	```
	from BiLSTM-CRF.engines.BiLSTM_CRFs_finetune import BiLSTM_CRFs as BC
	from BiLSTM-CRF.engines.DataManager import DataManager
	from BiLSTM-CRF.engines.Configer import Configer
	from BiLSTM-CRF.engines.utils import get_logger

	...

	config_file = r'/home/projects/system.config'
	configs = Configer(config_file)

	logger = get_logger(configs.log_dir)
	configs.show_data_summary(logger) # optional

	dataManager = DataManager(configs, logger)
	model = BC(configs, logger, dataManager)
		
	###### mode == 'train':
	model.train()

	###### mode == 'test':
	model.test()


	## Module Structure


	```

	├── main.py
	├── system.config
	├── README.md
	│
	├── checkpoints
	│   ├── BILSTM-CRFs
	│   │   ├── checkpoint
	│   │   └── ...
	│   └── ...
	├── data
	│   ├── logs
	│   ├── vocabs
	│   ├── train_file_name.xlsx
	│   ├── dev_file_name.xlsx
	│   ├── test_file_name.xlsx
	│   └── ...
	├── engines
	│   ├── BiLSTM_CRFs_finetune.py
	│   ├── Configer.py
	│   ├── DataManager.py
	│   └── utils.py
	└── 
	```

	- Folds
	    - in `engines` fold, providing the core functioning py.
	    - in `checkpoints-subfold` fold, model checkpoints are stored.
	    
	- Files
	    - `main.py` is the entry python file for the system.
	    - `system.config` is the configure file for all the system settings.
	    - `BiLSTM_CRFs_finetune.py` is the main model.
	    - `Configer.py` parses the `system.config`.
	    - `DataManager.py` manages the datasets and scheduling.
	    - `utils.py` provides on the fly tools.    

	## Quick Start

	Under following steps:

	#### step 1. composing your configure file in `system.config`.

	- configure the Datasets(Input/Output).
	- configure the Model architecture.


	#### step 2. starting training (necessary and compulsory)

	- configure the running mode.
	- configure the training setting.
	- run `main.py`.


	#### step 3. starting testing (optional)

	- configure the running mode.
	- configure the testing setting.
	- run `main.py`.


2. The 'SdBNN_GloVe' file is systematic deciation-based BNN(SdBNN) on GloVe embeddings.
	
	Taking the model of the CoNLL data set as an example, the following is introduced (the model of the WNUT data set and its structure are the same, only some parameters are different):
	
	Data:
		Download the glove related files according to the /SdBNN_GloVe/CoNLL/makefile
		/SdBNN_GloVe/CoNLL/data/SdBNN_train.txt is the train set of SdBNN.
		/SdBNN_GloVe/CoNLL/data/SdBNN1_train.txt is the train set of SdBNN1.
		/SdBNN_GloVe/CoNLL/data/SdBNN2_train.txt is the train set of SdBNN2.
		/SdBNN_GloVe/CoNLL/data/add_clear_softmax_conllpp_dev.txt and /SdBNN_GloVe/CoNLL/data/add_clear_softmax_original_conllpp_test.txt are the dataset of CoNLL.
		/SdBNN_GloVe/CoNLL/data/add_clear_softmax_conllpp_test.txt are the dataset of CoNLL++
	
	Run:
		(1) Build vocab from the data and extract trimmed glove vectors according to the config in /SdBNN_GloVe/CoNLL/model/config.py
 		'''
			python build_data.py

		'''
		(2) Train the model with
		'''
			python train.py

		'''
 		(3) Evaluate and interact with the model with
		'''
			python evaluate.py

		'''
	
3. The 'SdBNN_Flair' file is systematic deciation-based BNN(SdBNN) on Flair embeddings.
	
	The three SdBNN-based methods use the same underlying model(/SdBNN_Flair/flair_ner.py and  /SdBNN_Flair/sequence_tagger_with_system_error.py)
	
	Data:
		/SdBNN_Flair/data/SdBNN is the data set of SdBNN.
		/SdBNN_Flair/data/SdBNN1 is the data set of SdBNN1.
		/SdBNN_Flair/data/SdBNN2 is the data set of SdBNN2.
	
	Run:
		bash run.sh
		

