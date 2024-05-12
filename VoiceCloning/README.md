# Voice Cloning Module
This repository is an implementation of [Transfer Learning from Speaker Verification to
Multispeaker Text-To-Speech Synthesis](https://arxiv.org/pdf/1806.04558.pdf) (SV2TTS).

SV2TTS is a deep learning framework in three stages. In the first stage, one creates a digital representation of a voice from a few seconds of audio. In the second and third stages, this representation is used as reference to generate speech given arbitrary text.

## Setup

### 1. Install Requirements
i. Both Windows and Linux are supported. A GPU is recommended for training and for inference speed, but is not mandatory.
ii. Python 3.7 is recommended. Python 3.5 or greater should work, but you'll probably have to tweak the dependencies' versions. I recommend setting up a virtual environment using `venv`, but this is optional.
iii. Install [ffmpeg](https://ffmpeg.org/download.html#get-packages). This is necessary for reading audio files.
iv. Install [PyTorch](https://pytorch.org/get-started/locally/). Pick the latest stable version, your operating system, your package manager (pip by default) and finally pick any of the proposed CUDA versions if you have a GPU, otherwise pick CPU. Run the given command.
v. Install the remaining requirements with `pip install -r requirements.txt`

### 2. Training Models
#### i) Encoder
You can run encoder_train.py with the following optional parameters
- run_id (required): This is a string argument that the user must provide. It specifies a name for the current training run. This name is used for several purposes:
    - Identifies the training run for logging and organization.
    - Determines the directory where training outputs are stored (e.g., saved models).
    - Allows resuming training from a previously saved state with the same run_id (unless --force_restart is used).
- clean_data_root (required): This argument (type Path) specifies the directory containing preprocessed data for training the encoder part of the model.
- models_dir (optional): This argument (type Path) allows specifying the root directory for storing all trained models. By default, it's set to "saved_models". A subdirectory named after run_id will be created within this directory to store the specific model weights, backups, and plots generated during training.
vis_every (optional): This argument (type int) controls how often (number of steps) the training script updates and displays plots related to the loss function. The default value is 10 steps.
- umap_every (optional): This argument (type int) controls how often (number of steps) the script updates a dimensionality reduction technique called UMAP to visualize the data. Setting it to 0 disables UMAP updates. The default value is 100 steps.
- save_every (optional): This argument (type int) controls how often (number of steps) the script saves the current model state. Setting it to 0 disables saving the model during training. The default value is 500 steps.
- backup_every (optional): This argument (type int) controls how often (number of steps) the script creates backups of the saved model state. Setting it to 0 disables backups. The default value is 7500 steps.
- force_restart (optional): This argument is a flag (type bool). When set to True, it instructs the script to not load any previously saved model state and start training from scratch, even if a model with the same run_id exists.
- visdom_server (optional): This argument (type str) allows specifying the address of a visualization server (likely Visdom) used for plotting training progress. The default value is "http://localhost".
- no_visdom (optional): This argument is a flag (type bool). When set to True, it disables using Visdom for visualization altogether.

#### ii) Vocoder
You can run vocoder_train.py with the following optional parameters
- run_id (required): This is a string argument that the user must provide. It specifies a name for the current training run. This name is used for several purposes, similar to the previous snippet:
    - Identifies the training run for logging and organization.
    - Determines the directory where training outputs are stored (e.g., saved models).
    - Allows resuming training from a previously saved state with the same run_id (unless --force_restart is used).
- datasets_root (required): This argument (type Path) specifies the main directory containing the overall dataset for training. This directory should contain a subdirectory named "SV2TTS".
- syn_dir (optional): This argument (type Path) allows specifying the path to a subdirectory named "synthesizer" within the SV2TTS directory. This directory is expected to contain ground truth data for training, such as mel spectrograms, audio waveforms, and embeddings. If not provided, it defaults to <datasets_root>/SV2TTS/synthesizer/.
- voc_dir (optional): This argument (type Path) allows specifying the path to a subdirectory named "vocoder" within the SV2TTS directory. This directory is expected to contain preprocessed mel spectrograms for training, likely generated by a separate vocoder model. If not provided and --ground_truth is not used, it defaults to <datasets_root>/SV2TTS/vocoder/.
- m (same as previous snippet): This argument (type Path) allows specifying the root directory for storing all trained models. By default, it's set to "saved_models". A subdirectory named after run_id will be created within this directory to store the specific model weights, backups, and generated audio samples during training.
- g (ground_truth): This argument is a flag (type bool). When set to True, it instructs the script to train on ground truth mel spectrograms located in <datasets_root>/SV2TTS/synthesizer/mels. This takes priority over using preprocessed data from --voc_dir.
- s (same as previous snippet): This argument (type int) controls how often (number of steps) the script saves the current model state. Setting it to 0 disables saving the model during training. The default value is 1000 steps.
- b (same as previous snippet): This argument (type int) controls how often (number of steps) the script creates backups of the saved model state. Setting it to 0 disables backups. The default value is 25000 steps.
- f (same as previous snippet): This argument is a flag (type bool). When set to True, it instructs the script to not load any previously saved model state and start training from scratch, even if a model with the same run_id exists.

#### iii) Synthesizer
You can run synthesizer_train.py with the following optional parameters
- run_id (required): This argument (type str) is the same as before. It specifies a name for the current training run, used for organization, output storage, and potentially resuming training.
- syn_dir (required): This argument (type Path) differs from the previous snippets. Here, it's a required argument that specifies the path to the synthesizer directory. This directory is expected to contain the ground truth data for training, such as mel spectrograms, audio waveforms (wavs), and embeddings.
- m (same as previous snippet): This argument (type Path) allows specifying the root directory for storing all trained models. By default, it's set to "saved_models". A subdirectory named after run_id will be created within this directory to store the specific model weights and logs.
- s (same as previous snippet): This argument (type int) controls how often (number of steps) the script saves the current model state. Setting it to 0 disables saving the model during training. The default value is 1000 steps.
- b (same as previous snippet): This argument (type int) controls how often (number of steps) the script creates backups of the saved model state. Setting it to 0 disables backups. The default value is 25000 steps.
- f (same as previous snippet): This argument is a flag (type bool). When set to True, it instructs the script to not load any previously saved model state and start training from scratch, even if a model with the same run_id exists.
- hparams (optional): This argument (type str) allows specifying hyperparameter overrides for the training process. Hyperparameters are settings that control the training behavior but are not part of the model itself. This argument takes a comma-separated list of "name=value" pairs, allowing you to adjust specific hyperparameters without modifying the main training script. The default value is an empty string, meaning no overrides are applied.

Pretrained models are now downloaded automatically. If this doesn't work for you, you can manually download them [here](https://drive.google.com/drive/folders/1gepTaAwryRh7FjCKuIF3hCT0WpH7f1_z?usp=sharing).

### 3. (Optional) Test Configuration
Before you download any dataset, you can begin by testing your configuration with:

`python demo_cli.py`

If all tests pass, you're good to go.

### 4. (Optional) Download Datasets
For playing with the toolbox alone, I only recommend downloading [`LibriSpeech/train-clean-100`](https://www.openslr.org/resources/12/train-clean-100.tar.gz). Extract the contents as `<datasets_root>/LibriSpeech/train-clean-100` where `<datasets_root>` is a directory of your choosing. Other datasets are supported in the toolbox, see [here](https://github.com/CorentinJ/Real-Time-Voice-Cloning/wiki/Training#datasets). You're free not to download any dataset, but then you will need your own data as audio files or you will have to record it with the toolbox.

### 5. Launch the Toolbox
You can then try the toolbox:

`python demo_toolbox.py -d <datasets_root>`  
or  
`python demo_toolbox.py`  

depending on whether you downloaded any datasets. If you are running an X-server or if you have the error `Aborted (core dumped)`, see [this issue](https://github.com/CorentinJ/Real-Time-Voice-Cloning/issues/11#issuecomment-504733590).
