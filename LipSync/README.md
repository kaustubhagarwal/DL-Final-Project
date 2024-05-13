# **Lip Syncing Module**

This code is part of the paper: _A Lip Sync Expert Is All You Need for Speech to Lip Generation In the Wild_ published at ACM Multimedia 2020. 

[Paper](http://arxiv.org/abs/2008.10010)

## Prerequisites
Install necessary packages using `pip install -r requirements.txt`.

## Lip-syncing videos using the pre-trained models (Inference)

```bash
python inference.py --checkpoint_path <ckpt> --face <video.mp4> --audio <an-audio-source> 
```
The result is saved (by default) in `../result.mp4`. You can specify it as an argument,  similar to several other available options. The audio source can be any file supported by `FFMPEG` containing audio data: `*.wav`, `*.mp3` or even a video file, from which the code will automatically extract the audio.

## Training

There are two major steps: (i) Train the expert lip-sync discriminator, (ii) Train the Wav2Lip model(s).

##### Training the expert discriminator
You can download [the pre-trained weights](https://drive.google.com/drive/folders/1uN6hci10QlIZb7pg1D5MVaMY6Dgrnsp5?usp=share_link) if you want to skip this step.
Else you can run
```bash
python color_syncnet_train_plots.py
```
##### Training the Wav2Lip models
You can either train the model without the additional visual quality disriminator (< 1 day of training) or use the discriminator (~2 days). For the former, run: 
```bash
python wav2lip_train.py --data_root lrs2_preprocessed/ --checkpoint_dir <folder_to_save_checkpoints> --syncnet_checkpoint_path <path_to_expert_disc_checkpoint>
```

To train with the visual quality discriminator, you should run `hq_wav2lip_train.py` instead. The arguments for both the files are similar. In both the cases, you can resume training as well. Look at `python wav2lip_train.py --help` for more details. You can also set additional less commonly-used hyper-parameters at the bottom of the `hparams.py` file.

Training on datasets other than LRS2
------------------------------------
Training on other datasets might require modifications to the code. Please read the following before you raise an issue:

- You might not get good results by training/fine-tuning on a few minutes of a single speaker. This is a separate research problem, to which we do not have a solution yet. Thus, we would most likely not be able to resolve your issue. 
- You must train the expert discriminator for your own dataset before training Wav2Lip.
- If it is your own dataset downloaded from the web, in most cases, needs to be sync-corrected.
- Be mindful of the FPS of the videos of your dataset. Changes to FPS would need significant code changes. 
- The expert discriminator's eval loss should go down to ~0.25 and the Wav2Lip eval sync loss should go down to ~0.2 to get good results. 

When raising an issue on this topic, please let us know that you are aware of all these points.

We have an HD model trained on a dataset allowing commercial usage. The size of the generated face will be 192 x 288 in our new model.

Evaluation
----------
Please check the `evaluation/` folder for the instructions.

License and Citation
----------
Theis repository can only be used for personal/research/non-commercial purposes. However, for commercial requests, please contact us directly at radrabha.m@research.iiit.ac.in or prajwal.k@research.iiit.ac.in. We have an HD model trained on a dataset allowing commercial usage. The size of the generated face will be 192 x 288 in our new model. Please cite the following paper if you use this repository:
```
@inproceedings{10.1145/3394171.3413532,
author = {Prajwal, K R and Mukhopadhyay, Rudrabha and Namboodiri, Vinay P. and Jawahar, C.V.},
title = {A Lip Sync Expert Is All You Need for Speech to Lip Generation In the Wild},
year = {2020},
isbn = {9781450379885},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3394171.3413532},
doi = {10.1145/3394171.3413532},
booktitle = {Proceedings of the 28th ACM International Conference on Multimedia},
pages = {484–492},
numpages = {9},
keywords = {lip sync, talking face generation, video generation},
location = {Seattle, WA, USA},
series = {MM '20}
}
```


Acknowledgements
----------
Parts of the code structure is inspired by this [TTS repository](https://github.com/r9y9/deepvoice3_pytorch). We thank the author for this wonderful code. The code for Face Detection has been taken from the [face_alignment](https://github.com/1adrianb/face-alignment) repository. We thank the authors for releasing their code and models. We thank [zabique](https://github.com/zabique) for the tutorial collab notebook.
