# Playable Video Generation
<br>
<p align="center">
    <img src="./resources/interactive_sequences.gif"/> <br />
</p>
<br>

> **Playable Video Generation**<br>
> [Willi Menapace](https://www.willimenapace.com/), [Stéphane Lathuilière](https://stelat.eu/), [Sergey Tulyakov](http://www.stulyakov.com/), [Aliaksandr Siarohin](https://github.com/AliaksandrSiarohin), [Elisa Ricci](http://elisaricci.eu/)<br>
> ArXiv<br>

> Paper: [ArXiv](https://arxiv.org/abs/2101.12195)<br>
> Supplementary: [Website](https://willi-menapace.github.io/playable-video-generation-website/)<br>
> Demo: [Try it Live](https://willi-menapace.github.io/playable-video-generation-website/play.html)<br>

> **Abstract:** *This paper introduces the unsupervised learning problem of playable video generation (PVG). In PVG, we aim at allowing a user to control the generated video by selecting a discrete action at every time step as when playing a video game. The difficulty of the task lies both in learning semantically consistent actions and in generating realistic videos conditioned on the user input. We propose a novel framework for PVG that is trained in a self-supervised manner on a large dataset of unlabelled videos. We employ an encoder-decoder architecture where the predicted action labels act as bottleneck. The network is constrained to learn a rich action space using, as main driving loss, a reconstruction loss on the generated video. We demonstrate the effectiveness of the proposed approach on several datasets with wide environment variety.*

# Overview

<br>
<p align="center">
    <img src="./resources/architecture.png"/> <br />
    <em>
    Figure 1. Illustration of the proposed CADDY model for playable video generation.
    </em>
</p>
<br>

Given a set of completely unlabeled videos, we jointly learn a set of discrete actions and a video generation model conditioned on the learned actions. At test time, the user can control the generated video on-the-fly providing action labels as if he or she was playing a videogame. We name our method CADDY. Our architecture for unsupervised playable video generation is composed by several components. An encoder E extracts frame representations from the input sequence. A temporal model estimates the successive states using a recurrent dynamics network R and an action network A which predicts the action label corresponding to the current action performed in the input sequence. Finally, a decoder D reconstructs the input frames. The model is trained using reconstruction as the main driving loss.

# Installation
We provide both a Conda environment and a Docker image for executing the code.

## Conda

Create and activate the conda environment:

`conda env create -f env.yml`

`conda activate video-generation`

## Docker

Build the docker image
`docker build -t video-generation:1.0 .`

Run the docker image. Mount the root directory to `/video-generation` in the docker container:
`docker run -it --gpus all --ipc=host -v /path/to/directory/video-generation:/video-generation video-generation:1.0 /bin/bash`

# Directory structure

Please create the following directories in the root of the project:

- `results`
- `checkpoints`
- `data`

# Datasets
Datasets can be downloaded at the following link:
[Google Drive](https://drive.google.com/drive/folders/1CuHK_-cFWih0F8AxB4b76FoBQ9RjWMww?usp=sharing)

- Breakout: Coming soon
- BAIR: bair_256_ours.tar.gz
- Tennis: Coming soon

Extract them under the `data` folder

# Pretrained Models

Pretrained models can be downloaded at the following link:
[Google Drive](https://drive.google.com/drive/folders/1xLlJ8Xh6_wOEEARwBcoeVng2Bbi-wAah?usp=sharing)

Place the directories under the `checkpoints` folder

# Playing

After downloading the checkpoints, you can interact with the model to generate playable videos with the following commands:

- Bair:
`python play.py --config configs/01_bair.yaml`

- Breakout:
`python play.py configs/breakout/02_breakout.yaml`

- Tennis:
`python play.py --config configs/03_tennis.yaml`

Alternatively, you can also try our [Live Demo](https://willi-menapace.github.io/playable-video-generation-website/play.html).

# Training

The models can be trained with the following command:

`python train.py --config configs/<config_file>`

Checkpoints are saved at regular intervals in the `checkpoints` directory.
Training can be monitored through Weights and Biases by logging to your personal account with the command:
`wandb init`

# Evaluation

Evaluation requires two steps. First, an evaluation dataset is built. Second, evaluation is carried our on the evaluation dataset.

To build the evaluation dataset please issue:

`python build_evaluation_dataset.py --config configs/<config_file>`

To run evaluation issue:

`python evaluate_dataset.py --config configs/evaluation/configs/<config_file>`

Evaluation results are saved under the `evaluation_results` directory.
