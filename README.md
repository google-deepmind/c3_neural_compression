# C3 (neural compression)

This repository contains code for reproducing results in the paper
*C3: High-performance and low-complexity neural compression from a single image or video*
(abstract and arxiv link below).

C3 paper link: https://arxiv.org/abs/2312.02753

Project page: https://c3-neural-compression.github.io/

*Abstract: Most neural compression models are trained on large datasets of images or videos
in order to generalize to unseen data. Such generalization typically requires
large and expressive architectures with a high decoding complexity. Here we
introduce C3, a neural compression method with strong rate-distortion (RD)
performance that instead overfits a small model to each image or video
separately. The resulting decoding complexity of C3 can be an order of magnitude
lower than neural baselines with similar RD performance. C3 builds on COOL-CHIC
(Ladune et al.) and makes several simple and effective improvements for images.
We further develop new methodology to apply C3 to videos. On the CLIC2020 image
benchmark, we match the RD performance of VTM, the reference implementation of
the H.266 codec, with less than 3k MACs/pixel for decoding. On the UVG video
benchmark, we match the RD performance of the Video Compression Transformer
(Mentzer et al.), a well-established neural video codec, with less than 5k
MACs/pixel for decoding.*

This code can be used to train and evaluate the C3 model in the paper, that can
be used to reproduce the empirical results of the paper, including the
psnr/per-frame-mse values
(logged as `psnr_quantized` / `per_frame_distortion_quantized`) and the
corresponding bpp values (logged as `bpp_total`) for each image / video patch.
We have tested this code on a single NVIDIA P100 and V100 GPU, with Python 3.10.
Around 20M / 300M / 25G of hard disk space is required to download the
Kodak / CLIC2020 / UVG datasets respectively.

C3 builds on top of [COOL-CHIC](https://arxiv.org/abs/2212.05458) with official
[PyTorch implementation](https://github.com/Orange-OpenSource/Cool-Chic).

## Rate Distortion values and MACs per pixel

The rate-distortion values and MACs per pixel for C3 and other compression baselines can be found in the files under the `baselines` directory.

## Setup

We recommend installing this package into a Python virtual environment.
To set up a Python virtual environment with the required dependencies, run:

```shell
# create virtual environment
python3 -m venv /tmp/c3_venv
source /tmp/c3_venv/bin/activate
# update pip, setuptools and wheel
pip3 install --upgrade pip setuptools wheel
# clone repository
git clone https://github.com/google-deepmind/c3_neural_compression.git
# Navigate to root directory
cd c3_neural_compression
# install all required packages
pip3 install -r requirements.txt
# Include this directory in PYTHONPATH so we can import modules.
export PYTHONPATH=${PWD}:$PYTHONPATH
```

Once done with virtual environment, deactivate with command:

```shell
deactivate
```

then delete venv with command:

```shell
rm -r /tmp/c3_venv
```

## Setup UVG dataset (optional)
The Kodak and CLIC2020 image datasets are automatically downloaded via the data loader in `utils/data_loading.py`. However the UVG(UVG-1k) dataset requires some manual preparation.
Here are some instructions for Debian linux.

To set up the UVG dataset, first install `7z` (to unzip .7z files into .yuv files) and `ffmpeg` (to convert .yuv files to .png frames) via commands:

```shell
sudo apt-get install p7zip-full
sudo apt install ffmpeg
```

Then run `bash download_uvg.sh` after modifying the `ROOT` variable in `download_uvg.sh` to be the desired directory for storing the data. Note that this can take around 20 minutes.

## Run experiments
Set the hyperparameters in `image.py` or `video.py` as desired by modifying
the config values. Then inside the virtual environment,

```shell
cd experiments
```

and run the
[JAXline](https://github.com/deepmind/jaxline) experiment via command:

```shell
python3 -m image --config=../configs/kodak.py
```

or

```shell
python3 -m image --config=../configs/clic2020.py
```

or

```shell
python3 -m video --config=../configs/uvg.py
```

Note that for the UVG experiment, the value of `exp.dataset.root_dir` must match the value of the `ROOT` variable used for `download_uvg.sh`.

## Citing this work
If you use this code in your work, we ask you to please cite our work:

```latex
@article{c3_neural_compression,
  title={C3: High-performance and low-complexity neural compression from a single image or video},
  author={Kim, Hyunjik and Bauer, Matthias and Theis, Lucas and Schwarz, Jonathan Richard and Dupont, Emilien},
  journal={arXiv preprint arXiv:2312.02753},
  year={2023}
}
```

## License and disclaimer

Copyright 2024 DeepMind Technologies Limited

All software is licensed under the Apache License, Version 2.0 (Apache 2.0);
you may not use this file except in compliance with the Apache 2.0 license.
You may obtain a copy of the Apache 2.0 license at:
https://www.apache.org/licenses/LICENSE-2.0

All other materials are licensed under the Creative Commons Attribution 4.0
International License (CC-BY). You may obtain a copy of the CC-BY license at:
https://creativecommons.org/licenses/by/4.0/legalcode

Unless required by applicable law or agreed to in writing, all software and
materials distributed here under the Apache 2.0 or CC-BY licenses are
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
either express or implied. See the licenses for the specific language governing
permissions and limitations under those licenses.

This is not an official Google product.
