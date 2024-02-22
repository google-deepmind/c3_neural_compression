# Compression baselines

This directory contains compression results (in terms of bits-per-pixel and
PSNR) for various codecs (both classical and neural) on image and video
datasets.


## Datasets

The current available datasets are:

-   The [Kodak](https://r0k.us/graphics/kodak/) dataset, containing 24 images of
    resolution `512 x 768` or `768 x 512`. See `kodak.py`.
-   The [CLIC2020 professional validation dataset](http://clic.compression.cc/2021/tasks/index.html),
    containing 41 images at various resolutions. See `clic2020.py`.
-   The [UVG](https://ultravideo.fi/) dataset, containing 7 videos of resolution
    `1080 x 1920`, with either `300` or `600` frames. See `uvg.py`.


## Results format

The results are stored in a dictionary containing various fields. The definition
of each field is given below:

-   `bpp`: Bits-per-pixel. The number of bits required to store the image or
    video divided by the total number of pixels in the image or video.
-   `psnr`: Peak Signal to Noise Ratio in dB. For images, the PSNR is computed
    *per image* and then averaged across images. For videos, the PSNR is
    computed *per frame* and then averaged across frames. The reported PSNR is
    then the average across all videos of the average per frame PSNRs.
-   `psnr_of_mean_mse`: (Optional) For images, the PSNR obtained by first
    computing the MSE of each image and averaging this across images. The PSNR
    is then computed based on this average MSE.
-   `meta`: Dictionary containing meta-information about codec.

The definition of each field in meta information is given below:

-   `source`: Source of numerical results.
-   `reference`: Reference to paper or implementation of codec.
-   `type`: One of `classical`, `autoencoder` and `neural-field`. `classical`
    refers to traditional codecs such as JPEG. `autoencoder` refers to
    autoencoder based neural codecs. `neural-field` refers to neural field based
    codecs. Note that the distinction between a `neural field` and `autoencoder`
    based codec can be blurry.
-   `data`: (Optional) One of `single` and `multi`. `single` refers to a codec
    trained on a single image or video. `multi` refers to a codec trained on
    multiple images or videos (typically a large dataset). This field is not
    relevant for `classical` codecs.
-   `macs_per_pixel`: (Optional) Approximate amount of MACs per pixel required
    to decode an image or video. Contains a dict with three keys: 1. `min`: the
    MACs per pixel of the smallest model used by the neural codec, 2. `max`: the
    MACs per pixel of the largest model used by the neural codec, 3. `source`:
    the source of the numbers.
