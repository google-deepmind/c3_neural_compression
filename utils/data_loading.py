# Copyright 2024 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Utils for loading and processing datasets."""

import os
import shutil
from typing import Any, Callable
import urllib
import urllib.request

import numpy as np
from PIL import Image
import torch
from torch.utils import data
from torchvision.datasets import folder
from torchvision.datasets import utils as dset_utils
from torchvision.transforms import v2 as tfms
import tqdm


DATASET_ATTRIBUTES = {
    'clic2020': {
        'num_channels': 3,
        'resolution': None,  # Resolution varies by image
        'type': 'image',
        'train_size': 41,
        'test_size': 41,
    },
    'kodak': {
        'num_channels': 3,
        # H x W
        'resolution': (512, 768),
        'type': 'image',
        'train_size': 24,
        'test_size': 24,
    },  # Identical set of 24 images
    'uvg': {
        'filenames': [
            'Beauty',
            'Bosphorus',
            'HoneyBee',
            'Jockey',
            'ReadySetGo',
            'ShakeNDry',
            'YachtRide',
        ],
        # Total number of frames in each of the video above
        'frames': [600, 600, 600, 600, 600, 300, 600],
        'num_channels': 3,
        'resolution': (1080, 1920),
        'fps': 120,
        'original_format': '420_8bit_YUV',  # 4:2:0 chroma subsampled 8 bit YUV
        'type': 'video',
        'train_size': 6 * 600 + 300,  # total number of frames
        'test_size': 6 * 600 + 300,  # total number of frames
    },
}


class Kodak(data.Dataset):
  """Data loader for Kodak image dataset at https://r0k.us/graphics/kodak/ ."""

  def __init__(
      self,
      root: str,
      force_download: bool = False,
      transform: Callable[[Any], torch.Tensor] | None = None,
  ):
    """Constructor.

    Args:
        root: base directory for downloading dataset. Directory is created if it
          does not already exist.
        force_download: if False, only downloads the dataset if it doesn't
          already exist. If True, force downloads the dataset into the root,
          overwriting existing files.
        transform: callable for transforming the loaded images.
    """
    self.root = root
    self.transform = transform
    self.num_images = 24

    self.path_list = [
        os.path.join(self.root, 'kodim{:02}.png'.format(i))
        for i in range(1, self.num_images + 1)  # Kodak images start at 1
    ]

    if force_download:
      self._download()
    else:
      # Check if root directory exists
      if os.path.exists(self.root):
        # Check that there is a correct number of png files.
        download_files = False
        count = 0
        for filename in os.listdir(self.root):
          if filename.endswith('.png'):
            count += 1
        if count != self.num_images:
          print('Files are missing, so proceed with download.')
          download_files = True
      else:
        os.makedirs(self.root)
        download_files = True

      if download_files:
        self._download()
      else:
        print(
            'Files already exist and `force_download=False`, so do not download'
        )

  def _download(self):
    for i in tqdm.tqdm(range(self.num_images), desc='Downloading Kodak images'):
      path = self.path_list[i]
      img_num = i + 1  # Kodak images start at 1
      img_name = 'kodim{:02}.png'.format(img_num)
      url = 'http://r0k.us/graphics/kodak/kodak/' + img_name
      with (
          urllib.request.urlopen(url) as response,
          open(path, 'wb') as out_file,
      ):
        shutil.copyfileobj(response, out_file)

  def __len__(self):
    return len(self.path_list)

  def __getitem__(self, idx):
    path = self.path_list[idx]
    image = folder.default_loader(str(path))

    if self.transform is not None:
      image = self.transform(image)

    return {'array': image}


class CLIC2020(data.Dataset):
  """Data loader for the CLIC2020 validation image dataset at http://compression.cc/tasks/ ."""

  data_dict = {
      'filename': 'val.zip',
      'md5': '7111ee240435911db04dbc5f40d50272',
      'url': (
          'https://data.vision.ee.ethz.ch/cvl/clic/professional_valid_2020.zip'
      ),
  }

  def __init__(
      self,
      root: str,
      force_download: bool = False,
      transform: Callable[[Image.Image], torch.Tensor] | None = None,
  ):
    """Constructor.

    Args:
      root: base directory for downloading dataset. Directory is created if it
        does not already exist.
      force_download: if False, only downloads the dataset if it doesn't already
        exist. If True, force downloads the dataset into the root, overwriting
        existing files.
      transform: callable for transforming the loaded images.
    """
    self.root = root
    self.root_valid = os.path.join(root, 'valid')
    self.transform = transform
    self.num_images = 41

    if force_download:
      self._download()
    else:
      # Check if root directory exists
      if os.path.exists(self.root_valid):
        # Check that there is a correct number of png files.
        download_files = False
        count = 0
        for filename in os.listdir(self.root_valid):
          if filename.endswith('.png'):
            count += 1
        if count != self.num_images:
          print('Files are missing, so proceed with download.')
          download_files = True
      else:
        os.makedirs(self.root, exist_ok=True)
        download_files = True

      if download_files:
        self._download()
      else:
        print('Files already exist and `force_download=False`, so do not '
              'download')

      paths = sorted(os.listdir(self.root_valid))
      assert len(paths) == self.num_images
      self.path_list = [os.path.join(self.root_valid, path) for path in paths]

  def __getitem__(self, index: int) -> Image.Image:
    path = self.path_list[index]

    image = folder.default_loader(path)

    if self.transform is not None:
      image = self.transform(image)

    return {'array': image}

  def __len__(self) -> int:
    return len(self.path_list)

  def _download(self):
    extract_root = str(self.root)
    dset_utils.download_and_extract_archive(
        **self.data_dict,
        download_root=str(self.root),
        extract_root=extract_root,
    )


class UVG(data.Dataset):
  """Data loader for UVG dataset at https://ultravideo.fi/dataset.html ."""

  def __init__(
      self,
      root: str,
      patch_size: tuple[int, int, int] = (300, 1080, 1920),
      transform: Callable[[Image.Image], torch.Tensor] | None = None,
  ):
    """Constructor.

    Args:
      root: base directory for downloading dataset.
      patch_size: dimensionality of our video patch as a tuple (t, h, w).
      transform: callable for transforming the each frame of loaded video.
    """
    self.root = root
    self.transform = transform

    input_res = DATASET_ATTRIBUTES['uvg']['resolution']
    video_names = DATASET_ATTRIBUTES['uvg']['filenames']
    self.num_frames_per_vid = DATASET_ATTRIBUTES['uvg']['frames']
    self.cum_frames = np.cumsum(self.num_frames_per_vid)  # [600, ..., 3900]
    self.cum_frames_from_zero = [0, *self.cum_frames][:-1]  # [0, ..., 3300]

    self.path_list = []
    for video_idx, video_name in enumerate(video_names):
      png_dir = os.path.join(self.root, video_name)
      assert os.path.exists(png_dir)
      count = 0
      for filename in sorted(os.listdir(png_dir)):
        if filename.endswith('.png'):
          self.path_list.append(os.path.join(png_dir, filename))
          count += 1
      assert count == self.num_frames_per_vid[video_idx], count

    self.num_total_frames = len(self.path_list)
    assert self.num_total_frames == sum(self.num_frames_per_vid)
    self.pt, self.ph, self.pw = patch_size
    assert 300 % self.pt == 0
    assert input_res[0] % self.ph == 0
    assert input_res[1] % self.pw == 0
    # (T//pt, H//ph, W//pw)
    self.num_patches = (
        self.num_total_frames // self.pt,
        input_res[0] // self.ph,
        input_res[1] // self.pw,
    )
    # Compute the start and end indices for each video, where indices are
    # assigned to patches.
    # Note that if num_spatial_patches = 1 and pt = 1 (i.e. each patch is a
    # single frame), then the below is identical to self.cum_frames_from_zero.
    num_spatial_patches = self.num_patches[1] * self.num_patches[2]
    self.start_idx_per_vid = [
        num_spatial_patches*frame_idx//self.pt
        for frame_idx in self.cum_frames_from_zero
    ]  # [num_spatial_patches*0//pt, ..., num_spatial_patches*3300//pt]
    self.end_idx_per_vid = [
        num_spatial_patches*frame_idx//self.pt
        for frame_idx in self.cum_frames
    ]  # [num_spatial_patches*600//pt, ..., num_spatial_patches*3900//pt]

  def load_frame(self, frame_idx: int):
    """Load a single frame."""
    assert frame_idx < self.num_total_frames
    path = self.path_list[frame_idx]
    frame = folder.default_loader(path)
    if self.transform is not None:
      frame = self.transform(frame)
    return frame  # [H, W, C]

  def load_patch(self, patch_idx: tuple[int, int, int]):
    """Load a single patch from 3D patch index."""
    patch_idx_t, patch_idx_h, patch_idx_w = patch_idx
    start_h = patch_idx_h * self.ph
    start_w = patch_idx_w * self.pw
    patch_frames = []
    for dt in range(self.pt):
      t = patch_idx_t * self.pt + dt
      frame = self.load_frame(t)
      patch_frame = frame[
          start_h: start_h + self.ph, start_w: start_w + self.pw
      ]
      patch_frames.append(patch_frame)
    patch = torch.stack(patch_frames, dim=0)  # [pt, ph, pw, C]
    return patch

  def __getitem__(self, index: int) -> Image.Image:
    # Note that index ranges from 0 to num_total_patches - 1 where
    # num_total_patches = np.prod(self.num_patches)

    # Compute video_idx from index
    video_idx = None
    start_idx = None
    for video_idx, (start_idx, end_idx) in enumerate(
        zip(self.start_idx_per_vid, self.end_idx_per_vid)
    ):
      if index < end_idx:
        break
    assert video_idx in [0, 1, 2, 3, 4, 5, 6]
    _, nph, npw = self.num_patches
    # The below are the indices for a given patch in each of the axes.
    # e.g. patch_idx = (0, 1, 2) would give the patch
    # vid[:pt, ph:2*ph, 2*pw:3*pw] for the first video `vid`.
    patch_idx = (
        index // (nph * npw),
        (index % (nph * npw)) // npw,
        (index % (nph * npw)) % npw,
    )
    patch = self.load_patch(patch_idx)  # [pt, ph, pw, C]
    video_id = [video_idx] * self.pt
    # Below is the timestep within the video, so its values are in [0, 599]
    patch_first_frame_idx = (index - start_idx) // (nph * npw)
    timestep = [patch_first_frame_idx * self.pt + dt for dt in range(self.pt)]
    return {
        'array': patch,
        'timestep': timestep,
        'video_id': video_id,
        'patch_id': index,
    }

  def __len__(self) -> int:
    return np.prod(self.num_patches)


def load_dataset(
    dataset_name: str,
    root: str,
    skip_examples: int | None = None,
    num_examples: int | None = None,
    # The below args are for UVG data only.
    num_frames: int | None = None,
    spatial_patch_size: tuple[int, int] | None = None,
    video_idx: int | None = None,
):
  """Pytorch dataset loaders.

  Args:
    dataset_name (string): One of elements of DATASET_NAMES.
    root (string): Absolute path of root directory in which the dataset
      files live.
    skip_examples (int): Number of examples to skip.
    num_examples (int): If not None, returns only the first num_examples of the
      dataset.
    num_frames (int): Number of frames in a single patch of video.
    spatial_patch_size (tuple): Height and width of a single patch of video.
    video_idx (int): Video index to be used for training on particular videos.
      If set to None, train on all videos.

  Returns:
    dataset iterator with fields 'array' (float32 in [0,1]) and additionally
    for UVG: 'timestep' (int32), 'video_id' (int32), 'patch_id' (int32).
  """
  # Define transform that is applied to each image / frame of video.
  transform = tfms.Compose([
      # Convert PIL image to pytorch tensor.
      tfms.ToImage(),
      # [C, H, W] -> [H, W, C].
      tfms.Lambda(lambda im: im.permute((-2, -1, -3)).contiguous()),
      # Scale from [0, 255] to [0, 1] range.
      tfms.ToDtype(torch.float32, scale=True),
  ])

  # Load dataset
  if dataset_name.startswith('uvg'):
    patch_size = (num_frames, *spatial_patch_size)
    ds = UVG(
        root=root,
        patch_size=patch_size,
        transform=transform,
    )
    # Get indices to obtain a subset of the dataset from video_idx,
    # skip_examples and num_examples.
    # First narrow down based on video_idx.
    if video_idx:
      start_idx = ds.start_idx_per_vid[video_idx]
      end_idx = ds.end_idx_per_vid[video_idx]
    else:
      start_idx = ds.start_idx_per_vid[0]
      end_idx = ds.end_idx_per_vid[-1]
  elif dataset_name.startswith('kodak'):
    ds = Kodak(root=root, transform=transform)
    start_idx = 0
    end_idx = ds.num_images
  elif dataset_name.startswith('clic'):
    ds = CLIC2020(root=root, transform=transform)
    start_idx = 0
    end_idx = ds.num_images
  else:
    raise ValueError(f'Unrecognized dataset {dataset_name}.')

  # Adjust start_idx and end_idx based on skip_examples and num_examples
  if skip_examples is not None:
    start_idx = start_idx + skip_examples
  if num_examples is not None:
    end_idx = min(end_idx, start_idx + num_examples)

  indices = tuple(range(start_idx, end_idx))
  ds = data.Subset(ds, indices)

  # Convert to DataLoader
  dl = data.DataLoader(ds, batch_size=None)

  return dl
