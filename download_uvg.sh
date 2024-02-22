#!/bin/bash
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


# Script for donwloading and preparing UVG dataset from https://ultravideo.fi/dataset.html
# Modify the ROOT to be the directory in which you would like to download the
# data and convert it to png files.
export ROOT="/tmp/uvg"

video_names=(
    Beauty
    Bosphorus
    HoneyBee
    Jockey
    ReadySetGo
    ShakeNDry
    YachtRide
)

for vid in "${video_names[@]}"; do
  # Download video
  wget -P ${ROOT} https://ultravideo.fi/video/${vid}_1920x1080_120fps_420_8bit_YUV_RAW.7z
  # Unzip
  7z x ${ROOT}/${vid}_1920x1080_120fps_420_8bit_YUV_RAW.7z -o${ROOT}
  # Create directory for video
  mkdir ${ROOT}/${vid}
  # Convert video to png files.
  # For some reason, the unzipped 7z file is named 'ReadySteadyGo' instead of 'ReadySetGo'.
  if [[ $vid == "ReadySetGo" ]]; then
    ffmpeg -video_size 1920x1080 -pixel_format yuv420p -i ${ROOT}/ReadySteadyGo_1920x1080_120fps_420_8bit_YUV.yuv ${ROOT}/${vid}/%4d.png
  else
    ffmpeg -video_size 1920x1080 -pixel_format yuv420p -i ${ROOT}/${vid}_1920x1080_120fps_420_8bit_YUV.yuv ${ROOT}/${vid}/%4d.png
  fi
  # Then remove 7z, yuv and txt files
  rm -f ${ROOT}/*.7z
  rm -f ${ROOT}/*.yuv
  rm -f ${ROOT}/*.txt
done
