#!/bin/bash

# Create the checkpoints directory if it doesn't exist
mkdir -p checkpoints

# Prompt the user to select a model
echo "Please select the model to download:"
echo "s) Depth-Anything-V2-Small (24.8M)"
echo "b) Depth-Anything-V2-Base  (97.5M)"
echo "l) Depth-Anything-V2-Large (335.3M)"
echo "a) All models"
read -p "Enter your choice (s/b/l/a): " choice

# Function to download a model only if it doesn't already exist
download_model() {
  local url=$1
  local filename=$2
  local filepath="checkpoints/$filename"

  if [ -f "$filepath" ]; then
    echo "$filename already exists. Skipping download."
  else
    echo "Downloading $filename ..."
    wget -O "$filepath" "$url"
    echo "Downloaded to $filepath"
  fi
}

# Process user choice
case "$choice" in
  s)
    download_model "https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth?download=true" "depth_anything_v2_vits.pth"
    ;;
  b)
    download_model "https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth?download=true" "depth_anything_v2_vitb.pth"
    ;;
  l)
    download_model "https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth?download=true" "depth_anything_v2_vitl.pth"
    ;;
  a)
    download_model "https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth?download=true" "depth_anything_v2_vits.pth"
    download_model "https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth?download=true" "depth_anything_v2_vitb.pth"
    download_model "https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth?download=true" "depth_anything_v2_vitl.pth"
    ;;
  *)
    echo "Invalid choice. Exiting."
    exit 1
    ;;
esac
