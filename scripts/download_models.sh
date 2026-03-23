#!/usr/bin/env bash
set -e

mkdir -p checkpoints

if [ ! -f checkpoints/groundingdino_swint_ogc.pth ]; then
  echo "downloading groundingdino checkpoint..."
  wget -O checkpoints/groundingdino_swint_ogc.pth \
    https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
fi

if [ ! -f checkpoints/sam_vit_h_4b8939.pth ]; then
  echo "downloading sam checkpoint..."
  wget -O checkpoints/sam_vit_h_4b8939.pth \
    https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
fi

echo "all checkpoints ready"