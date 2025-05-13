#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --job-name=gray-scott
#SBATCH --gpus=1
#SBATCH --output=out_gray.log

module load CUDA
nvcc -O2 -lm sample.cu -o sample

#sizes=("256" "512" "1024" "2048" "4096")
#modes=("-s" "-p")

sizes=("256")
modes=("-p")

# Loop through each size and mode combination
for size in "${sizes[@]}"; do
  for mode in "${modes[@]}"; do
    # Run the sampling command with the current size and mode
    echo "Running sample $size with mode $mode..."
    srun sample $size $mode

    # Convert the frames to a video using ffmpeg
    echo "Converting frames to video for $size with mode $mode..."
    ffmpeg -y -framerate 10 -start_number 1 -i frames/frame_%04d.png videos/output${mode}_${size}.mp4 -loglevel quiet
  done
done
