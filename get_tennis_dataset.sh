#!/bin/bash

mkdir tmp
cd tmp

# Download youtube files
youtube-dl -o djokovic_federer_wimbledon.mp4 https://www.youtube.com/watch?v=TUikJi0Qhhw
youtube-dl -o nadal_kyrgios_wimbledon.mp4 https://www.youtube.com/watch?v=T4S5YmO0KOU

mv djokovic_federer_wimbledon.f137.mp4 djokovic_federer_wimbledon.mp4
mv nadal_kyrgios_wimbledon.f137.mp4 nadal_kyrgios_wimbledon.mp4
cd ..

# Splits the videos in shorter sequences
python -m dataset.acquisition.split_and_resize_video
# Moves the splitted segments to the tmp directory
mv tmp/djokovic_federer_wimbledon_splits/* tmp
mv tmp/nadal_kyrgios_wimbledon_splits/* tmp

# Extracts sequences from videos according to the annotations
python -m dataset.acquisition.convert_annotated_video_directory
# Makes train and validation splits
python -m dataset.acquisition.train_val_test_split
# Makes val and test sequences of fixed length
python -m dataset.acquisition.subsample_videos_and_make_fixed_length

mkdir data/tennis_v4_256_ours
mv tmp/tennis_v4_256_ours/train data/tennis_v4_256_ours/train
mv tmp/tennis_v4_256_ours/val_fixed_length data/tennis_v4_256_ours/val
mv tmp/tennis_v4_256_ours/test_fixed_length data/tennis_v4_256_ours/test

rm -rf tmp