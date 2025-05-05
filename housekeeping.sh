#!/bin/bash

TARGET_DIR="media"

cd "$TARGET_DIR" || exit

# Delete everything except the requested files
find . -maxdepth 1 -type f ! -name 'last.mp3' ! -name 'recorded_audi1.mp3' -delete
