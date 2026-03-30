#!/bin/bash
# Compose SAM3 segmentation mask images into a video

# Parameters
INPUT_DIR="./outputs/sam3_visualizations"
OUTPUT_VIDEO="${INPUT_DIR}/demo_masked_video.mp4"
FRAMERATE=30

# Check directory
if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: directory $INPUT_DIR does not exist"
    exit 1
fi

# Count frames
FRAME_COUNT=$(ls "$INPUT_DIR"/*_obj0.jpg 2>/dev/null | wc -l)
if [ $FRAME_COUNT -eq 0 ]; then
    echo "Error: no mask images found"
    exit 1
fi

echo "Found $FRAME_COUNT mask image frames"

# Create symlinks for ffmpeg sequential reading (filenames may contain spaces)
TMP_DIR=$(mktemp -d)
cd "$INPUT_DIR"
i=0
for f in $(ls *_frame*_obj0.jpg | sort -V); do
    ln -s "$(pwd)/$f" "$TMP_DIR/frame_$(printf "%04d" $i).jpg"
    i=$((i+1))
done

echo "Generating video..."
ffmpeg -framerate $FRAMERATE \
    -i "$TMP_DIR/frame_%04d.jpg" \
    -c:v libx264 \
    -pix_fmt yuv420p \
    -y "$OUTPUT_VIDEO" \
    2>&1 | grep -E "(frame=|Output)"

# Clean up temporary files
rm -rf "$TMP_DIR"

if [ -f "$OUTPUT_VIDEO" ]; then
    echo "Video generated: $OUTPUT_VIDEO"
    echo "Video info:"
    ffprobe -v error -show_entries format=duration:stream=width,height,nb_frames -of default=noprint_wrappers=1 "$OUTPUT_VIDEO"
else
    echo "Video generation failed"
    exit 1
fi
