# Vocals Separation Tool

A Python-based tool for extracting vocals from audio files using the Demucs v4 model. This tool is optimized for processing large numbers of audio files efficiently using multiple GPUs.

## Features

- Multi-GPU support (default: 3 GPUs)
- Batch processing for improved efficiency
- Preserves original directory structure in output
- Supports multiple audio formats (mp3, wav, flac, ogg, m4a)
- Configurable output bitrate and segment length
- Comprehensive error logging and progress tracking
- Failed file tracking for each GPU

## Requirements

- Python 3.x
- CUDA-compatible GPU(s)

## Installation

1. Clone this repository
2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

Basic usage:
```bash
python d_vocals.py <input_directory> --bitrate 64
```

Advanced usage with all options:
```bash
python d_vocals.py <input_directory> \
    --output vocal_output \
    --segment 7 \
    --model htdemucs \
    --bitrate 128
```

### Command Line Arguments

- `inputs`: Input audio files or directories (required)
- `--output`: Output directory (default: "vocal_output")
- `--segment`: Segment length in seconds, max 7.8 (default: 7)
- `--model`: Model to use (default: "htdemucs")
- `--bitrate`: MP3 output bitrate in kbps (default: 128)

## Performance Considerations

The tool uses batch processing to improve GPU utilization. Key performance factors:

- **Batch Size**: Default is 4 files per batch. Larger batches may:
  - Increase memory usage due to padding
  - Affect processing time
  - Risk out-of-memory errors on longer files

- **GPU Memory**: Each audio file requires:
  - Memory for input audio
  - Model weights
  - Processing buffers
  - Output separations

- **File Length**: Longer files in a batch cause more padding overhead as all files are padded to match the longest file's length

## Output Structure

- Processed files maintain the same directory structure as input
- Output files are named: `original_filename_vocals.mp3`
- Failed files are logged in:
  - Individual GPU logs: `failed_files_gpuX.txt`
  - Combined log: `failed_files.txt`

## Logging

- Progress is logged to both console and `vocals_separation.log`
- Includes:
  - Processing progress per GPU
  - Error messages
  - Estimated completion time
  - Failed file tracking

## Error Handling

The tool includes robust error handling:
- Individual file failures don't stop batch processing
- Failed files are tracked per GPU
- Comprehensive error logging
- Memory management between batches

## Known Limitations

- Maximum segment length of 7.8 seconds (Demucs v4 limitation)
- GPU memory usage increases with batch size
- Padding overhead in batches with varying file lengths