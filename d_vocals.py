import os
import gc
import json
import torch
import torchaudio
import numpy as np
import concurrent.futures
from pathlib import Path
from tqdm import tqdm
import argparse
import glob
import demucs.separate
import shutil
import warnings

warnings.filterwarnings("ignore")

# Constants
OUTPUT_SAMPLE_RATE = 16000  # 16kHz output
OUTPUT_CHANNELS = 1  # mono output
MP3_BITRATE = 96  # kbps, good balance for voice
BASE_DIR = "."
OUTPUT_DIR = os.path.join(BASE_DIR, "vocal_output")

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

def process_file(args):
    """Process a single file using Demucs."""
    audio_file, output_base_dir, input_base_dir, gpu_id = args
    
    try:
        # Calculate relative path to maintain folder structure
        rel_path = os.path.relpath(audio_file, input_base_dir)
        
        # Create output path with same folder structure
        output_name = f"{os.path.splitext(os.path.basename(audio_file))[0]}_vocals.mp3"
        output_path = os.path.join(output_base_dir, os.path.dirname(rel_path), output_name)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Set CUDA device for this process
        if gpu_id is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Use CPU if no GPU specified
        
        # Create temporary directory for Demucs output
        with tempfile.TemporaryDirectory() as temp_dir:
            # Run Demucs
            args = [
                "-n", "htdemucs",  # use htdemucs model
                "--two-stems=vocals",  # only extract vocals
                "--mp3",  # output as mp3
                "--mp3-bitrate", str(MP3_BITRATE),
                "--segment", "7",  # optimal segment length for htdemucs
                "-j", "1",  # single thread per process since we're using multiprocessing
                "-o", temp_dir,  # output to temp directory
            ]
            
            # Add device selection
            if gpu_id is not None:
                args.extend(["-d", "cuda"])
            else:
                args.extend(["-d", "cpu"])
                
            # Add input file
            args.append(audio_file)
            
            # Run Demucs
            demucs.separate.main(args)
            
            # Find the vocals file in the output
            model_dir = "htdemucs"
            track_name = os.path.splitext(os.path.basename(audio_file))[0]
            vocals_file = os.path.join(temp_dir, model_dir, track_name, "vocals.mp3")
            
            if os.path.exists(vocals_file):
                # Convert to 16kHz mono
                waveform, sr = torchaudio.load(vocals_file)
                if waveform.shape[0] > 1:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)
                if sr != OUTPUT_SAMPLE_RATE:
                    waveform = torchaudio.transforms.Resample(sr, OUTPUT_SAMPLE_RATE)(waveform)
                
                # Save final output
                torchaudio.save(
                    output_path,
                    waveform,
                    OUTPUT_SAMPLE_RATE,
                    format='mp3',
                    encoding_args={'bit_rate': MP3_BITRATE * 1000}
                )
                return output_path
            
        return None
                
    except Exception as e:
        print(f"Error processing {audio_file}: {str(e)}")
        return None

def process_batch(args):
    """Process a batch of files using multiple GPUs."""
    file_list, output_base_dir, input_base_dir, gpu_id = args
    
    results = []
    for audio_file in tqdm(file_list, desc="Processing files", leave=False):
        result = process_file((audio_file, output_base_dir, input_base_dir, gpu_id))
        if result:
            results.append(result)
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Extract vocals from multiple audio files using Demucs")
    parser.add_argument("inputs", nargs='+', help="Input audio file paths or directory (use '*' for wildcards)")
    parser.add_argument("--output", help="Output directory for vocals")
    parser.add_argument("--gpus", type=int, default=None, help="Number of GPUs to use (default: all available)")
    parser.add_argument("--workers", type=int, default=None, help="Number of parallel workers (default: 2x num_gpus)")
    
    args = parser.parse_args()
    
    # Set output directory
    output_base_dir = args.output if args.output else OUTPUT_DIR
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Collect all input files
    input_files = []
    input_base_dirs = []
    
    for input_path in args.inputs:
        # If it's a directory, get all audio files in it
        if os.path.isdir(input_path):
            base_dir = input_path
            for ext in ['mp3', 'wav', 'flac', 'ogg', 'm4a']:
                files = glob.glob(os.path.join(input_path, f"**/*.{ext}"), recursive=True)
                input_files.extend(files)
                input_base_dirs.extend([base_dir] * len(files))
        
        # If it contains wildcards, expand them
        elif '*' in input_path:
            base_dir = os.path.dirname(input_path.split('*')[0])
            if not base_dir:
                base_dir = "."
            files = glob.glob(input_path)
            input_files.extend(files)
            input_base_dirs.extend([base_dir] * len(files))
        
        # Otherwise, add the file directly
        else:
            input_files.append(input_path)
            input_base_dirs.append(os.path.dirname(input_path) if os.path.dirname(input_path) else ".")
    
    # Remove duplicates while preserving order
    seen = set()
    unique_files = []
    unique_base_dirs = []
    for f, d in zip(input_files, input_base_dirs):
        if f not in seen:
            seen.add(f)
            unique_files.append(f)
            unique_base_dirs.append(d)
    
    input_files = unique_files
    input_base_dirs = unique_base_dirs
    
    if not input_files:
        print("No input files found.")
        exit(1)
    
    print(f"Found {len(input_files)} files to process")
    
    # Setup GPU distribution
    num_gpus = args.gpus if args.gpus is not None else torch.cuda.device_count()
    num_gpus = min(num_gpus, torch.cuda.device_count())
    
    # Set number of workers (2x num_gpus by default)
    num_workers = args.workers if args.workers is not None else max(1, num_gpus * 2)
    
    # Split files into batches
    files_per_worker = len(input_files) // num_workers + (1 if len(input_files) % num_workers else 0)
    batches = []
    
    for i in range(0, len(input_files), files_per_worker):
        batch_files = input_files[i:i + files_per_worker]
        # Assign GPUs round-robin to batches, starting from 0
        batch_gpu_id = i % num_gpus if num_gpus > 0 else None
        batches.append((batch_files, output_base_dir, input_base_dirs[0], batch_gpu_id))
    
    # Process all batches in parallel
    processed_files = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_batch, batch) for batch in batches]
        
        # Show progress bar
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing batches"):
            try:
                results = future.result()
                processed_files.extend(results)
            except Exception as e:
                print(f"Error processing batch: {str(e)}")
    
    print(f"\nSuccessfully processed {len(processed_files)} files")

if __name__ == "__main__":
    import tempfile
    main() 