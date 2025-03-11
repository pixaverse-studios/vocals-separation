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
from demucs.pretrained import get_model
from demucs.apply import apply_model
import warnings
import torch.multiprocessing as mp

warnings.filterwarnings("ignore")

# Constants
BASE_DIR = "."
OUTPUT_DIR = os.path.join(BASE_DIR, "vocal_output")
TARGET_SAMPLE_RATE = 16000
TARGET_CHANNELS = 1
MP3_BITRATE = 96  # Reduced bitrate for 16kHz mono

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_audio(audio_path, device):
    """Load audio file and convert to target format"""
    try:
        # Load audio with torchaudio
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Move waveform to device first
        waveform = waveform.to(device)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Resample if necessary
        if sample_rate != TARGET_SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(
                sample_rate, TARGET_SAMPLE_RATE
            ).to(device)
            waveform = resampler(waveform)
        
        return waveform, TARGET_SAMPLE_RATE
    
    except Exception as e:
        print(f"Error loading {audio_path}: {str(e)}")
        return None, None

def save_audio(waveform, sample_rate, output_path):
    """Save audio in MP3 format"""
    try:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save as MP3
        torchaudio.save(
            output_path,
            waveform,
            sample_rate,
            format="mp3",
            encoding_args={
                "mp3_bitrate": MP3_BITRATE
            }
        )
        return True
    except Exception as e:
        print(f"Error saving {output_path}: {str(e)}")
        return False

class DemucsVocalExtractor:
    def __init__(self, gpu_id=0, segment_size=7.8):
        self.device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
        
        # Set memory optimization
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.backends.cudnn.benchmark = True
            # Set memory optimization
            torch.cuda.set_per_process_memory_fraction(0.7, gpu_id)  # Use 70% of GPU memory
        
        self.model = get_model("htdemucs").to(self.device)
        self.segment_size = segment_size
        self.sample_rate = self.model.samplerate
        
        # Set segment size in samples
        self.segment_samples = int(self.segment_size * self.sample_rate)
        
        # Optimize for inference
        self.model.eval()
        torch.set_grad_enabled(False)
    
    def extract_vocals(self, audio_path, output_path):
        """Extract vocals from audio file"""
        try:
            # Load and normalize audio
            waveform, sr = load_audio(audio_path, self.device)
            if waveform is None:
                return False
            
            # Convert to model's sample rate if needed
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(
                    sr, self.sample_rate
                ).to(self.device)
                waveform = resampler(waveform)
            
            # Process audio
            with torch.no_grad():
                # Move to device
                waveform = waveform.to(self.device)
                
                # Apply model with segments
                sources = apply_model(
                    self.model,
                    waveform,
                    device=self.device,
                    segment=self.segment_samples,
                    overlap=0.1
                )
                
                # Get vocals and convert to mono
                vocals = sources[3]  # Index 3 is vocals in Demucs
                if vocals.shape[0] > 1:
                    vocals = vocals.mean(0, keepdim=True)
                
                # Resample to target sample rate
                if self.sample_rate != TARGET_SAMPLE_RATE:
                    resampler = torchaudio.transforms.Resample(
                        self.sample_rate, TARGET_SAMPLE_RATE
                    ).to(self.device)
                    vocals = resampler(vocals)
                
                # Save output
                vocals = vocals.cpu()
                success = save_audio(vocals, TARGET_SAMPLE_RATE, output_path)
                
                # Clean up
                del sources, vocals
                torch.cuda.empty_cache()
                
                return success
                
        except Exception as e:
            print(f"Error processing {audio_path}: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

def process_file(args):
    """Process a single file - used for parallel processing"""
    audio_file, output_base_dir, input_base_dir, gpu_id = args
    
    try:
        # Calculate relative path to maintain folder structure
        rel_path = os.path.relpath(audio_file, input_base_dir)
        
        # Create output path with same folder structure
        output_path = os.path.join(
            output_base_dir, 
            os.path.dirname(rel_path),
            f"{os.path.splitext(os.path.basename(audio_file))[0]}_vocals.mp3"
        )
        
        # Create extractor
        extractor = DemucsVocalExtractor(gpu_id=gpu_id)
        
        # Process file
        success = extractor.extract_vocals(audio_file, output_path)
        
        # Clean up
        del extractor
        gc.collect()
        torch.cuda.empty_cache()
        
        return output_path if success else None
        
    except Exception as e:
        print(f"Error in process_file for {audio_file}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def process_batch(file_list, input_base_dir, output_base_dir, num_gpus=1, max_workers=None):
    """Process a batch of files in parallel"""
    # Set max workers based on CPU count if not specified
    if max_workers is None:
        max_workers = min(32, os.cpu_count() * 2)  # Limit to reasonable number
    
    # Prepare arguments for each file processing task
    tasks = []
    for i, file_path in enumerate(file_list):
        gpu_id = i % num_gpus  # Distribute tasks across available GPUs
        tasks.append((file_path, output_base_dir, input_base_dir, gpu_id))
    
    # Process files in parallel
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_file, task) for task in tasks]
        
        # Show progress bar
        for future in tqdm(
            concurrent.futures.as_completed(futures), 
            total=len(futures),
            desc="Processing files"
        ):
            try:
                result = future.result()
                if result:
                    results.append(result)
            except Exception as e:
                print(f"Error processing task: {str(e)}")
    
    return results

if __name__ == "__main__":
    # Set up multiprocessing for CUDA
    mp.set_start_method('spawn', force=True)
    
    parser = argparse.ArgumentParser(description="Extract vocals from multiple audio files using Demucs")
    parser.add_argument("inputs", nargs='+', help="Input audio file paths or directory (use '*' for wildcards)")
    parser.add_argument("--output", help="Output directory for vocals")
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs to use (default: 1)")
    parser.add_argument("--workers", type=int, help="Number of parallel workers (default: CPU count * 2)")
    
    args = parser.parse_args()
    
    # Set output directory
    output_base_dir = args.output if args.output else OUTPUT_DIR
    
    # Ensure output directory exists
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
    
    # Process all files in parallel
    processed_files = []
    
    # Process by input base directory to maintain structure
    unique_base_dirs_set = set(input_base_dirs)
    
    for base_dir in unique_base_dirs_set:
        # Get files for this base directory
        files_for_dir = [f for f, d in zip(input_files, input_base_dirs) if d == base_dir]
        
        print(f"\nProcessing {len(files_for_dir)} files from {base_dir}")
        
        results = process_batch(
            files_for_dir, 
            base_dir,
            output_base_dir,
            num_gpus=args.gpus,
            max_workers=args.workers
        )
        
        processed_files.extend(results)
    
    print(f"\nSuccessfully processed {len(processed_files)} files") 