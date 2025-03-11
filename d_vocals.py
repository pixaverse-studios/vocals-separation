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
OUTPUT_SAMPLE_RATE = 16000  # 16kHz output
OUTPUT_CHANNELS = 1  # mono output
MP3_BITRATE = 96  # kbps, good balance for voice
SEGMENT_LENGTH = 7  # seconds, optimal for htdemucs
BASE_DIR = "."
OUTPUT_DIR = os.path.join(BASE_DIR, "vocal_output")

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

class VocalExtractor:
    def __init__(self, model_name='htdemucs', device_ids=None, segment_length=SEGMENT_LENGTH):
        """Initialize the vocal extractor with specified model and devices."""
        self.model_name = model_name
        self.segment_length = segment_length
        self.device_ids = device_ids if device_ids else list(range(torch.cuda.device_count()))
        self.num_gpus = len(self.device_ids) if self.device_ids else 0
        
        # Load model on CPU first
        self.model = get_model(model_name)
        self.model.eval()
        
        # Initialize GPU queue if available
        if self.num_gpus > 0:
            self.gpu_queue = mp.Queue()
            for device_id in self.device_ids:
                self.gpu_queue.put(device_id)
    
    def get_device(self):
        """Get next available GPU or CPU."""
        if self.num_gpus > 0:
            device_id = self.gpu_queue.get()
            device = torch.device(f'cuda:{device_id}')
            return device, device_id
        return torch.device('cpu'), None
    
    def release_device(self, device_id):
        """Release GPU back to queue."""
        if device_id is not None:
            self.gpu_queue.put(device_id)
    
    def load_audio(self, audio_path):
        """Load audio file and resample if necessary."""
        try:
            waveform, sample_rate = torchaudio.load(audio_path)
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Resample if necessary
            if sample_rate != self.model.samplerate:
                waveform = torchaudio.transforms.Resample(
                    sample_rate, self.model.samplerate
                )(waveform)
            
            return waveform, self.model.samplerate
        except Exception as e:
            print(f"Error loading {audio_path}: {str(e)}")
            return None, None

    def save_audio(self, waveform, output_path, original_sr):
        """Save audio as MP3 with specified sample rate and channels."""
        try:
            # Resample to target sample rate
            if original_sr != OUTPUT_SAMPLE_RATE:
                waveform = torchaudio.transforms.Resample(
                    original_sr, OUTPUT_SAMPLE_RATE
                )(waveform)
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Save as MP3
            torchaudio.save(
                output_path,
                waveform,
                OUTPUT_SAMPLE_RATE,
                format='mp3',
                encoding_args={'bit_rate': MP3_BITRATE * 1000}
            )
            return True
        except Exception as e:
            print(f"Error saving {output_path}: {str(e)}")
            return False

    def process_segment(self, waveform, device, offset=0, total_length=None):
        """Process a single segment of audio."""
        try:
            # Move model to device if needed
            if next(self.model.parameters()).device != device:
                self.model.to(device)
            
            # Apply model and extract vocals
            with torch.no_grad():
                sources = apply_model(
                    self.model,
                    waveform.to(device),
                    device=device,
                    split=True,
                    overlap=0.1,
                    progress=False
                )
                
                # Extract vocals (usually the last source)
                vocals = sources[-1]
                
                # Move back to CPU to free GPU memory
                vocals = vocals.cpu()
                
            return vocals
            
        except Exception as e:
            print(f"Error processing segment: {str(e)}")
            return None

    def process_audio(self, audio_path, output_path=None):
        """Process a single audio file."""
        try:
            # Setup output path
            if output_path is None:
                filename = os.path.basename(audio_path)
                output_path = os.path.join(
                    OUTPUT_DIR,
                    f"{os.path.splitext(filename)[0]}_vocals.mp3"
                )
            
            # Load audio
            waveform, sr = self.load_audio(audio_path)
            if waveform is None:
                return None
            
            # Get device
            device, device_id = self.get_device()
            
            try:
                # Process audio
                vocals = self.process_segment(waveform, device)
                if vocals is None:
                    return None
                
                # Save result
                success = self.save_audio(vocals, output_path, sr)
                
                # Clean up
                del vocals
                torch.cuda.empty_cache()
                
                return output_path if success else None
                
            finally:
                # Always release device
                self.release_device(device_id)
                
        except Exception as e:
            print(f"Error processing {audio_path}: {str(e)}")
            return None

def process_batch(args):
    """Process a batch of files using multiple GPUs."""
    file_list, output_base_dir, input_base_dir, gpu_ids = args
    
    # Initialize extractor for this process
    extractor = VocalExtractor(device_ids=gpu_ids)
    
    results = []
    for audio_file in tqdm(file_list, desc="Processing files", leave=False):
        try:
            # Calculate relative path to maintain folder structure
            rel_path = os.path.relpath(audio_file, input_base_dir)
            
            # Create output path with same folder structure
            output_path = os.path.join(
                output_base_dir,
                os.path.dirname(rel_path),
                f"{os.path.splitext(os.path.basename(audio_file))[0]}_vocals.mp3"
            )
            
            result = extractor.process_audio(audio_file, output_path)
            if result:
                results.append(result)
                
        except Exception as e:
            print(f"Error processing {audio_file}: {str(e)}")
            continue
    
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
        # Assign GPUs round-robin to batches
        batch_gpu_ids = [i % num_gpus] if num_gpus > 0 else None
        batches.append((batch_files, output_base_dir, input_base_dirs[0], batch_gpu_ids))
    
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
    # For better multiprocessing on Unix
    mp.set_start_method('spawn', force=True)
    main() 