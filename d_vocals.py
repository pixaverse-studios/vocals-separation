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
        
        # Add batch dimension for Demucs (batch, channels, length)
        waveform = waveform.unsqueeze(0)
        
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
    def __init__(self, gpu_id=0, segment_size=7.8, batch_size=4):
        self.device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        
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
    
    def load_audio_batch(self, audio_paths):
        """Load a batch of audio files"""
        waveforms = []
        max_length = 0
        valid_indices = []
        
        # First pass: load all files and find max length
        for idx, path in enumerate(audio_paths):
            try:
                waveform, sr = load_audio(path, self.device)
                if waveform is not None:
                    max_length = max(max_length, waveform.shape[-1])
                    waveforms.append(waveform)
                    valid_indices.append(idx)
            except Exception as e:
                print(f"Error loading {path}: {str(e)}")
                continue
        
        if not waveforms:
            return None, []
        
        # Pad all waveforms to max length
        padded_waveforms = []
        for waveform in waveforms:
            if waveform.shape[-1] < max_length:
                padding = max_length - waveform.shape[-1]
                padded = torch.nn.functional.pad(waveform, (0, padding))
                padded_waveforms.append(padded)
            else:
                padded_waveforms.append(waveform)
        
        # Stack into batch
        batch = torch.cat(padded_waveforms, dim=0)
        return batch, valid_indices
    
    def extract_vocals_batch(self, audio_paths, output_paths):
        """Extract vocals from a batch of audio files"""
        try:
            # Load audio batch
            waveform_batch, valid_indices = self.load_audio_batch(audio_paths)
            if waveform_batch is None:
                return []
            
            # Convert to model's sample rate if needed
            if TARGET_SAMPLE_RATE != self.sample_rate:
                resampler = torchaudio.transforms.Resample(
                    TARGET_SAMPLE_RATE, self.sample_rate
                ).to(self.device)
                waveform_batch = resampler(waveform_batch)
            
            # Process audio
            with torch.no_grad():
                # Apply model with segments
                sources = apply_model(
                    self.model,
                    waveform_batch,
                    device=self.device,
                    segment=self.segment_samples,
                    overlap=0.1
                )
                
                # Get vocals and convert to mono
                vocals_batch = sources[:, 3]  # Index 3 is vocals in Demucs
                if vocals_batch.shape[1] > 1:  # If stereo
                    vocals_batch = vocals_batch.mean(1, keepdim=True)
                
                # Resample to target sample rate if needed
                if self.sample_rate != TARGET_SAMPLE_RATE:
                    resampler = torchaudio.transforms.Resample(
                        self.sample_rate, TARGET_SAMPLE_RATE
                    ).to(self.device)
                    vocals_batch = resampler(vocals_batch)
                
                # Process each item in batch
                successful_paths = []
                for i, vocals in enumerate(vocals_batch):
                    output_path = output_paths[valid_indices[i]]
                    # Save output
                    vocals = vocals.cpu()
                    success = save_audio(vocals, TARGET_SAMPLE_RATE, output_path)
                    if success:
                        successful_paths.append(output_path)
                
                # Clean up
                del sources, vocals_batch
                torch.cuda.empty_cache()
                
                return successful_paths
                
        except Exception as e:
            print(f"Error processing batch: {str(e)}")
            import traceback
            traceback.print_exc()
            return []

def process_file_batch(args):
    """Process a batch of files"""
    audio_files, output_base_dir, input_base_dir, gpu_id = args
    
    try:
        # Create output paths
        output_paths = []
        for audio_file in audio_files:
            rel_path = os.path.relpath(audio_file, input_base_dir)
            output_path = os.path.join(
                output_base_dir, 
                os.path.dirname(rel_path),
                f"{os.path.splitext(os.path.basename(audio_file))[0]}_vocals.mp3"
            )
            output_paths.append(output_path)
        
        # Create extractor
        extractor = DemucsVocalExtractor(gpu_id=gpu_id)
        
        # Process batch
        successful_paths = extractor.extract_vocals_batch(audio_files, output_paths)
        
        # Clean up
        del extractor
        gc.collect()
        torch.cuda.empty_cache()
        
        return successful_paths
        
    except Exception as e:
        print(f"Error in process_file_batch: {str(e)}")
        import traceback
        traceback.print_exc()
        return []

def process_batch(file_list, input_base_dir, output_base_dir, num_gpus=1, max_workers=None, batch_size=4):
    """Process files in batches"""
    # Set max workers based on CPU count if not specified
    if max_workers is None:
        max_workers = min(32, os.cpu_count() * 2)  # Limit to reasonable number
    
    # Create batches of files
    batches = [file_list[i:i + batch_size] for i in range(0, len(file_list), batch_size)]
    
    # Prepare arguments for each batch processing task
    tasks = []
    for i, batch in enumerate(batches):
        gpu_id = i % num_gpus  # Distribute tasks across available GPUs
        tasks.append((batch, output_base_dir, input_base_dir, gpu_id))
    
    # Process batches in parallel
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_file_batch, task) for task in tasks]
        
        # Show progress bar
        for future in tqdm(
            concurrent.futures.as_completed(futures), 
            total=len(futures),
            desc="Processing batches"
        ):
            try:
                batch_results = future.result()
                results.extend(batch_results)
            except Exception as e:
                print(f"Error processing batch task: {str(e)}")
    
    return results

if __name__ == "__main__":
    # Set up multiprocessing for CUDA
    mp.set_start_method('spawn', force=True)
    
    parser = argparse.ArgumentParser(description="Extract vocals from multiple audio files using Demucs")
    parser.add_argument("inputs", nargs='+', help="Input audio file paths or directory (use '*' for wildcards)")
    parser.add_argument("--output", help="Output directory for vocals")
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs to use (default: 1)")
    parser.add_argument("--workers", type=int, help="Number of parallel workers (default: CPU count * 2)")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for processing (default: 4)")
    
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
            max_workers=args.workers,
            batch_size=args.batch_size
        )
        
        processed_files.extend(results)
    
    print(f"\nSuccessfully processed {len(processed_files)} files") 