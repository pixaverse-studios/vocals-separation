import os
import gc
import torch
import torchaudio
import concurrent.futures
import numpy as np
from pathlib import Path
from tqdm import tqdm
import demucs.api

# Constants
OUTPUT_SAMPLE_RATE = 16000  # 16kHz output
OUTPUT_CHANNELS = 1  # mono output
DEFAULT_SEGMENT = 7  # Demucs v4 supports up to 7.8s segments
NUM_GPUS = 3  # Number of available GPUs
MAX_WORKERS = 32  # Reduced number of workers
BATCH_SIZE = 1000  # Process files in batches

class VocalSeparator:
    def __init__(self, output_dir="vocal_output", model_name="htdemucs", segment=DEFAULT_SEGMENT, bitrate=128):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.bitrate = bitrate
        
        # Initialize separators for each GPU
        self.separators = []
        for gpu_id in range(NUM_GPUS):
            device = f"cuda:{gpu_id}"
            separator = demucs.api.Separator(
                model=model_name,
                segment=segment,
                device=device,
                progress=False
            )
            self.separators.append(separator)
        
        self.current_gpu = 0
        
    def _get_next_gpu(self):
        """Round-robin GPU selection"""
        gpu_id = self.current_gpu
        self.current_gpu = (self.current_gpu + 1) % NUM_GPUS
        return gpu_id
    
    def _process_file(self, input_file):
        """Process a single file using the next available GPU"""
        try:
            # Get next available GPU and its separator
            gpu_id = self._get_next_gpu()
            separator = self.separators[gpu_id]
            
            # Generate output path
            rel_path = Path(input_file).stem
            output_path = self.output_dir / f"{rel_path}_vocals.mp3"
            
            # Skip if output already exists
            if output_path.exists():
                return None
            
            # Separate vocals
            _, separated = separator.separate_audio_file(str(input_file))
            
            # Extract vocals and convert to mono 16kHz
            vocals = separated['vocals']
            if vocals.shape[0] > 1:  # If stereo, convert to mono
                vocals = vocals.mean(dim=0, keepdim=True)
            
            # Resample if needed
            if separator.samplerate != OUTPUT_SAMPLE_RATE:
                resampler = torchaudio.transforms.Resample(
                    separator.samplerate, 
                    OUTPUT_SAMPLE_RATE
                ).to(vocals.device)
                vocals = resampler(vocals)
            
            # Save as MP3
            demucs.api.save_audio(
                vocals,
                str(output_path),
                samplerate=OUTPUT_SAMPLE_RATE,
                bitrate=self.bitrate
            )
            
            # Clear memory
            del vocals, separated
            torch.cuda.empty_cache()
            
            return str(output_path)
            
        except Exception as e:
            print(f"Error processing {input_file}: {str(e)}")
            return None

    def process_batch(self, batch_files):
        """Process a batch of files"""
        results = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_file = {
                executor.submit(self._process_file, input_file): input_file 
                for input_file in batch_files
            }
            
            for future in concurrent.futures.as_completed(future_to_file):
                input_file = future_to_file[future]
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                except Exception as e:
                    print(f"Error processing {input_file}: {str(e)}")
                
        return results

    def process_files(self, input_files):
        """Process all files in batches"""
        all_results = []
        total_batches = (len(input_files) + BATCH_SIZE - 1) // BATCH_SIZE
        
        with tqdm(total=len(input_files), desc="Processing files") as pbar:
            for i in range(0, len(input_files), BATCH_SIZE):
                batch = input_files[i:i + BATCH_SIZE]
                results = self.process_batch(batch)
                all_results.extend(results)
                pbar.update(len(batch))
                
                # Force garbage collection between batches
                gc.collect()
                torch.cuda.empty_cache()
        
        return all_results

def main():
    global BATCH_SIZE, MAX_WORKERS
    
    import argparse
    parser = argparse.ArgumentParser(description="Extract vocals using Demucs v4")
    parser.add_argument("inputs", nargs="+", help="Input audio files or directories")
    parser.add_argument("--output", default="vocal_output", help="Output directory")
    parser.add_argument("--segment", type=float, default=DEFAULT_SEGMENT, 
                       help="Segment length in seconds (max 7.8)")
    parser.add_argument("--model", default="htdemucs", 
                       help="Model to use (htdemucs, htdemucs_ft)")
    parser.add_argument("--bitrate", type=int, default=128,
                       help="MP3 output bitrate in kbps (default: 128)")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                       help=f"Number of files to process in each batch (default: {BATCH_SIZE})")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS,
                       help=f"Number of parallel workers (default: {MAX_WORKERS})")
    
    args = parser.parse_args()
    
    # Update constants based on args
    BATCH_SIZE = args.batch_size
    MAX_WORKERS = args.workers
    
    # Collect all input files
    input_files = []
    for input_path in args.inputs:
        path = Path(input_path)
        if path.is_dir():
            for ext in ['mp3', 'wav', 'flac', 'ogg', 'm4a']:
                input_files.extend(path.rglob(f"*.{ext}"))
        else:
            input_files.append(path)
    
    # Remove duplicates while preserving order
    input_files = list(dict.fromkeys(input_files))
    
    if not input_files:
        print("No input files found.")
        return
    
    print(f"Found {len(input_files)} files to process")
    print(f"Processing in batches of {BATCH_SIZE} files with {MAX_WORKERS} workers")
    
    # Initialize separator and process files
    separator = VocalSeparator(
        output_dir=args.output,
        model_name=args.model,
        segment=args.segment,
        bitrate=args.bitrate
    )
    
    processed_files = separator.process_files(input_files)
    print(f"\nSuccessfully processed {len(processed_files)} files")

if __name__ == "__main__":
    main() 