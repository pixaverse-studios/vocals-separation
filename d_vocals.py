import os
import gc
import torch
import torchaudio
import concurrent.futures
import numpy as np
from pathlib import Path
from tqdm import tqdm
import demucs.api
import json
import logging
import time
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('vocals_separation.log'),
        logging.StreamHandler()
    ]
)

# Constants
OUTPUT_SAMPLE_RATE = 16000  # 16kHz output
OUTPUT_CHANNELS = 1  # mono output
DEFAULT_SEGMENT = 7  # Demucs v4 supports up to 7.8s segments
NUM_GPUS = 3  # Number of available GPUs
MAX_WORKERS = 32  # Reduced number of workers
BATCH_SIZE = 1000  # Process files in batches
MAX_RETRIES = 3  # Maximum number of retries for failed files

class VocalSeparator:
    def __init__(self, output_dir="vocal_output", model_name="htdemucs", segment=DEFAULT_SEGMENT, bitrate=128):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.bitrate = bitrate
        self.progress_file = self.output_dir / "progress.json"
        
        # Load progress if exists
        self.processed_files = self._load_progress()
        
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
        
    def _load_progress(self):
        """Load progress from progress file"""
        if self.progress_file.exists():
            try:
                with open(self.progress_file) as f:
                    return set(json.load(f))
            except Exception as e:
                logging.warning(f"Failed to load progress file: {e}")
        return set()
    
    def _save_progress(self):
        """Save progress to progress file"""
        try:
            with open(self.progress_file, 'w') as f:
                json.dump(list(self.processed_files), f)
        except Exception as e:
            logging.error(f"Failed to save progress: {e}")
    
    def _get_next_gpu(self):
        """Round-robin GPU selection"""
        gpu_id = self.current_gpu
        self.current_gpu = (self.current_gpu + 1) % NUM_GPUS
        return gpu_id
    
    def _process_file(self, input_file):
        """Process a single file using the next available GPU"""
        input_file = str(input_file)
        if input_file in self.processed_files:
            return None
            
        retries = 0
        while retries < MAX_RETRIES:
            try:
                # Get next available GPU and its separator
                gpu_id = self._get_next_gpu()
                separator = self.separators[gpu_id]
                
                # Generate output path
                rel_path = Path(input_file).stem
                output_path = self.output_dir / f"{rel_path}_vocals.mp3"
                
                # Skip if output already exists
                if output_path.exists():
                    self.processed_files.add(input_file)
                    self._save_progress()
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
                
                # Mark as processed
                self.processed_files.add(input_file)
                self._save_progress()
                
                return str(output_path)
                
            except Exception as e:
                retries += 1
                error_msg = f"Error processing {input_file} (attempt {retries}/{MAX_RETRIES}): {str(e)}"
                if retries < MAX_RETRIES:
                    logging.warning(error_msg)
                    # Reset GPU state
                    torch.cuda.empty_cache()
                    time.sleep(1)  # Wait a bit before retrying
                else:
                    logging.error(error_msg)
                    return None
        
        return None

    def process_batch(self, batch_files):
        """Process a batch of files"""
        results = []
        failed_files = []
        
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
                    logging.error(f"Batch processing error for {input_file}: {str(e)}")
                    failed_files.append(input_file)
        
        return results, failed_files

    def process_files(self, input_files):
        """Process all files in batches"""
        all_results = []
        all_failed = []
        total_batches = (len(input_files) + BATCH_SIZE - 1) // BATCH_SIZE
        
        # Filter out already processed files
        input_files = [f for f in input_files if str(f) not in self.processed_files]
        if not input_files:
            logging.info("All files have been processed already")
            return [], []
        
        logging.info(f"Starting processing of {len(input_files)} files")
        with tqdm(total=len(input_files), desc="Processing files") as pbar:
            for i in range(0, len(input_files), BATCH_SIZE):
                batch = input_files[i:i + BATCH_SIZE]
                try:
                    results, failed = self.process_batch(batch)
                    all_results.extend(results)
                    all_failed.extend(failed)
                    pbar.update(len(batch))
                except Exception as e:
                    logging.error(f"Error processing batch starting at index {i}: {str(e)}")
                finally:
                    # Force garbage collection between batches
                    gc.collect()
                    torch.cuda.empty_cache()
                    # Save progress after each batch
                    self._save_progress()
        
        if all_failed:
            logging.warning(f"Failed to process {len(all_failed)} files")
            with open(self.output_dir / "failed_files.txt", 'w') as f:
                for file in all_failed:
                    f.write(f"{file}\n")
        
        return all_results, all_failed

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
    parser.add_argument("--retry", action="store_true",
                       help="Retry processing failed files from previous run")
    
    args = parser.parse_args()
    
    # Update constants based on args
    BATCH_SIZE = args.batch_size
    MAX_WORKERS = args.workers
    
    # If retrying failed files
    if args.retry:
        failed_files_path = Path(args.output) / "failed_files.txt"
        if failed_files_path.exists():
            with open(failed_files_path) as f:
                input_files = [line.strip() for line in f]
            logging.info(f"Retrying {len(input_files)} failed files")
        else:
            logging.error("No failed files found to retry")
            return
    else:
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
        logging.error("No input files found.")
        return
    
    logging.info(f"Found {len(input_files)} files to process")
    logging.info(f"Processing in batches of {BATCH_SIZE} files with {MAX_WORKERS} workers")
    
    # Initialize separator and process files
    separator = VocalSeparator(
        output_dir=args.output,
        model_name=args.model,
        segment=args.segment,
        bitrate=args.bitrate
    )
    
    processed_files, failed_files = separator.process_files(input_files)
    
    logging.info(f"\nProcessing complete:")
    logging.info(f"Successfully processed: {len(processed_files)} files")
    if failed_files:
        logging.warning(f"Failed to process: {len(failed_files)} files")
        logging.info("Failed files have been saved to failed_files.txt")
        logging.info("You can retry failed files using the --retry flag")

if __name__ == "__main__":
    main() 