import os
import gc
import torch
import torchaudio
from pathlib import Path
from tqdm import tqdm
import demucs.api
import logging
import time
import multiprocessing as mp
from itertools import islice

# Set multiprocessing start method to 'spawn'
mp.set_start_method('spawn', force=True)

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
NUM_GPUS = 3  # Number of GPUs to use

class VocalSeparator:
    def __init__(self, output_dir="vocal_output", model_name="htdemucs", segment=DEFAULT_SEGMENT, bitrate=128, gpu_id=0):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.bitrate = bitrate
        self.gpu_id = gpu_id
        
        logging.info(f"[GPU {gpu_id}] Loading model weights...")
        start_time = time.time()
        # Initialize separator on specified GPU
        self.separator = demucs.api.Separator(
            model=model_name,
            segment=segment,
            device=f"cuda:{gpu_id}",
            progress=False
        )
        # Force model weight loading by doing a tiny separation
        dummy_audio = torch.zeros(2, 44100, device=f"cuda:{gpu_id}")  # 1 second of silence
        self.separator.separate_tensor(dummy_audio)
        logging.info(f"[GPU {gpu_id}] Model weights loaded successfully in {time.time() - start_time:.2f} seconds")
    
    def process_file(self, input_file, base_input_dir=None):
        """Process a single file while preserving directory structure"""
        try:
            # Generate output path preserving directory structure
            input_path = Path(input_file)
            if base_input_dir:
                # Get the relative path from base_input_dir
                rel_path = input_path.relative_to(base_input_dir)
                # Create output path with preserved structure
                output_path = self.output_dir / rel_path.parent / f"{rel_path.stem}_vocals.mp3"
            else:
                # If no base_input_dir, just use the filename
                output_path = self.output_dir / f"{input_path.stem}_vocals.mp3"
            
            # Create parent directories if they don't exist
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Skip if output already exists
            if output_path.exists():
                return None

            # Time the entire process
            total_start = time.time()
            
            # Time loading audio
            load_start = time.time()
            # Separate vocals
            _, separated = self.separator.separate_audio_file(str(input_file))
            load_time = time.time() - load_start
            
            # Time post-processing (mono conversion and resampling)
            post_start = time.time()
            # Extract vocals and convert to mono 16kHz
            vocals = separated['vocals']
            if vocals.shape[0] > 1:  # If stereo, convert to mono
                vocals = vocals.mean(dim=0, keepdim=True)
            
            # Resample if needed
            if self.separator.samplerate != OUTPUT_SAMPLE_RATE:
                resampler = torchaudio.transforms.Resample(
                    self.separator.samplerate, 
                    OUTPUT_SAMPLE_RATE
                ).to(vocals.device)
                vocals = resampler(vocals)
            post_time = time.time() - post_start
            
            # Time saving
            save_start = time.time()
            # Save as MP3
            demucs.api.save_audio(
                vocals,
                str(output_path),
                samplerate=OUTPUT_SAMPLE_RATE,
                bitrate=self.bitrate
            )
            save_time = time.time() - save_start
            
            # Time cleanup
            cleanup_start = time.time()
            # Clear memory
            del vocals, separated
            torch.cuda.empty_cache()
            cleanup_time = time.time() - cleanup_start
            
            total_time = time.time() - total_start
            
            # Log timing information
            logging.info(f"\n[GPU {self.gpu_id}] Processing times for {input_path.name}:")
            logging.info(f"[GPU {self.gpu_id}]   Loading and separation: {load_time:.2f}s")
            logging.info(f"[GPU {self.gpu_id}]   Post-processing: {post_time:.2f}s")
            logging.info(f"[GPU {self.gpu_id}]   Saving: {save_time:.2f}s")
            logging.info(f"[GPU {self.gpu_id}]   Cleanup: {cleanup_time:.2f}s")
            logging.info(f"[GPU {self.gpu_id}]   Total time: {total_time:.2f}s")
            
            return str(output_path)
            
        except Exception as e:
            logging.error(f"[GPU {self.gpu_id}] Error processing {input_file}: {str(e)}")
            return None

    def process_files(self, input_files, base_input_dir=None):
        """Process all files sequentially"""
        processed_files = []
        failed_files = []
        
        with tqdm(total=len(input_files), desc=f"[GPU {self.gpu_id}] Processing files") as pbar:
            for input_file in input_files:
                try:
                    result = self.process_file(input_file, base_input_dir)
                    if result:
                        processed_files.append(result)
                    else:
                        failed_files.append(str(input_file))
                except Exception as e:
                    logging.error(f"[GPU {self.gpu_id}] Error processing {input_file}: {str(e)}")
                    failed_files.append(str(input_file))
                finally:
                    pbar.update(1)
                    gc.collect()
                    torch.cuda.empty_cache()
        
        if failed_files:
            failed_file_path = self.output_dir / f"failed_files_gpu{self.gpu_id}.txt"
            with open(failed_file_path, 'w') as f:
                for file in failed_files:
                    f.write(f"{file}\n")
        
        return processed_files, failed_files

def process_gpu_batch(gpu_id, files, output_dir, model_name, segment, bitrate, base_input_dir):
    """Function to run in each process"""
    separator = VocalSeparator(
        output_dir=output_dir,
        model_name=model_name,
        segment=segment,
        bitrate=bitrate,
        gpu_id=gpu_id
    )
    return separator.process_files(files, base_input_dir)

def split_list(lst, n):
    """Split a list into n roughly equal parts"""
    k, m = divmod(len(lst), n)
    return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]

def main():
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
    
    args = parser.parse_args()
    
    # Collect all input files and their base directories
    input_files = []
    base_input_dir = None
    
    for input_path in args.inputs:
        path = Path(input_path)
        if path.is_dir():
            base_input_dir = path  # Use the input directory as base
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
    
    # Split files into NUM_GPUS groups
    file_groups = split_list(input_files, NUM_GPUS)
    
    # Create processes for each GPU
    processes = []
    for gpu_id in range(NUM_GPUS):
        p = mp.Process(
            target=process_gpu_batch,
            args=(
                gpu_id,
                file_groups[gpu_id],
                args.output,
                args.model,
                args.segment,
                args.bitrate,
                base_input_dir
            )
        )
        processes.append(p)
        p.start()
    
    # Wait for all processes to complete
    for p in processes:
        p.join()
    
    logging.info("\nAll processes completed")
    
    # Combine failed files from all GPUs
    all_failed_files = []
    for gpu_id in range(NUM_GPUS):
        failed_file_path = Path(args.output) / f"failed_files_gpu{gpu_id}.txt"
        if failed_file_path.exists():
            with open(failed_file_path) as f:
                all_failed_files.extend(f.readlines())
    
    if all_failed_files:
        with open(Path(args.output) / "failed_files.txt", 'w') as f:
            f.writelines(all_failed_files)
        logging.warning(f"Failed to process {len(all_failed_files)} files")
        logging.info("Failed files have been saved to failed_files.txt")

if __name__ == "__main__":
    # Required for Windows support
    mp.freeze_support()
    main() 