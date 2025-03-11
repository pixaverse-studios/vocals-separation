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
import threading
from queue import Queue

# Set multiprocessing start method to 'spawn'
mp.set_start_method('spawn', force=True)

# Set up logging - only log errors to file
logging.basicConfig(
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('vocals_separation.log'),
    ]
)

# Constants
OUTPUT_SAMPLE_RATE = 16000  # 16kHz output
OUTPUT_CHANNELS = 1  # mono output
DEFAULT_SEGMENT = 7  # Demucs v4 supports up to 7.8s segments
NUM_GPUS = 3  # Number of GPUs to use

# Shared progress tracking
progress_queue = Queue()
total_processed = mp.Value('i', 0)
start_time = None

def progress_monitor(total_files):
    """Monitor progress across all GPUs"""
    pbar = tqdm(total=total_files, desc="Processing files", unit="file")
    while True:
        val = progress_queue.get()
        if val == "DONE":
            break
        with total_processed.get_lock():
            total_processed.value += 1
            pbar.update(1)
            # Calculate ETA
            if total_processed.value > 0 and start_time is not None:
                elapsed = time.time() - start_time
                rate = total_processed.value / elapsed
                eta = (total_files - total_processed.value) / rate
                hours = int(eta // 3600)
                minutes = int((eta % 3600) // 60)
                pbar.set_postfix({'ETA': f'{hours}h {minutes}m', 'Speed': f'{rate:.2f} files/s'})
    pbar.close()

class VocalSeparator:
    def __init__(self, output_dir="vocal_output", model_name="htdemucs", segment=DEFAULT_SEGMENT, bitrate=128, gpu_id=0):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.bitrate = bitrate
        self.gpu_id = gpu_id
        
        # Initialize separator on specified GPU
        self.separator = demucs.api.Separator(
            model=model_name,
            segment=segment,
            device=f"cuda:{gpu_id}",
            progress=False
        )
        # Force model weight loading by doing a tiny separation
        dummy_audio = torch.zeros(2, 44100, device=f"cuda:{gpu_id}")
        self.separator.separate_tensor(dummy_audio)
    
    def process_file(self, input_file, base_input_dir=None):
        try:
            # Generate output path preserving directory structure
            input_path = Path(input_file)
            if base_input_dir:
                rel_path = input_path.relative_to(base_input_dir)
                output_path = self.output_dir / rel_path.parent / f"{rel_path.stem}_vocals.mp3"
            else:
                output_path = self.output_dir / f"{input_path.stem}_vocals.mp3"
            
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            if output_path.exists():
                return None

            # Separate vocals
            _, separated = self.separator.separate_audio_file(str(input_file))
            
            # Extract vocals and convert to mono 16kHz
            vocals = separated['vocals']
            if vocals.shape[0] > 1:
                vocals = vocals.mean(dim=0, keepdim=True)
            
            # Resample if needed
            if self.separator.samplerate != OUTPUT_SAMPLE_RATE:
                resampler = torchaudio.transforms.Resample(
                    self.separator.samplerate, 
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
            logging.error(f"Error processing {input_file}: {str(e)}")
            return None

    def process_files(self, input_files, base_input_dir=None):
        processed_files = []
        failed_files = []
        
        for input_file in input_files:
            try:
                result = self.process_file(input_file, base_input_dir)
                if result:
                    processed_files.append(result)
                else:
                    failed_files.append(str(input_file))
            except Exception as e:
                logging.error(f"Error processing {input_file}: {str(e)}")
                failed_files.append(str(input_file))
            finally:
                progress_queue.put(1)
        
        if failed_files:
            failed_file_path = self.output_dir / f"failed_files_gpu{self.gpu_id}.txt"
            with open(failed_file_path, 'w') as f:
                for file in failed_files:
                    f.write(f"{file}\n")
        
        return processed_files, failed_files

def process_gpu_batch(gpu_id, files, output_dir, model_name, segment, bitrate, base_input_dir):
    separator = VocalSeparator(
        output_dir=output_dir,
        model_name=model_name,
        segment=segment,
        bitrate=bitrate,
        gpu_id=gpu_id
    )
    return separator.process_files(files, base_input_dir)

def split_list(lst, n):
    k, m = divmod(len(lst), n)
    return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]

def main():
    global start_time
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
    
    # Collect all input files
    input_files = []
    base_input_dir = None
    
    for input_path in args.inputs:
        path = Path(input_path)
        if path.is_dir():
            base_input_dir = path
            for ext in ['mp3', 'wav', 'flac', 'ogg', 'm4a']:
                input_files.extend(path.rglob(f"*.{ext}"))
        else:
            input_files.append(path)
    
    input_files = list(dict.fromkeys(input_files))
    
    if not input_files:
        print("No input files found.")
        return
    
    total_files = len(input_files)
    print(f"Found {total_files} files to process")
    
    # Start progress monitoring thread
    monitor_thread = threading.Thread(target=progress_monitor, args=(total_files,))
    monitor_thread.daemon = True
    monitor_thread.start()
    
    # Split files into GPU groups
    file_groups = split_list(input_files, NUM_GPUS)
    
    # Start timing
    start_time = time.time()
    
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
    
    # Signal progress monitor to finish
    progress_queue.put("DONE")
    monitor_thread.join()
    
    # Print final statistics
    total_time = time.time() - start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)
    files_per_second = total_files / total_time
    
    print(f"\nProcessing complete in {hours}h {minutes}m {seconds}s")
    print(f"Average processing speed: {files_per_second:.2f} files/second")
    
    # Combine failed files
    all_failed_files = []
    for gpu_id in range(NUM_GPUS):
        failed_file_path = Path(args.output) / f"failed_files_gpu{gpu_id}.txt"
        if failed_file_path.exists():
            with open(failed_file_path) as f:
                all_failed_files.extend(f.readlines())
    
    if all_failed_files:
        with open(Path(args.output) / "failed_files.txt", 'w') as f:
            f.writelines(all_failed_files)
        print(f"Failed to process {len(all_failed_files)} files")
        print("Failed files have been saved to failed_files.txt")

if __name__ == "__main__":
    mp.freeze_support()
    main() 