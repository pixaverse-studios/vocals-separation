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
from datetime import datetime, timedelta

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
BATCH_SIZE = 4  # Number of files to process at once

class VocalSeparator:
    def __init__(self, output_dir="vocal_output", model_name="htdemucs", segment=DEFAULT_SEGMENT, bitrate=128, gpu_id=0):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.bitrate = bitrate
        self.gpu_id = gpu_id
        
        logging.info(f"[GPU {gpu_id}] Loading model weights...")
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
        logging.info(f"[GPU {gpu_id}] Model ready")
    
    def process_batch(self, batch_files, base_input_dir=None):
        """Process a batch of files together"""
        try:
            # Load all audio files in batch
            audios = []
            output_paths = []
            valid_files = []
            max_length = 0
            
            # First pass: load audio and find max length
            for input_file in batch_files:
                try:
                    # Generate output path
                    input_path = Path(input_file)
                    if base_input_dir:
                        rel_path = input_path.relative_to(base_input_dir)
                        output_path = self.output_dir / rel_path.parent / f"{rel_path.stem}_vocals.mp3"
                    else:
                        output_path = self.output_dir / f"{input_path.stem}_vocals.mp3"
                    
                    # Skip if output exists
                    if output_path.exists():
                        continue
                        
                    # Create parent directories
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Load audio
                    audio, sr = torchaudio.load(str(input_file))
                    max_length = max(max_length, audio.shape[-1])
                    
                    # Keep track of valid files
                    audios.append(audio)
                    output_paths.append(output_path)
                    valid_files.append(input_file)
                    
                except Exception as e:
                    logging.error(f"[GPU {self.gpu_id}] Error loading {input_file}: {str(e)}")
                    continue
            
            if not valid_files:
                return [], valid_files
            
            # Second pass: pad audios to same length
            padded_audios = []
            for audio in audios:
                if audio.shape[-1] < max_length:
                    padding = max_length - audio.shape[-1]
                    audio = torch.nn.functional.pad(audio, (0, padding))
                padded_audios.append(audio)
            
            # Stack into batch and process
            audio_batch = torch.stack(padded_audios).to(f"cuda:{self.gpu_id}")
            separated_batch = self.separator.separate_tensor(audio_batch)
            
            # Process and save each result
            processed_files = []
            for i, (vocals, output_path) in enumerate(zip(separated_batch['vocals'], output_paths)):
                try:
                    # Convert to mono if stereo
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
                    processed_files.append(str(output_path))
                    
                except Exception as e:
                    logging.error(f"[GPU {self.gpu_id}] Error saving {output_path}: {str(e)}")
                    continue
            
            # Clear memory
            del audio_batch, separated_batch, padded_audios, audios
            torch.cuda.empty_cache()
            
            return processed_files, valid_files
            
        except Exception as e:
            logging.error(f"[GPU {self.gpu_id}] Batch processing error: {str(e)}")
            return [], batch_files

    def process_files(self, input_files, base_input_dir=None):
        """Process files in batches"""
        processed_files = []
        failed_files = []
        
        total_files = len(input_files)
        files_per_gpu = total_files // NUM_GPUS
        estimated_time_per_file = 1.0  # seconds
        total_estimated_time = timedelta(seconds=files_per_gpu * estimated_time_per_file)
        estimated_completion = datetime.now() + total_estimated_time
        
        logging.info(f"[GPU {self.gpu_id}] Processing {len(input_files)} files in batches of {BATCH_SIZE}. Estimated completion: {estimated_completion.strftime('%H:%M:%S')}")
        
        with tqdm(total=len(input_files), desc=f"[GPU {self.gpu_id}]") as pbar:
            # Process files in batches
            for i in range(0, len(input_files), BATCH_SIZE):
                batch_files = input_files[i:i + BATCH_SIZE]
                batch_processed, batch_attempted = self.process_batch(batch_files, base_input_dir)
                
                # Update progress and track results
                processed_files.extend(batch_processed)
                failed_files.extend([f for f in batch_attempted if f not in batch_processed])
                pbar.update(len(batch_attempted))
        
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