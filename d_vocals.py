import os
import gc
import torch
import torchaudio
from pathlib import Path
from tqdm import tqdm
import demucs.api
import logging
import concurrent.futures

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
    def __init__(self, output_dir="vocal_output", model_name="htdemucs", segment=DEFAULT_SEGMENT, bitrate=128):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.bitrate = bitrate
        
        # Initialize separators for all GPUs
        logging.info("Pre-loading model weights for all GPUs...")
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
        logging.info("Model weights loaded successfully")
    
    def process_file_on_gpu(self, input_file, gpu_id, base_input_dir=None):
        """Process a single file on a specific GPU"""
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
            
            # Get the separator for this GPU
            separator = self.separators[gpu_id]
            
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
            logging.error(f"Error processing {input_file} on GPU {gpu_id}: {str(e)}")
            return None

    def process_files(self, input_files, base_input_dir=None):
        """Process files in parallel using all GPUs"""
        processed_files = []
        failed_files = []
        
        # Create a thread pool with NUM_GPUS workers
        with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_GPUS) as executor:
            # Create futures for each file, distributing them across GPUs
            futures = []
            for i, input_file in enumerate(input_files):
                gpu_id = i % NUM_GPUS
                future = executor.submit(self.process_file_on_gpu, input_file, gpu_id, base_input_dir)
                futures.append((future, input_file))
            
            # Process results as they complete
            with tqdm(total=len(input_files), desc="Processing files") as pbar:
                for future, input_file in futures:
                    try:
                        result = future.result()
                        if result:
                            processed_files.append(result)
                        else:
                            failed_files.append(str(input_file))
                    except Exception as e:
                        logging.error(f"Error processing {input_file}: {str(e)}")
                        failed_files.append(str(input_file))
                    finally:
                        pbar.update(1)
                        gc.collect()
        
        if failed_files:
            with open(self.output_dir / "failed_files.txt", 'w') as f:
                for file in failed_files:
                    f.write(f"{file}\n")
        
        return processed_files, failed_files

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
    
    # Initialize separator and process files
    separator = VocalSeparator(
        output_dir=args.output,
        model_name=args.model,
        segment=args.segment,
        bitrate=args.bitrate
    )
    
    processed_files, failed_files = separator.process_files(input_files, base_input_dir)
    
    logging.info(f"\nProcessing complete:")
    logging.info(f"Successfully processed: {len(processed_files)} files")
    if failed_files:
        logging.warning(f"Failed to process: {len(failed_files)} files")
        logging.info("Failed files have been saved to failed_files.txt")

if __name__ == "__main__":
    main() 