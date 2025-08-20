import os
import gc
import torch
import torchaudio
from pathlib import Path
from tqdm import tqdm
from demucs.pretrained import get_model
from demucs.apply import apply_model
from demucs.audio import convert_audio, save_audio  # available in demucs.audio
import logging
import time
import queue
import multiprocessing as mp
from itertools import islice
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor

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
DEFAULT_SEGMENT = 7.8  # Max practical segment for Demucs v4 (~7.8s)
NUM_GPUS = 8  # Number of GPUs to use
BATCH_SIZE = 32  # Number of files to process at once

class VocalSeparator:
    def __init__(self, output_dir="vocal_output", model_name="htdemucs", segment=DEFAULT_SEGMENT, bitrate=128, gpu_id=0, overlap=0.25, shifts=0, io_workers=4, save_workers=2):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.bitrate = bitrate
        self.gpu_id = gpu_id
        self.overlap = overlap
        self.shifts = shifts
        self.segment = segment
        self.io_workers = io_workers
        self.save_workers = save_workers
        
        logging.info(f"[GPU {gpu_id}] Loading model weights...")
        # Runtime knobs to squeeze perf on A100
        try:
            import torch.backends.cudnn as cudnn
            cudnn.benchmark = True
        except Exception:
            pass
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            if hasattr(torch, 'set_float32_matmul_precision'):
                torch.set_float32_matmul_precision('high')
        except Exception:
            pass
        # Load Demucs model on specified GPU
        self.model = get_model(model_name)
        self.model.to(f"cuda:{gpu_id}")
        self.model.eval()
        # Force model weight loading by doing a tiny separation
        dummy_audio = torch.zeros(1, getattr(self.model, 'audio_channels', 2), int(getattr(self.model, 'samplerate', 44100)))  # (B, C, T)
        with torch.no_grad():
            apply_model(
                self.model,
                dummy_audio,
                device=f"cuda:{gpu_id}",
                split=True,
                segment=self.segment,
                shifts=self.shifts,
                progress=False
            )
        logging.info(f"[GPU {gpu_id}] Model ready")
    
    def process_batch(self, batch_files, base_input_dir=None):
        """Process a batch of files together"""
        try:
            # Load all audio files in batch
            audios = []
            output_paths = []
            valid_files = []
            max_length = 0
            
            # First pass: load audio and find max length (in parallel)
            def _load_one(input_file):
                try:
                    input_path = Path(input_file)
                    if base_input_dir:
                        rel_path = input_path.relative_to(base_input_dir)
                        output_path = self.output_dir / rel_path.parent / f"{rel_path.stem}_vocals.mp3"
                    else:
                        output_path = self.output_dir / f"{input_path.stem}_vocals.mp3"

                    if output_path.exists():
                        return None

                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    audio, sr = torchaudio.load(str(input_file))
                    target_sr = getattr(self.model, 'samplerate', sr)
                    target_ch = getattr(self.model, 'audio_channels', audio.shape[0])
                    audio = convert_audio(audio, sr, target_sr, target_ch)
                    return (audio, output_path, input_file)
                except Exception as e:
                    logging.error(f"[GPU {self.gpu_id}] Error loading {input_file}: {str(e)}")
                    return None

            with ThreadPoolExecutor(max_workers=self.io_workers) as pool:
                for res in pool.map(_load_one, batch_files):
                    if res is None:
                        continue
                    audio, output_path, in_file = res
                    max_length = max(max_length, audio.shape[-1])
                    audios.append(audio)
                    output_paths.append(output_path)
                    valid_files.append(in_file)
            
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
            audio_batch = torch.stack(padded_audios)  # (B, C, T)
            # Speed up H2D copies
            audio_batch = audio_batch.pin_memory().to(f"cuda:{self.gpu_id}", non_blocking=True)
            with torch.inference_mode():
                estimates = apply_model(
                    self.model,
                    audio_batch,
                    device=f"cuda:{self.gpu_id}",
                    split=True,
                    segment=self.segment,
                    overlap=self.overlap,
                    shifts=self.shifts,
                    progress=False
                )
            
            # Process and save each result
            processed_files = []
            vocals_index = self.model.sources.index('vocals') if hasattr(self.model, 'sources') else 0
            # Move estimates to CPU before saving to free up GPU for next batch
            estimates_cpu = estimates.detach().cpu()
            def _save_one(i, output_path):
                try:
                    vocals = estimates_cpu[i, vocals_index]
                    if vocals.shape[0] > 1:
                        vocals = vocals.mean(dim=0, keepdim=True)
                    save_audio(
                        vocals,
                        str(output_path),
                        samplerate=getattr(self.model, 'samplerate', 44100),
                        bitrate=self.bitrate
                    )
                    return str(output_path)
                except Exception as e:
                    logging.error(f"[GPU {self.gpu_id}] Error saving {output_path}: {str(e)}")
                    return None
            with ThreadPoolExecutor(max_workers=self.save_workers) as pool:
                futures = [pool.submit(_save_one, i, output_path) for i, output_path in enumerate(output_paths)]
                for fut in futures:
                    res = fut.result()
                    if res:
                        processed_files.append(res)
            
            # Clear memory
            del audio_batch, estimates, padded_audios, audios, estimates_cpu
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

def gpu_worker(gpu_id, output_dir, model_name, segment, bitrate, overlap, shifts, batch_size, base_input_dir, task_queue, progress_queue, io_workers, save_workers):
    # Bind this process to a specific GPU
    try:
        torch.cuda.set_device(gpu_id)
    except Exception:
        pass
    separator = VocalSeparator(
        output_dir=output_dir,
        model_name=model_name,
        segment=segment,
        bitrate=bitrate,
        gpu_id=gpu_id,
        overlap=overlap,
        shifts=shifts,
        io_workers=io_workers,
        save_workers=save_workers,
    )
    processed_all, failed_all = [], []
    while True:
        batch_files = []
        for _ in range(batch_size):
            try:
                batch_files.append(task_queue.get_nowait())
            except Exception:
                break
        if not batch_files:
            break
        batch_processed, batch_attempted = separator.process_batch(batch_files, base_input_dir)
        processed_all.extend(batch_processed)
        failed_all.extend([f for f in batch_attempted if f not in batch_processed])
        try:
            # Count progress by files pulled from the queue to include skipped files
            progress_queue.put(len(batch_files))
        except Exception:
            pass
    if failed_all:
        failed_file_path = Path(output_dir) / f"failed_files_gpu{gpu_id}.txt"
        with open(failed_file_path, 'w') as f:
            for file in failed_all:
                f.write(f"{file}\n")

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
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size per GPU")
    parser.add_argument("--overlap", type=float, default=0.25, help="Chunk overlap ratio (lower is faster, may reduce quality)")
    parser.add_argument("--shifts", type=int, default=0, help="Test-time augmentation shifts (0 is fastest)")
    parser.add_argument("--num-gpus", type=int, default=None, help="Number of GPUs to use (default: all available)")
    parser.add_argument("--io-workers", type=int, default=4, help="Parallel workers for audio loading per GPU")
    parser.add_argument("--save-workers", type=int, default=2, help="Parallel workers for audio saving per GPU")
    
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
    
    # Dynamically balance work across GPUs using a shared queue
    manager = mp.Manager()
    task_queue = manager.Queue()
    progress_queue = manager.Queue()
    for f in input_files:
        task_queue.put(f)

    # Determine number of GPUs
    num_gpus = args.num_gpus if args.num_gpus is not None else torch.cuda.device_count() or NUM_GPUS

    processes = []
    for gpu_id in range(num_gpus):
        p = mp.Process(
            target=gpu_worker,
            args=(
                gpu_id,
                args.output,
                args.model,
                args.segment,
                args.bitrate,
                args.overlap,
                args.shifts,
                args.batch_size,
                base_input_dir,
                task_queue,
                progress_queue,
                args.io_workers,
                args.save_workers,
            ),
        )
        processes.append(p)
        p.start()

    # Global progress bar with ETA
    processed_count = 0
    total = len(input_files)
    with tqdm(total=total, desc="All GPUs", unit="file", dynamic_ncols=True) as pbar:
        while processed_count < total:
            try:
                inc = progress_queue.get(timeout=0.5)
                processed_count += inc
                pbar.update(inc)
            except queue.Empty:
                # If all workers are done and queue is empty, exit
                if not any(p.is_alive() for p in processes):
                    # Drain any remaining increments
                    while True:
                        try:
                            inc = progress_queue.get_nowait()
                            processed_count += inc
                            pbar.update(inc)
                        except Exception:
                            break
                    break

    for p in processes:
        p.join()
    
    logging.info("\nAll processes completed")
    
    # Combine failed files from all GPUs
    all_failed_files = []
    for gpu_id in range(num_gpus):
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