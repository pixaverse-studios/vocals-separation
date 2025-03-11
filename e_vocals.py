import os
import gc
import hashlib
import queue
import threading
import json
import librosa
import numpy as np
import soundfile as sf
import torch
import onnxruntime as ort
import warnings
from tqdm import tqdm
import argparse
import concurrent.futures
import glob
from pathlib import Path

warnings.filterwarnings("ignore")

BASE_DIR = "."
mdxnet_models_dir = os.path.join(BASE_DIR, "mdx_models")
output_dir = os.path.join(BASE_DIR, "vocal_output")

# Create directories if they don't exist
os.makedirs(mdxnet_models_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

MDX_DOWNLOAD_LINK = "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/"
UVR_MODELS = ["UVR-MDX-NET-Voc_FT.onnx"]  # Only need the vocal model

class MDXModel:
    def __init__(
        self,
        device,
        dim_f,
        dim_t,
        n_fft,
        hop=1024,
        stem_name=None,
        compensation=1.000,
    ):
        self.dim_f = dim_f
        self.dim_t = dim_t
        self.dim_c = 4
        self.n_fft = n_fft
        self.hop = hop
        self.stem_name = stem_name
        self.compensation = compensation
        self.n_bins = self.n_fft // 2 + 1
        self.chunk_size = hop * (self.dim_t - 1)
        self.window = torch.hann_window(window_length=self.n_fft, periodic=True).to(device)
        out_c = self.dim_c
        self.freq_pad = torch.zeros([1, out_c, self.n_bins - self.dim_f, self.dim_t]).to(device)

    def stft(self, x):
        x = x.reshape([-1, self.chunk_size])
        x = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop,
            window=self.window,
            center=True,
            return_complex=True,
        )
        x = torch.view_as_real(x)
        x = x.permute([0, 3, 1, 2])
        x = x.reshape([-1, 2, 2, self.n_bins, self.dim_t]).reshape(
            [-1, 4, self.n_bins, self.dim_t]
        )
        return x[:, :, : self.dim_f]

    def istft(self, x, freq_pad=None):
        freq_pad = (
            self.freq_pad.repeat([x.shape[0], 1, 1, 1])
            if freq_pad is None
            else freq_pad
        )
        x = torch.cat([x, freq_pad], -2)
        x = x.reshape([-1, 2, 2, self.n_bins, self.dim_t]).reshape(
            [-1, 2, self.n_bins, self.dim_t]
        )
        x = x.permute([0, 2, 3, 1])
        x = x.contiguous()
        x = torch.view_as_complex(x)
        x = torch.istft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop,
            window=self.window,
            center=True,
        )
        return x.reshape([-1, 2, self.chunk_size])

class MDX:
    DEFAULT_SR = 44100
    # Unit: seconds
    DEFAULT_CHUNK_SIZE = 0 * DEFAULT_SR
    DEFAULT_MARGIN_SIZE = 1 * DEFAULT_SR

    def __init__(self, model_path: str, params: MDXModel, processor=0):
        # Set the device and the provider (CPU or CUDA)
        self.device = (
            torch.device(f"cuda:{processor}")
            if processor >= 0
            else torch.device("cpu")
        )
        self.provider = (
            ["CUDAExecutionProvider"]
            if processor >= 0
            else ["CPUExecutionProvider"]
        )
        self.model = params
        # Load the ONNX model using ONNX Runtime
        self.ort = ort.InferenceSession(model_path, providers=self.provider)
        # Preload the model for faster performance
        self.ort.run(
            None,
            {"input": torch.rand(1, 4, params.dim_f, params.dim_t).numpy()},
        )
        self.process = lambda spec: self.ort.run(
            None, {"input": spec.cpu().numpy()}
        )[0]
        self.prog = None

    @staticmethod
    def get_hash(model_path):
        try:
            with open(model_path, "rb") as f:
                f.seek(-10000 * 1024, 2)
                model_hash = hashlib.md5(f.read()).hexdigest()
        except:
            # noqa
            model_hash = hashlib.md5(open(model_path, "rb").read()).hexdigest()
        return model_hash

    @staticmethod
    def segment(
        wave,
        combine=True,
        chunk_size=DEFAULT_CHUNK_SIZE,
        margin_size=DEFAULT_MARGIN_SIZE,
    ):
        """
        Segment or join segmented wave array
        Args:
            wave: (np.array) Wave array to be segmented or joined
            combine: (bool) If True, combines segmented wave array.
                    If False, segments wave array.
            chunk_size: (int) Size of each segment (in samples)
            margin_size: (int) Size of margin between segments (in samples)
        Returns:
            numpy array: Segmented or joined wave array
        """
        if combine:
            # Initializing as None instead of [] for later numpy array concatenation
            processed_wave = None
            for segment_count, segment in enumerate(wave):
                start = 0 if segment_count == 0 else margin_size
                end = None if segment_count == len(wave) - 1 else -margin_size
                if margin_size == 0:
                    end = None
                if processed_wave is None:
                    # Create array for first segment
                    processed_wave = segment[:, start:end]
                else:
                    # Concatenate to existing array for subsequent segments
                    processed_wave = np.concatenate(
                        (processed_wave, segment[:, start:end]), axis=-1
                    )
        else:
            processed_wave = []
            sample_count = wave.shape[-1]
            if chunk_size <= 0 or chunk_size > sample_count:
                chunk_size = sample_count
            if margin_size > chunk_size:
                margin_size = chunk_size
            for segment_count, skip in enumerate(
                range(0, sample_count, chunk_size)
            ):
                margin = 0 if segment_count == 0 else margin_size
                end = min(skip + chunk_size + margin_size, sample_count)
                start = skip - margin
                cut = wave[:, start:end].copy()
                processed_wave.append(cut)
                if end == sample_count:
                    break
        return processed_wave

    def pad_wave(self, wave):
        """
        Pad the wave array to match the required chunk size
        Args:
            wave: (np.array) Wave array to be padded
        Returns:
            tuple: (padded_wave, pad, trim)
                - padded_wave: Padded wave array
                - pad: Number of samples that were padded
                - trim: Number of samples that were trimmed
        """
        n_sample = wave.shape[1]
        trim = self.model.n_fft // 2
        gen_size = self.model.chunk_size - 2 * trim
        pad = gen_size - n_sample % gen_size
        # Padded wave
        wave_p = np.concatenate(
            (
                np.zeros((2, trim)),
                wave,
                np.zeros((2, pad)),
                np.zeros((2, trim)),
            ),
            1,
        )
        mix_waves = []
        for i in range(0, n_sample + pad, gen_size):
            waves = np.array(wave_p[:, i:i + self.model.chunk_size])
            mix_waves.append(waves)
        mix_waves = torch.tensor(mix_waves, dtype=torch.float32).to(
            self.device
        )
        return mix_waves, pad, trim

    def _process_wave(self, mix_waves, trim, pad, q: queue.Queue, _id: int):
        """
        Process each wave segment in a multi-threaded environment
        Args:
            mix_waves: (torch.Tensor) Wave segments to be processed
            trim: (int) Number of samples trimmed during padding
            pad: (int) Number of samples padded during padding
            q: (queue.Queue) Queue to hold the processed wave segments
            _id: (int) Identifier of the processed wave segment
        Returns:
            numpy array: Processed wave segment
        """
        mix_waves = mix_waves.split(1)
        with torch.no_grad():
            pw = []
            for mix_wave in mix_waves:
                self.prog.update()
                spec = self.model.stft(mix_wave)
                processed_spec = torch.tensor(self.process(spec))
                processed_wav = self.model.istft(
                    processed_spec.to(self.device)
                )
                processed_wav = (
                    processed_wav[:, :, trim:-trim]
                    .transpose(0, 1)
                    .reshape(2, -1)
                    .cpu()
                    .numpy()
                )
                pw.append(processed_wav)
            processed_signal = np.concatenate(pw, axis=-1)[:, :-pad]
            q.put({_id: processed_signal})
        return processed_signal

    def process_wave(self, wave: np.array, mt_threads=1):
        """
        Process the wave array in a multi-threaded environment
        Args:
            wave: (np.array) Wave array to be processed
            mt_threads: (int) Number of threads to be used for processing
        Returns:
            numpy array: Processed wave array
        """
        self.prog = tqdm(total=0)
        chunk = wave.shape[-1] // mt_threads
        waves = self.segment(wave, False, chunk)
        # Create a queue to hold the processed wave segments
        q = queue.Queue()
        threads = []
        for c, batch in enumerate(waves):
            mix_waves, pad, trim = self.pad_wave(batch)
            self.prog.total = len(mix_waves) * mt_threads
            thread = threading.Thread(
                target=self._process_wave, args=(mix_waves, trim, pad, q, c)
            )
            thread.start()
            threads.append(thread)
        for thread in threads:
            thread.join()
        self.prog.close()
        processed_batches = []
        while not q.empty():
            processed_batches.append(q.get())
        processed_batches = [
            list(wave.values())[0]
            for wave in sorted(
                processed_batches, key=lambda d: list(d.keys())[0]
            )
        ]
        assert len(processed_batches) == len(
            waves
        ), "Incomplete processed batches, please reduce batch size!"
        return self.segment(processed_batches, True, chunk)

def download_model(url, output_dir):
    """Download model if it doesn't exist"""
    import requests
    from pathlib import Path
    
    os.makedirs(output_dir, exist_ok=True)
    model_name = url.split("/")[-1]
    model_path = os.path.join(output_dir, model_name)
    
    if not os.path.exists(model_path):
        print(f"Downloading {model_name}...")
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        # Download with progress bar
        with open(model_path, 'wb') as f, tqdm(
            desc=model_name,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(chunk_size=1024):
                size = f.write(data)
                bar.update(size)
    else:
        print(f"Model {model_name} already exists")
    
    return model_path

def load_as_stereo(audio_path, sr=44100):
    """Load audio directly as stereo without creating interim files"""
    try:
        wave, sr = librosa.load(audio_path, mono=False, sr=sr)
        
        # If mono, convert to stereo
        if len(wave.shape) == 1 or wave.shape[0] == 1:
            wave = np.stack([wave, wave]) if len(wave.shape) == 1 else np.stack([wave[0], wave[0]])
            
        return wave, sr
    except Exception as e:
        print(f"Error loading audio: {str(e)}")
        
        # Try using ffmpeg to load into memory directly
        import subprocess
        import tempfile
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=True) as temp_file:
            temp_path = temp_file.name
            cmd = f'ffmpeg -y -loglevel error -i "{audio_path}" -ac 2 -f wav "{temp_path}"'
            subprocess.run(cmd, shell=True, check=True)
            
            # Load the converted file
            wave, sr = librosa.load(temp_path, mono=False, sr=sr)
            
        return wave, sr

def download_data_json():
    """Download the data.json file if it doesn't exist"""
    import requests
    
    data_path = os.path.join(mdxnet_models_dir, "data.json")
    
    if not os.path.exists(data_path):
        print("Downloading data.json...")
        url = "https://raw.githubusercontent.com/TRvlvr/model_repo/main/mdx_models_data/data.json"
        response = requests.get(url)
        
        with open(data_path, 'w') as f:
            f.write(response.text)
        print("data.json downloaded")
    else:
        print("data.json already exists")
    
    return data_path

def extract_vocals(audio_file_path, output_path=None, model_path=None, mdx_model_params=None, gpu_id=0):
    """Extract vocals from the input audio file"""
    try:
        # Setup output path
        if output_path is None:
            filename = os.path.basename(audio_file_path)
            output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_vocals.wav")
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Load audio directly as stereo
        print(f"Loading audio: {audio_file_path}")
        wave, sr = load_as_stereo(audio_file_path)
        
        # Setup model
        model_hash = MDX.get_hash(model_path)
        mp = mdx_model_params.get(model_hash)
        
        # Use GPU if available, with specified GPU ID
        device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
        processor_num = gpu_id if torch.cuda.is_available() else -1
        
        print(f"Processing {os.path.basename(audio_file_path)} on {device}...")
        
        # Create model
        model = MDXModel(
            device,
            dim_f=mp["mdx_dim_f_set"],
            dim_t=2 ** mp["mdx_dim_t_set"],
            n_fft=mp["mdx_n_fft_scale_set"],
            stem_name=mp["primary_stem"],
            compensation=mp["compensate"],
        )
        
        # Create MDX processor
        mdx_sess = MDX(model_path, model, processor=processor_num)
        
        # Normalize
        peak = max(np.max(wave), abs(np.min(wave)))
        wave /= peak
        
        # Set threads based on GPU VRAM
        m_threads = 1
        if torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(device).total_memory / 1024**3
            m_threads = 1 if vram_gb < 8 else (4 if vram_gb > 32 else 2)
        
        # Process wave
        wave_processed = mdx_sess.process_wave(wave, m_threads)
        
        # Return to previous peak
        wave_processed *= peak
        
        # Save output
        print(f"Saving vocals to {output_path}")
        sf.write(output_path, wave_processed.T, sr)
        
        # Clean up
        del mdx_sess, wave_processed, wave
        gc.collect()
        torch.cuda.empty_cache()
        
        return output_path
    
    except Exception as e:
        print(f"Error processing {audio_file_path}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def process_file(args):
    """Process a single file - used for parallel processing"""
    audio_file, output_base_dir, input_base_dir, model_path, mdx_model_params, gpu_id = args
    
    try:
        # Calculate relative path to maintain folder structure
        rel_path = os.path.relpath(audio_file, input_base_dir)
        
        # Create output path with same folder structure
        output_path = os.path.join(output_base_dir, os.path.dirname(rel_path), 
                                f"{os.path.splitext(os.path.basename(audio_file))[0]}_vocals.wav")
        
        return extract_vocals(audio_file, output_path, model_path, mdx_model_params, gpu_id)
    except Exception as e:
        print(f"Error in process_file for {audio_file}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def process_batch(file_list, input_base_dir, output_base_dir, num_gpus=1, max_workers=4):
    """Process a batch of files in parallel"""
    # Download model and data.json
    model_path = download_model(MDX_DOWNLOAD_LINK + UVR_MODELS[0], mdxnet_models_dir)
    data_json_path = download_data_json()
    
    # Load model parameters
    with open(data_json_path) as f:
        mdx_model_params = json.load(f)
    
    # Prepare arguments for each file processing task
    tasks = []
    for i, file_path in enumerate(file_list):
        gpu_id = i % num_gpus  # Distribute tasks across available GPUs
        tasks.append((file_path, output_base_dir, input_base_dir, model_path, mdx_model_params, gpu_id))
    
    # Process files in parallel
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_file, task) for task in tasks]
        
        # Show progress bar
        for i, future in enumerate(tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing files")):
            try:
                result = future.result()
                if result:
                    results.append(result)
            except Exception as e:
                print(f"Error processing task: {str(e)}")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract vocals from multiple audio files using MDX-Net")
    parser.add_argument("inputs", nargs='+', help="Input audio file paths or directory (use '*' for wildcards)")
    parser.add_argument("--output", help="Output directory for vocals")
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs to use (default: 1)")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers (default: 4)")
    
    args = parser.parse_args()
    
    # Set output directory
    output_base_dir = args.output if args.output else output_dir
    
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
