"""Core Fugatto model and processing classes."""

from typing import Dict, Any, Optional, Union, List
# Conditional imports for testing
try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False
    
    # Mock torch for testing
    class MockTorch:
        @staticmethod
        def cuda():
            class MockCuda:
                @staticmethod
                def is_available():
                    return False
            return MockCuda()
        
        @staticmethod
        def tensor(data):
            class MockTensor:
                def __init__(self, data):
                    self.data = data
                def to(self, device):
                    return self
                def __getitem__(self, item):
                    return self.data[item] if hasattr(self.data, '__getitem__') else self.data
                def __len__(self):
                    return len(self.data) if hasattr(self.data, '__len__') else 1
            return MockTensor(data)
        
        @staticmethod
        def no_grad():
            class MockNoGrad:
                def __enter__(self):
                    return self
                def __exit__(self, exc_type, exc_val, exc_tb):
                    pass
            return MockNoGrad()
            
        class nn:
            class Module:
                def __init__(self):
                    pass
                def to(self, device):
                    return self
                def eval(self):
                    return self
            
            class Linear:
                def __init__(self, in_features, out_features):
                    self.in_features = in_features
                    self.out_features = out_features
                    
            class Embedding:
                def __init__(self, num_embeddings, embedding_dim):
                    pass
                    
            class Sequential:
                def __init__(self, *layers):
                    pass
                    
            class ReLU:
                def __init__(self):
                    pass
    
    if not HAS_TORCH:
        torch = MockTorch()
        torch.nn = MockTorch.nn

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    # Mock numpy for testing
    class MockNumpy:
        @staticmethod
        def zeros(shape, dtype=None):
            if isinstance(shape, int):
                return [0.0] * shape
            return [[0.0 for _ in range(shape[1])] for _ in range(shape[0])]
        
        @staticmethod
        def ones(shape, dtype=None):
            if isinstance(shape, int):
                return [1.0] * shape
            return [[1.0 for _ in range(shape[1])] for _ in range(shape[0])]
        
        @staticmethod
        def array(data, dtype=None):
            return list(data) if hasattr(data, '__iter__') else [data]
        
        @staticmethod
        def mean(data):
            return sum(data) / len(data) if data else 0
        
        @staticmethod
        def std(data):
            if not data:
                return 0
            mean_val = sum(data) / len(data)
            variance = sum((x - mean_val) ** 2 for x in data) / len(data)
            return variance ** 0.5
        
        @staticmethod
        def max(data):
            return max(data) if data else 0
        
        @staticmethod
        def min(data):
            return min(data) if data else 0
        
        @staticmethod
        def abs(data):
            if hasattr(data, '__iter__'):
                return [abs(x) for x in data]
            return abs(data)
        
        @staticmethod
        def sqrt(data):
            if hasattr(data, '__iter__'):
                return [x**0.5 for x in data]
            return data**0.5
        
        @staticmethod
        def sum(data):
            return sum(data) if hasattr(data, '__iter__') else data
        
        @staticmethod
        def clip(data, min_val, max_val):
            if hasattr(data, '__iter__'):
                return [max(min_val, min(x, max_val)) for x in data]
            return max(min_val, min(data, max_val))
        
        @staticmethod
        def linspace(start, stop, num):
            if num <= 1:
                return [stop]
            step = (stop - start) / (num - 1)
            return [start + i * step for i in range(num)]
        
        @staticmethod
        def concatenate(arrays):
            result = []
            for arr in arrays:
                if hasattr(arr, '__iter__'):
                    result.extend(arr)
                else:
                    result.append(arr)
            return result
        
        @staticmethod
        def interp(x, xp, fp):
            # Simple linear interpolation
            result = []
            for xi in x:
                for i in range(len(xp) - 1):
                    if xp[i] <= xi <= xp[i + 1]:
                        # Linear interpolation
                        t = (xi - xp[i]) / (xp[i + 1] - xp[i]) if xp[i + 1] != xp[i] else 0
                        yi = fp[i] + t * (fp[i + 1] - fp[i])
                        result.append(yi)
                        break
                else:
                    # Out of bounds, use nearest
                    if xi <= xp[0]:
                        result.append(fp[0])
                    else:
                        result.append(fp[-1])
            return result
        
        @staticmethod
        def full(shape, fill_value, dtype=None):
            if isinstance(shape, int):
                return [fill_value] * shape
            return [[fill_value for _ in range(shape[1])] for _ in range(shape[0])]
        
        @staticmethod
        def sin(data):
            import math
            if hasattr(data, '__iter__'):
                return [math.sin(x) for x in data]
            return math.sin(data)
        
        @staticmethod
        def random():
            import random
            class MockRandom:
                @staticmethod
                def uniform(low, high):
                    return random.uniform(low, high)
                @staticmethod
                def normal(mean, std):
                    return random.gauss(mean, std) 
            return MockRandom()
        
        # Mock numpy types
        ndarray = list
        float32 = float
        pi = 3.141592653589793
    
    if not HAS_NUMPY:
        np = MockNumpy()
import warnings
import logging
from pathlib import Path
from .optimization import get_audio_cache, performance_optimized

try:
    import librosa
    import soundfile as sf
except ImportError:
    librosa = None
    sf = None
    warnings.warn("librosa and soundfile not available. Audio I/O will use fallback methods.")

try:
    from transformers import AutoTokenizer, AutoModel
except ImportError:
    AutoTokenizer = None
    AutoModel = None
    warnings.warn("transformers not available. Model loading will use fallback methods.")

logger = logging.getLogger(__name__)


class FugattoModel:
    """Main Fugatto model interface for audio generation and transformation."""
    
    def __init__(self, model_name: str = "nvidia/fugatto-base", device: Optional[str] = None):
        """Initialize Fugatto model.
        
        Args:
            model_name: HuggingFace model identifier or local path
            device: Device to run model on ('cuda', 'cpu', or None for auto)
        """
        self.model_name = model_name
        self.device = device or ('cuda' if (HAS_TORCH and torch.cuda.is_available()) else 'cpu')
        self.sample_rate = 48000
        self.max_duration = 30.0  # Maximum generation duration in seconds
        
        # Model components (initialized lazily)
        self._tokenizer = None
        self._model = None
        self._loaded = False
        
        # Performance optimization
        self._audio_cache = get_audio_cache()
        
        logger.info(f"Initialized FugattoModel with device: {self.device}")
    
    @classmethod
    def from_pretrained(cls, model_name: str, device: Optional[str] = None) -> "FugattoModel":
        """Load pretrained Fugatto model.
        
        Args:
            model_name: HuggingFace model identifier or local path
            device: Device to run model on
            
        Returns:
            Initialized FugattoModel instance
        """
        instance = cls(model_name, device)
        instance._load_model()
        return instance
    
    def _load_model(self) -> None:
        """Load the model and tokenizer."""
        if self._loaded:
            return
            
        logger.info(f"Loading Fugatto model: {self.model_name}")
        
        try:
            if HAS_TORCH and AutoTokenizer is not None and AutoModel is not None:
                # Try to load from HuggingFace
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self._model = AutoModel.from_pretrained(self.model_name)
                self._model.to(self.device)
                self._model.eval()
                logger.info("Model loaded successfully from HuggingFace")
            else:
                logger.warning("Torch or transformers not available, using mock model")
                self._create_mock_model()
        except Exception as e:
            logger.warning(f"Failed to load model from HuggingFace: {e}. Using mock model.")
            self._create_mock_model()
            
        self._loaded = True
    
    def _create_mock_model(self) -> None:
        """Create a mock model for testing/development."""
        logger.info("Creating mock Fugatto model for development")
        
        class MockTokenizer:
            def encode(self, text: str, return_tensors: str = None):
                # Simple character-based tokenization
                tokens = [ord(c) % 1000 for c in text[:50]]  # Limit to 50 chars
                if return_tensors == "pt":
                    return torch.tensor([tokens])
                return tokens
        
        class MockFugattoModel:
            def __init__(self, sample_rate: int = 48000):
                self.sample_rate = sample_rate
                self.embedding_dim = 512
                # Simple mock layers without actual neural network functionality
                self.audio_embedding = "mock_audio_embedding"
                self.text_embedding = "mock_text_embedding"
                self.generator = "mock_generator"
                
            def forward(self, input_ids, audio_input=None):
                # Simple mock forward pass
                import random
                # Generate a simple output based on input characteristics
                if hasattr(input_ids, '__len__'):
                    input_length = len(input_ids[0]) if isinstance(input_ids[0], list) else 1
                else:
                    input_length = 1
                
                # Simple mock output - return a mock tensor-like object
                output_value = 0.5 + 0.3 * (input_length % 3) / 3.0 + 0.2 * random.random()
                
                class MockOutput:
                    def __init__(self, value):
                        self.value = value
                    def cpu(self):
                        return self
                    def numpy(self):
                        return MockNumpy()
                    def flatten(self):
                        return [self.value]
                
                class MockNumpy:
                    def flatten(self):
                        return [output_value]
                
                return MockOutput(output_value)
            
            def to(self, device):
                return self
            
            def eval(self):
                return self
            
            def __call__(self, *args, **kwargs):
                return self.forward(*args, **kwargs)
        
        self._tokenizer = MockTokenizer()
        self._model = MockFugattoModel(self.sample_rate).to(self.device)
    
    @performance_optimized(cache_enabled=True)
    def generate(self, prompt: str, duration_seconds: float = 10.0, 
                temperature: float = 0.8, top_p: float = 0.95, 
                guidance_scale: float = 3.0) -> np.ndarray:
        """Generate audio from text prompt.
        
        Args:
            prompt: Text description of desired audio
            duration_seconds: Length of audio to generate (max 30s)
            temperature: Sampling temperature (0.1-1.5)
            top_p: Nucleus sampling parameter
            guidance_scale: Classifier-free guidance scale
            
        Returns:
            Generated audio as numpy array (mono, 48kHz)
        """
        if not self._loaded:
            self._load_model()
            
        # Validate inputs
        duration_seconds = min(duration_seconds, self.max_duration)
        temperature = max(0.1, min(temperature, 1.5))
        
        logger.info(f"Generating audio: '{prompt}' ({duration_seconds}s, T={temperature})")
        
        with torch.no_grad():
            # Tokenize prompt
            input_ids = self._tokenizer.encode(prompt, return_tensors="pt")
            if isinstance(input_ids, list):
                input_ids = torch.tensor([input_ids])
            input_ids = input_ids.to(self.device)
            
            # Calculate number of samples
            num_samples = int(duration_seconds * self.sample_rate)
            
            # Generate audio in chunks for memory efficiency
            chunk_size = self.sample_rate  # 1 second chunks
            audio_chunks = []
            
            for chunk_start in range(0, num_samples, chunk_size):
                chunk_samples = min(chunk_size, num_samples - chunk_start)
                
                # Add noise for temperature control
                noise_scale = temperature * 0.1
                
                # Generate chunk
                if hasattr(self._model, 'forward'):
                    # Mock model path
                    output = self._model(input_ids)
                    
                    # Expand to chunk size with controlled randomness
                    if hasattr(output, 'cpu'):
                        base_pattern = output.cpu().numpy().flatten()[0]
                    elif hasattr(output, 'flatten'):
                        flat_output = output.flatten()
                        base_pattern = flat_output[0] if len(flat_output) > 0 else 0.5
                    else:
                        base_pattern = 0.5
                    chunk = np.full(chunk_samples, base_pattern)
                    
                    # Add temporal variation
                    t = np.linspace(0, 1, chunk_samples)
                    import random
                    variation = np.sin(2 * np.pi * t * random.uniform(0.5, 2.0)) * noise_scale
                    chunk += variation
                    
                    # Add some harmonics for more natural sound
                    for harmonic in [2, 3, 4]:
                        chunk += 0.3 * np.sin(2 * np.pi * t * harmonic * random.uniform(0.5, 2.0)) * noise_scale
                else:
                    # Fallback: structured noise based on prompt
                    chunk = self._generate_structured_audio(prompt, chunk_samples, temperature)
                
                audio_chunks.append(chunk)
            
            # Combine chunks
            audio = np.concatenate(audio_chunks)
            
            # Normalize and apply final processing
            audio = self._post_process_audio(audio, temperature)
            
        logger.info(f"Generated {len(audio)/self.sample_rate:.2f}s of audio")
        return audio.astype(np.float32)
    
    def transform(self, audio: np.ndarray, prompt: str, 
                 strength: float = 0.7, preserve_length: bool = True) -> np.ndarray:
        """Transform audio with text conditioning.
        
        Args:
            audio: Input audio array
            prompt: Text description of desired transformation
            strength: Transformation strength (0.0-1.0)
            preserve_length: Whether to maintain original audio length
            
        Returns:
            Transformed audio array
        """
        if not self._loaded:
            self._load_model()
            
        strength = max(0.0, min(strength, 1.0))
        
        logger.info(f"Transforming audio with prompt: '{prompt}' (strength={strength})")
        
        with torch.no_grad():
            # Tokenize prompt
            input_ids = self._tokenizer.encode(prompt, return_tensors="pt")
            if isinstance(input_ids, list):
                input_ids = torch.tensor([input_ids])
            input_ids = input_ids.to(self.device)
            
            # Convert audio to tensor
            audio_tensor = torch.from_numpy(audio).float().to(self.device)
            
            if hasattr(self._model, 'forward'):
                # Generate transformation
                output = self._model(input_ids, audio_tensor)
                
                # Apply transformation with strength control
                if output.numel() == 1:
                    # Expand single output to match audio length
                    transformation = output.item() * np.ones_like(audio)
                else:
                    transformation = output.cpu().numpy().flatten()
                    # Resize to match input if needed
                    if len(transformation) != len(audio):
                        transformation = np.resize(transformation, len(audio))
            else:
                # Fallback transformation
                transformation = self._generate_structured_audio(prompt, len(audio), 0.5)
            
            # Blend original and transformed audio
            transformed = audio * (1 - strength) + transformation * strength
            
            # Apply smoothing to avoid artifacts
            transformed = self._apply_smoothing(transformed)
            
        logger.info(f"Transformed {len(audio)/self.sample_rate:.2f}s of audio")
        return transformed.astype(np.float32)
    
    def generate_multi(self, text_prompt: str, style_audio: Optional[np.ndarray] = None,
                      attributes: Optional[Dict[str, Any]] = None, 
                      duration_seconds: float = 10.0) -> np.ndarray:
        """Generate audio with multiple conditioning inputs.
        
        Args:
            text_prompt: Text description
            style_audio: Reference audio for style transfer
            attributes: Additional control attributes (tempo, key, etc.)
            duration_seconds: Output duration
            
        Returns:
            Generated audio array
        """
        logger.info(f"Multi-modal generation: '{text_prompt}'")
        
        # Base generation
        audio = self.generate(text_prompt, duration_seconds)
        
        # Apply style transfer if reference audio provided
        if style_audio is not None:
            style_prompt = "Transfer style from reference audio"
            audio = self.transform(audio, style_prompt, strength=0.4)
        
        # Apply attribute modifications
        if attributes:
            audio = self._apply_attributes(audio, attributes)
        
        return audio
    
    def _generate_structured_audio(self, prompt: str, num_samples: int, temperature: float) -> np.ndarray:
        """Generate structured audio based on prompt content."""
        # Create different patterns based on prompt keywords
        prompt_lower = prompt.lower()
        
        t = np.linspace(0, num_samples / self.sample_rate, num_samples)
        audio = np.zeros(num_samples)
        
        # Base frequency based on prompt
        base_freq = 440.0  # A4
        
        if any(word in prompt_lower for word in ['cat', 'meow']):
            # High-pitched, varying frequency for cat sounds
            freq_variation = 200 * np.sin(2 * np.pi * t * 3)
            audio = 0.3 * np.sin(2 * np.pi * (800 + freq_variation) * t)
            # Add some noise for realism
            audio += 0.1 * np.random.normal(0, 1, num_samples)
            
        elif any(word in prompt_lower for word in ['dog', 'bark']):
            # Lower-pitched, more aggressive patterns
            envelope = np.exp(-t * 2) * (t < 0.5)
            audio = 0.5 * np.sin(2 * np.pi * 300 * t) * envelope
            audio += 0.3 * np.sin(2 * np.pi * 600 * t) * envelope
            
        elif any(word in prompt_lower for word in ['ocean', 'wave', 'water']):
            # Noise-based for water sounds
            audio = 0.2 * np.random.normal(0, 1, num_samples)
            # Apply low-pass filtering effect
            for i in range(1, len(audio)):
                audio[i] = 0.7 * audio[i] + 0.3 * audio[i-1]
                
        elif any(word in prompt_lower for word in ['music', 'piano', 'guitar']):
            # Musical tones
            for harmonic in [1, 2, 3, 4]:
                audio += (0.5 / harmonic) * np.sin(2 * np.pi * base_freq * harmonic * t)
                
        else:
            # Default: structured noise with some tonal elements
            audio = 0.1 * np.random.normal(0, 1, num_samples)
            audio += 0.2 * np.sin(2 * np.pi * base_freq * t)
        
        # Apply temperature-based variation
        audio *= (1 + temperature * 0.5 * np.random.normal(0, 1, num_samples))
        
        return audio
    
    def _post_process_audio(self, audio: np.ndarray, temperature: float) -> np.ndarray:
        """Apply post-processing to generated audio."""
        # Normalize to prevent clipping
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val * 0.8  # Leave some headroom
        
        # Apply gentle fade in/out
        fade_samples = int(0.1 * self.sample_rate)  # 100ms fade
        if len(audio) > 2 * fade_samples:
            fade_in = np.linspace(0, 1, fade_samples)
            fade_out = np.linspace(1, 0, fade_samples)
            audio[:fade_samples] *= fade_in
            audio[-fade_samples:] *= fade_out
        
        return audio
    
    def _apply_smoothing(self, audio: np.ndarray, window_size: int = 5) -> np.ndarray:
        """Apply smoothing to reduce artifacts."""
        if len(audio) < window_size:
            return audio
            
        # Simple moving average
        kernel = np.ones(window_size) / window_size
        smoothed = np.convolve(audio, kernel, mode='same')
        return smoothed
    
    def _apply_attributes(self, audio: np.ndarray, attributes: Dict[str, Any]) -> np.ndarray:
        """Apply attribute-based modifications to audio."""
        modified = audio.copy()
        
        # Tempo adjustment (simple time-stretching simulation)
        if 'tempo' in attributes:
            tempo_factor = attributes['tempo'] / 120.0  # Assume 120 BPM baseline
            if tempo_factor != 1.0:
                # Simple resampling for tempo change
                new_length = int(len(audio) / tempo_factor)
                indices = np.linspace(0, len(audio) - 1, new_length)
                modified = np.interp(range(len(audio)), 
                                   np.linspace(0, len(audio) - 1, new_length), 
                                   modified[indices.astype(int)])
        
        # Reverb simulation
        if 'reverb' in attributes:
            reverb_amount = attributes['reverb']
            if reverb_amount > 0:
                delay_samples = int(0.05 * self.sample_rate)  # 50ms delay
                if len(modified) > delay_samples:
                    reverb = np.zeros_like(modified)
                    reverb[delay_samples:] = modified[:-delay_samples] * reverb_amount * 0.3
                    modified = modified + reverb
        
        return modified
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            'model_name': self.model_name,
            'device': self.device,
            'sample_rate': self.sample_rate,
            'max_duration': self.max_duration,
            'loaded': self._loaded,
            'cuda_available': torch.cuda.is_available(),
            'memory_usage': torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        }


class AudioProcessor:
    """Audio processing utilities for loading, saving and visualization."""
    
    def __init__(self, sample_rate: int = 48000, target_loudness: float = -14.0):
        """Initialize audio processor.
        
        Args:
            sample_rate: Target sample rate
            target_loudness: Target loudness in LUFS for normalization
        """
        self.sample_rate = sample_rate
        self.target_loudness = target_loudness
        self.supported_formats = ['.wav', '.mp3', '.flac', '.ogg', '.m4a']
        
        # Advanced processing parameters
        self.hop_length = 512
        self.n_fft = 2048
        self.n_mels = 128
        
        # Audio enhancement settings
        self.noise_gate_threshold = -60.0  # dB
        self.compressor_ratio = 4.0
        self.eq_bands = {
            'low': {'freq': 80, 'gain': 0, 'q': 0.7},
            'mid': {'freq': 1000, 'gain': 0, 'q': 0.7},
            'high': {'freq': 8000, 'gain': 0, 'q': 0.7}
        }
        
        logger.info(f"AudioProcessor initialized: {sample_rate}Hz, {target_loudness} LUFS")
    
    def load_audio(self, filepath: Union[str, Path], 
                   target_sample_rate: Optional[int] = None) -> np.ndarray:
        """Load audio file with automatic resampling.
        
        Args:
            filepath: Path to audio file
            target_sample_rate: Resample to this rate (uses self.sample_rate if None)
            
        Returns:
            Audio data as numpy array (mono)
        """
        filepath = Path(filepath)
        target_sr = target_sample_rate or self.sample_rate
        
        if not filepath.exists():
            raise FileNotFoundError(f"Audio file not found: {filepath}")
        
        if filepath.suffix.lower() not in self.supported_formats:
            logger.warning(f"Unsupported format {filepath.suffix}, attempting to load anyway")
        
        try:
            if sf is not None and librosa is not None:
                # Load with librosa for better format support
                audio, sr = librosa.load(str(filepath), sr=target_sr, mono=True)
                logger.info(f"Loaded {filepath.name}: {len(audio)/sr:.2f}s at {sr}Hz")
                return audio.astype(np.float32)
            else:
                # Fallback: try soundfile directly
                if sf is not None:
                    audio, sr = sf.read(str(filepath))
                    if audio.ndim > 1:
                        audio = np.mean(audio, axis=1)  # Convert to mono
                    
                    # Manual resampling if needed
                    if sr != target_sr:
                        audio = self._resample_audio(audio, sr, target_sr)
                        sr = target_sr
                    
                    logger.info(f"Loaded {filepath.name}: {len(audio)/sr:.2f}s at {sr}Hz")
                    return audio.astype(np.float32)
                else:
                    raise ImportError("No audio loading library available")
                    
        except Exception as e:
            logger.error(f"Failed to load audio {filepath}: {e}")
            # Return silence as fallback
            logger.warning("Returning 1 second of silence as fallback")
            return np.zeros(target_sr, dtype=np.float32)
    
    def save_audio(self, audio: np.ndarray, filepath: Union[str, Path], 
                   sample_rate: Optional[int] = None, normalize: bool = True,
                   format: Optional[str] = None) -> None:
        """Save audio to file with optional normalization.
        
        Args:
            audio: Audio data to save
            filepath: Output file path
            sample_rate: Sample rate (uses self.sample_rate if None)
            normalize: Whether to normalize loudness
            format: Audio format override
        """
        filepath = Path(filepath)
        sr = sample_rate or self.sample_rate
        
        # Create output directory if needed
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Normalize audio if requested
        if normalize:
            audio = self.normalize_loudness(audio)
        
        # Ensure audio is in valid range
        audio = np.clip(audio, -1.0, 1.0)
        
        try:
            if sf is not None:
                # Determine format from extension or parameter
                if format is None:
                    format = filepath.suffix.lower().lstrip('.')
                
                # Save with soundfile
                sf.write(str(filepath), audio, sr, format=format)
                logger.info(f"Saved audio: {filepath} ({len(audio)/sr:.2f}s, {sr}Hz)")
            else:
                # Fallback: save as numpy array
                np.save(filepath.with_suffix('.npy'), audio)
                logger.warning(f"Saved as numpy array: {filepath.with_suffix('.npy')}")
                
        except Exception as e:
            logger.error(f"Failed to save audio {filepath}: {e}")
            # Fallback: save as numpy array
            np.save(filepath.with_suffix('.npy'), audio)
            logger.info(f"Saved as numpy fallback: {filepath.with_suffix('.npy')}")
    
    def normalize_loudness(self, audio: np.ndarray, target_lufs: Optional[float] = None) -> np.ndarray:
        """Normalize audio to target loudness.
        
        Args:
            audio: Input audio
            target_lufs: Target loudness (uses self.target_loudness if None)
            
        Returns:
            Normalized audio
        """
        target = target_lufs or self.target_loudness
        
        # Simple RMS-based normalization (approximates loudness)
        rms = np.sqrt(np.mean(audio ** 2))
        if rms > 0:
            # Convert target LUFS to linear scale (rough approximation)
            target_rms = 10 ** (target / 20) * 0.1
            scale_factor = target_rms / rms
            normalized = audio * scale_factor
            
            # Prevent clipping
            max_val = np.max(np.abs(normalized))
            if max_val > 0.95:
                normalized = normalized * (0.95 / max_val)
                
            logger.debug(f"Normalized audio: RMS {rms:.4f} -> {np.sqrt(np.mean(normalized**2)):.4f}")
            return normalized
        else:
            return audio
    
    def preprocess(self, audio: np.ndarray, normalize: bool = True, 
                   trim_silence: bool = True, apply_filter: bool = False) -> np.ndarray:
        """Apply comprehensive preprocessing to audio.
        
        Args:
            audio: Input audio
            normalize: Apply loudness normalization
            trim_silence: Remove leading/trailing silence
            apply_filter: Apply basic filtering
            
        Returns:
            Preprocessed audio
        """
        processed = audio.copy()
        
        # Trim silence
        if trim_silence:
            processed = self._trim_silence(processed)
        
        # Apply basic filtering
        if apply_filter:
            processed = self._apply_basic_filter(processed)
        
        # Normalize loudness
        if normalize:
            processed = self.normalize_loudness(processed)
        
        logger.debug(f"Preprocessed audio: {len(audio)} -> {len(processed)} samples")
        return processed
    
    def _trim_silence(self, audio: np.ndarray, threshold: float = 0.01) -> np.ndarray:
        """Remove leading and trailing silence."""
        # Find non-silent regions
        non_silent = np.abs(audio) > threshold
        
        if not np.any(non_silent):
            # All silence, return short segment
            return audio[:int(0.1 * self.sample_rate)]
        
        # Find first and last non-silent samples
        first_sound = np.argmax(non_silent)
        last_sound = len(audio) - np.argmax(non_silent[::-1]) - 1
        
        # Add small padding
        padding = int(0.05 * self.sample_rate)  # 50ms
        start = max(0, first_sound - padding)
        end = min(len(audio), last_sound + padding)
        
        return audio[start:end]
    
    def _apply_basic_filter(self, audio: np.ndarray) -> np.ndarray:
        """Apply basic high-pass filter to remove DC offset and low-frequency noise."""
        # Simple high-pass filter (removes very low frequencies)
        if len(audio) < 2:
            return audio
            
        filtered = np.zeros_like(audio)
        alpha = 0.99  # High-pass filter coefficient
        
        filtered[0] = audio[0]
        for i in range(1, len(audio)):
            filtered[i] = alpha * filtered[i-1] + alpha * (audio[i] - audio[i-1])
        
        return filtered
    
    def _resample_audio(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Simple audio resampling."""
        if orig_sr == target_sr:
            return audio
            
        # Simple linear interpolation resampling
        ratio = target_sr / orig_sr
        new_length = int(len(audio) * ratio)
        
        old_indices = np.linspace(0, len(audio) - 1, new_length)
        resampled = np.interp(old_indices, np.arange(len(audio)), audio)
        
        logger.debug(f"Resampled: {orig_sr}Hz -> {target_sr}Hz ({len(audio)} -> {len(resampled)} samples)")
        return resampled
    
    def plot_comparison(self, audio1: np.ndarray, audio2: np.ndarray, 
                       labels: Optional[List[str]] = None, save_path: Optional[str] = None) -> None:
        """Plot before/after comparison of audio waveforms.
        
        Args:
            audio1: First audio signal
            audio2: Second audio signal
            labels: Labels for the audio signals
            save_path: Path to save plot (if None, attempts to display)
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available for plotting")
            print(f"Audio comparison: {len(audio1)} vs {len(audio2)} samples")
            print(f"RMS: {np.sqrt(np.mean(audio1**2)):.4f} vs {np.sqrt(np.mean(audio2**2)):.4f}")
            return
        
        labels = labels or ['Original', 'Processed']
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Time axes
        t1 = np.linspace(0, len(audio1) / self.sample_rate, len(audio1))
        t2 = np.linspace(0, len(audio2) / self.sample_rate, len(audio2))
        
        # Plot waveforms
        ax1.plot(t1, audio1, label=labels[0], alpha=0.8)
        ax1.set_title(f'{labels[0]} Waveform')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Amplitude')
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(t2, audio2, label=labels[1], alpha=0.8, color='orange')
        ax2.set_title(f'{labels[1]} Waveform')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Amplitude')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved comparison plot: {save_path}")
        else:
            try:
                plt.show()
            except:
                logger.warning("Could not display plot")
        
        plt.close()
    
    def extract_features(self, audio: np.ndarray) -> Dict[str, Any]:
        """Extract comprehensive audio features for analysis.
        
        Args:
            audio: Audio signal
            
        Returns:
            Dictionary with extracted features
        """
        features = {}
        
        # Basic statistics
        features.update(self.get_audio_stats(audio))
        
        # Spectral features
        try:
            if librosa is not None:
                # Spectral centroid
                spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)[0]
                features['spectral_centroid_mean'] = float(np.mean(spectral_centroids))
                features['spectral_centroid_std'] = float(np.std(spectral_centroids))
                
                # Spectral rolloff
                rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sample_rate)[0]
                features['spectral_rolloff_mean'] = float(np.mean(rolloff))
                
                # Zero crossing rate
                zcr = librosa.feature.zero_crossing_rate(audio)[0]
                features['zcr_mean'] = float(np.mean(zcr))
                
                # MFCCs
                mfccs = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=13)
                for i in range(min(5, mfccs.shape[0])):
                    features[f'mfcc_{i+1}_mean'] = float(np.mean(mfccs[i]))
                    features[f'mfcc_{i+1}_std'] = float(np.std(mfccs[i]))
                    
                # Chroma features
                chroma = librosa.feature.chroma_stft(y=audio, sr=self.sample_rate)
                features['chroma_mean'] = float(np.mean(chroma))
                
                # Tempo estimation
                tempo, beats = librosa.beat.beat_track(y=audio, sr=self.sample_rate)
                features['tempo'] = float(tempo)
                features['num_beats'] = len(beats)
                
            else:
                logger.warning("librosa not available, using basic spectral analysis")
                # Basic FFT-based features
                fft = np.fft.fft(audio)
                magnitude = np.abs(fft[:len(fft)//2])
                freqs = np.fft.fftfreq(len(fft), 1/self.sample_rate)[:len(fft)//2]
                
                # Spectral centroid approximation
                features['spectral_centroid_mean'] = float(np.sum(freqs * magnitude) / (np.sum(magnitude) + 1e-10))
                
                # Spectral energy in different bands
                low_band = magnitude[(freqs >= 20) & (freqs <= 200)]
                mid_band = magnitude[(freqs > 200) & (freqs <= 2000)]
                high_band = magnitude[(freqs > 2000) & (freqs <= 8000)]
                
                total_energy = np.sum(magnitude ** 2)
                features['low_band_energy'] = float(np.sum(low_band ** 2) / (total_energy + 1e-10))
                features['mid_band_energy'] = float(np.sum(mid_band ** 2) / (total_energy + 1e-10))
                features['high_band_energy'] = float(np.sum(high_band ** 2) / (total_energy + 1e-10))
                
        except Exception as e:
            logger.warning(f"Feature extraction error: {e}")
            
        return features
    
    def analyze_audio_quality(self, audio: np.ndarray) -> Dict[str, Any]:
        """Analyze audio quality metrics.
        
        Args:
            audio: Audio signal
            
        Returns:
            Dictionary with quality metrics
        """
        quality = {}
        
        # SNR estimation
        # Estimate noise floor from quietest 10% of samples
        sorted_abs = np.sort(np.abs(audio))
        noise_floor = np.mean(sorted_abs[:int(0.1 * len(sorted_abs))])
        signal_power = np.mean(audio ** 2)
        noise_power = noise_floor ** 2
        snr_db = 10 * np.log10((signal_power - noise_power) / (noise_power + 1e-10))
        quality['snr_db'] = float(snr_db)
        
        # THD estimation (simple harmonic analysis)
        if librosa is not None:
            try:
                # Pitch detection
                pitches, magnitudes = librosa.piptrack(y=audio, sr=self.sample_rate)
                fundamental_freq = 0
                
                # Find dominant frequency
                for t in range(pitches.shape[1]):
                    index = magnitudes[:, t].argmax()
                    pitch = pitches[index, t]
                    if pitch > 0:
                        fundamental_freq = pitch
                        break
                
                if fundamental_freq > 0:
                    # Simple THD calculation
                    fft = np.fft.fft(audio)
                    freqs = np.fft.fftfreq(len(fft), 1/self.sample_rate)
                    magnitude = np.abs(fft)
                    
                    # Find fundamental and harmonics
                    fundamental_idx = np.argmin(np.abs(freqs - fundamental_freq))
                    fundamental_power = magnitude[fundamental_idx] ** 2
                    
                    harmonic_power = 0
                    for harmonic in range(2, 6):  # 2nd to 5th harmonic
                        harmonic_freq = fundamental_freq * harmonic
                        if harmonic_freq < self.sample_rate / 2:
                            harmonic_idx = np.argmin(np.abs(freqs - harmonic_freq))
                            harmonic_power += magnitude[harmonic_idx] ** 2
                    
                    thd = harmonic_power / (fundamental_power + 1e-10)
                    quality['thd'] = float(thd)
                    quality['fundamental_freq'] = float(fundamental_freq)
                    
            except Exception as e:
                logger.debug(f"THD calculation error: {e}")
        
        # Clipping detection
        clipping_threshold = 0.99
        clipped_samples = np.sum(np.abs(audio) >= clipping_threshold)
        quality['clipping_ratio'] = float(clipped_samples / len(audio))
        
        # Dynamic range
        quality['crest_factor_db'] = float(20 * np.log10(np.max(np.abs(audio)) / (np.sqrt(np.mean(audio ** 2)) + 1e-10)))
        
        # Silence detection
        silence_threshold = 0.001
        silent_samples = np.sum(np.abs(audio) < silence_threshold)
        quality['silence_ratio'] = float(silent_samples / len(audio))
        
        return quality
    
    def enhance_audio(self, audio: np.ndarray, enhance_params: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """Apply audio enhancement processing.
        
        Args:
            audio: Input audio signal
            enhance_params: Enhancement parameters
            
        Returns:
            Enhanced audio signal
        """
        if enhance_params is None:
            enhance_params = {}
            
        enhanced = audio.copy()
        
        # Noise gate
        if enhance_params.get('noise_gate', True):
            enhanced = self._apply_noise_gate(enhanced, enhance_params.get('gate_threshold', self.noise_gate_threshold))
        
        # Dynamic range compression
        if enhance_params.get('compress', False):
            enhanced = self._apply_compressor(enhanced, 
                                            ratio=enhance_params.get('comp_ratio', self.compressor_ratio),
                                            threshold_db=enhance_params.get('comp_threshold', -20.0))
        
        # EQ
        if enhance_params.get('eq', False):
            eq_gains = enhance_params.get('eq_gains', {'low': 0, 'mid': 0, 'high': 0})
            enhanced = self._apply_eq(enhanced, eq_gains)
        
        # Normalization
        if enhance_params.get('normalize', True):
            enhanced = self.normalize_loudness(enhanced)
        
        logger.debug(f"Audio enhancement applied with params: {enhance_params}")
        return enhanced
    
    def _apply_noise_gate(self, audio: np.ndarray, threshold_db: float) -> np.ndarray:
        """Apply noise gate to reduce background noise."""
        threshold_linear = 10 ** (threshold_db / 20)
        
        # Simple gate: attenuate samples below threshold
        gate_ratio = 0.1  # Reduce by 90%
        mask = np.abs(audio) < threshold_linear
        gated = audio.copy()
        gated[mask] *= gate_ratio
        
        return gated
    
    def _apply_compressor(self, audio: np.ndarray, ratio: float, threshold_db: float) -> np.ndarray:
        """Apply dynamic range compression."""
        threshold_linear = 10 ** (threshold_db / 20)
        
        # Simple compressor
        compressed = audio.copy()
        over_threshold = np.abs(audio) > threshold_linear
        
        # Apply compression to samples over threshold
        for i in range(len(audio)):
            if over_threshold[i]:
                excess_db = 20 * np.log10(np.abs(audio[i]) / threshold_linear)
                compressed_excess_db = excess_db / ratio
                new_amplitude = threshold_linear * (10 ** (compressed_excess_db / 20))
                compressed[i] = np.sign(audio[i]) * new_amplitude
        
        return compressed
    
    def _apply_eq(self, audio: np.ndarray, gains: Dict[str, float]) -> np.ndarray:
        """Apply simple 3-band EQ."""
        if librosa is None:
            logger.warning("EQ requires librosa, returning original audio")
            return audio
            
        try:
            # Convert to frequency domain
            fft = np.fft.fft(audio)
            freqs = np.fft.fftfreq(len(fft), 1/self.sample_rate)
            
            # Apply gains to frequency bands
            for band, gain_db in gains.items():
                if band in self.eq_bands and gain_db != 0:
                    band_config = self.eq_bands[band]
                    center_freq = band_config['freq']
                    q = band_config['q']
                    
                    # Simple bell filter
                    gain_linear = 10 ** (gain_db / 20)
                    bandwidth = center_freq / q
                    
                    # Apply gain to frequency range
                    freq_mask = (np.abs(freqs) >= center_freq - bandwidth/2) & (np.abs(freqs) <= center_freq + bandwidth/2)
                    fft[freq_mask] *= gain_linear
            
            # Convert back to time domain
            eq_audio = np.real(np.fft.ifft(fft))
            return eq_audio.astype(np.float32)
            
        except Exception as e:
            logger.warning(f"EQ processing error: {e}")
            return audio
    
    def get_audio_stats(self, audio: np.ndarray) -> Dict[str, Any]:
        """Get comprehensive statistics about audio signal.
        
        Args:
            audio: Audio signal
            
        Returns:
            Dictionary with audio statistics
        """
        stats = {
            'duration_seconds': len(audio) / self.sample_rate,
            'sample_rate': self.sample_rate,
            'num_samples': len(audio),
            'rms': float(np.sqrt(np.mean(audio ** 2))),
            'peak': float(np.max(np.abs(audio))),
            'mean': float(np.mean(audio)),
            'std': float(np.std(audio)),
            'dynamic_range_db': float(20 * np.log10(np.max(np.abs(audio)) / (np.sqrt(np.mean(audio ** 2)) + 1e-10))),
            'zero_crossings': int(np.sum(np.diff(np.sign(audio)) != 0) / 2)
        }
        
        return stats