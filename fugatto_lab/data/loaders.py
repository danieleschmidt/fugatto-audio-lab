"""Data loaders and batch processing utilities."""

import logging
import numpy as np
from typing import Dict, Any, List, Optional, Generator, Callable, Union
from pathlib import Path
import random
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import time

from .dataset import AudioDataset
from ..core import AudioProcessor

logger = logging.getLogger(__name__)


class AudioDataLoader:
    """Data loader for efficient audio dataset iteration."""
    
    def __init__(self, dataset: AudioDataset, batch_size: int = 8,
                 shuffle: bool = True, num_workers: int = 0,
                 drop_last: bool = False, pin_memory: bool = False):
        """Initialize audio data loader.
        
        Args:
            dataset: Audio dataset to load from
            batch_size: Number of samples per batch
            shuffle: Whether to shuffle data between epochs
            num_workers: Number of worker processes for loading
            drop_last: Whether to drop incomplete last batch
            pin_memory: Whether to pin memory for faster GPU transfer
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = max(0, min(num_workers, mp.cpu_count()))
        self.drop_last = drop_last
        self.pin_memory = pin_memory
        
        self._indices = list(range(len(dataset)))
        self._current_epoch = 0
        
        logger.info(f"AudioDataLoader initialized: {len(dataset)} samples, "
                   f"batch_size={batch_size}, num_workers={self.num_workers}")
    
    def __len__(self) -> int:
        """Get number of batches in one epoch."""
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        else:
            return (len(self.dataset) + self.batch_size - 1) // self.batch_size
    
    def __iter__(self) -> Generator[Dict[str, Any], None, None]:
        """Iterate over batches."""
        # Shuffle indices if requested
        if self.shuffle:
            random.shuffle(self._indices)
        
        # Create batches
        for i in range(0, len(self._indices), self.batch_size):
            batch_indices = self._indices[i:i + self.batch_size]
            
            # Skip incomplete last batch if requested
            if self.drop_last and len(batch_indices) < self.batch_size:
                break
            
            # Load batch
            if self.num_workers > 0:
                batch = self._load_batch_parallel(batch_indices)
            else:
                batch = self._load_batch_sequential(batch_indices)
            
            yield batch
        
        self._current_epoch += 1
    
    def _load_batch_sequential(self, indices: List[int]) -> Dict[str, Any]:
        """Load batch sequentially."""
        batch_data = {
            'audio': [],
            'captions': [],
            'audio_paths': [],
            'durations': [],
            'sample_rates': [],
            'tags': [],
            'metadata': [],
            'indices': indices
        }
        
        for idx in indices:
            try:
                sample = self.dataset[idx]
                batch_data['audio'].append(sample['audio'])
                batch_data['captions'].append(sample['caption'])
                batch_data['audio_paths'].append(sample['audio_path'])
                batch_data['durations'].append(sample['duration_seconds'])
                batch_data['sample_rates'].append(sample['sample_rate'])
                batch_data['tags'].append(sample['tags'])
                batch_data['metadata'].append(sample['metadata'])
                
            except Exception as e:
                logger.error(f"Failed to load sample {idx}: {e}")
                # Use silence as fallback
                fallback_duration = 1.0
                fallback_sr = self.dataset.processor.sample_rate
                silence = np.zeros(int(fallback_duration * fallback_sr), dtype=np.float32)
                
                batch_data['audio'].append(silence)
                batch_data['captions'].append("ERROR: Failed to load audio")
                batch_data['audio_paths'].append("")
                batch_data['durations'].append(fallback_duration)
                batch_data['sample_rates'].append(fallback_sr)
                batch_data['tags'].append([])
                batch_data['metadata'].append({'error': str(e)})
        
        return batch_data
    
    def _load_batch_parallel(self, indices: List[int]) -> Dict[str, Any]:
        """Load batch using parallel workers."""
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [executor.submit(self._load_single_sample, idx) for idx in indices]
            samples = [future.result() for future in futures]
        
        # Organize into batch format
        batch_data = {
            'audio': [s['audio'] for s in samples],
            'captions': [s['caption'] for s in samples],
            'audio_paths': [s['audio_path'] for s in samples],
            'durations': [s['duration_seconds'] for s in samples],
            'sample_rates': [s['sample_rate'] for s in samples],
            'tags': [s['tags'] for s in samples],
            'metadata': [s['metadata'] for s in samples],
            'indices': indices
        }
        
        return batch_data
    
    def _load_single_sample(self, idx: int) -> Dict[str, Any]:
        """Load a single sample with error handling."""
        try:
            return self.dataset[idx]
        except Exception as e:
            logger.error(f"Failed to load sample {idx}: {e}")
            # Return fallback sample
            fallback_duration = 1.0
            fallback_sr = self.dataset.processor.sample_rate
            silence = np.zeros(int(fallback_duration * fallback_sr), dtype=np.float32)
            
            return {
                'audio': silence,
                'caption': "ERROR: Failed to load audio",
                'audio_path': "",
                'duration_seconds': fallback_duration,
                'sample_rate': fallback_sr,
                'tags': [],
                'metadata': {'error': str(e)},
                'index': idx
            }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get data loader statistics."""
        return {
            'dataset_size': len(self.dataset),
            'batch_size': self.batch_size,
            'num_batches': len(self),
            'num_workers': self.num_workers,
            'current_epoch': self._current_epoch,
            'shuffle': self.shuffle,
            'drop_last': self.drop_last
        }


class BatchProcessor:
    """Batch processor for efficient audio processing operations."""
    
    def __init__(self, processor: Optional[AudioProcessor] = None,
                 max_workers: int = 4):
        """Initialize batch processor.
        
        Args:
            processor: Audio processor instance
            max_workers: Maximum number of worker processes
        """
        self.processor = processor or AudioProcessor()
        self.max_workers = max_workers
        
        logger.info(f"BatchProcessor initialized with {max_workers} workers")
    
    def process_batch(self, audio_batch: List[np.ndarray],
                     operations: List[str],
                     operation_params: Optional[Dict[str, Any]] = None) -> List[np.ndarray]:
        """Process a batch of audio with specified operations.
        
        Args:
            audio_batch: List of audio arrays
            operations: List of operation names to apply
            operation_params: Parameters for operations
            
        Returns:
            List of processed audio arrays
        """
        operation_params = operation_params or {}
        
        if self.max_workers > 1:
            return self._process_batch_parallel(audio_batch, operations, operation_params)
        else:
            return self._process_batch_sequential(audio_batch, operations, operation_params)
    
    def _process_batch_sequential(self, audio_batch: List[np.ndarray],
                                operations: List[str],
                                operation_params: Dict[str, Any]) -> List[np.ndarray]:
        """Process batch sequentially."""
        processed_batch = []
        
        for audio in audio_batch:
            processed_audio = self._apply_operations(audio, operations, operation_params)
            processed_batch.append(processed_audio)
        
        return processed_batch
    
    def _process_batch_parallel(self, audio_batch: List[np.ndarray],
                              operations: List[str],
                              operation_params: Dict[str, Any]) -> List[np.ndarray]:
        """Process batch in parallel."""
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(self._apply_operations, audio, operations, operation_params)
                for audio in audio_batch
            ]
            processed_batch = [future.result() for future in futures]
        
        return processed_batch
    
    def _apply_operations(self, audio: np.ndarray, operations: List[str],
                         operation_params: Dict[str, Any]) -> np.ndarray:
        """Apply operations to single audio array."""
        processed = audio.copy()
        
        for operation in operations:
            if operation == 'normalize':
                target_lufs = operation_params.get('target_lufs', -14.0)
                processed = self.processor.normalize_loudness(processed, target_lufs)
            
            elif operation == 'trim_silence':
                threshold = operation_params.get('threshold', 0.01)
                processed = self.processor._trim_silence(processed, threshold)
            
            elif operation == 'filter':
                processed = self.processor._apply_basic_filter(processed)
            
            elif operation == 'resample':
                target_sr = operation_params.get('target_sample_rate', self.processor.sample_rate)
                original_sr = operation_params.get('original_sample_rate', self.processor.sample_rate)
                if target_sr != original_sr:
                    processed = self.processor._resample_audio(processed, original_sr, target_sr)
            
            elif operation == 'pad_or_trim':
                target_length = operation_params.get('target_length')
                if target_length:
                    if len(processed) > target_length:
                        processed = processed[:target_length]
                    elif len(processed) < target_length:
                        pad_length = target_length - len(processed)
                        processed = np.pad(processed, (0, pad_length), mode='constant')
            
            else:
                logger.warning(f"Unknown operation: {operation}")
        
        return processed
    
    def collate_batch(self, batch_data: Dict[str, List[Any]],
                     pad_audio: bool = True,
                     max_length: Optional[int] = None) -> Dict[str, Any]:
        """Collate batch data for training.
        
        Args:
            batch_data: Batch data dictionary
            pad_audio: Whether to pad audio to same length
            max_length: Maximum audio length (in samples)
            
        Returns:
            Collated batch data
        """
        audio_list = batch_data['audio']
        
        if pad_audio:
            # Find maximum length in batch
            if max_length is None:
                max_length = max(len(audio) for audio in audio_list)
            
            # Pad all audio to same length
            padded_audio = []
            for audio in audio_list:
                if len(audio) > max_length:
                    # Trim to max length
                    padded_audio.append(audio[:max_length])
                else:
                    # Pad to max length
                    pad_length = max_length - len(audio)
                    padded = np.pad(audio, (0, pad_length), mode='constant')
                    padded_audio.append(padded)
            
            # Stack into array
            audio_array = np.stack(padded_audio, axis=0)
        else:
            # Keep as list of varying lengths
            audio_array = audio_list
        
        collated = {
            'audio': audio_array,
            'captions': batch_data['captions'],
            'durations': np.array(batch_data['durations']),
            'sample_rates': np.array(batch_data['sample_rates']),
            'batch_size': len(batch_data['captions']),
            'max_length': max_length if pad_audio else None
        }
        
        # Add optional fields
        for key in ['tags', 'metadata', 'audio_paths', 'indices']:
            if key in batch_data:
                collated[key] = batch_data[key]
        
        return collated
    
    def create_data_loader(self, dataset: AudioDataset, **kwargs) -> AudioDataLoader:
        """Create data loader with default settings.
        
        Args:
            dataset: Audio dataset
            **kwargs: Additional data loader arguments
            
        Returns:
            Configured data loader
        """
        default_kwargs = {
            'batch_size': 8,
            'shuffle': True,
            'num_workers': min(4, mp.cpu_count()),
            'drop_last': False
        }
        default_kwargs.update(kwargs)
        
        return AudioDataLoader(dataset, **default_kwargs)
    
    def benchmark_loading(self, dataset: AudioDataset, num_batches: int = 10,
                         batch_size: int = 8) -> Dict[str, Any]:
        """Benchmark data loading performance.
        
        Args:
            dataset: Dataset to benchmark
            num_batches: Number of batches to load
            batch_size: Batch size for testing
            
        Returns:
            Benchmark results
        """
        logger.info(f"Benchmarking data loading: {num_batches} batches of size {batch_size}")
        
        # Test sequential loading
        data_loader_seq = AudioDataLoader(
            dataset, batch_size=batch_size, num_workers=0, shuffle=False
        )
        
        start_time = time.time()
        for i, batch in enumerate(data_loader_seq):
            if i >= num_batches:
                break
        seq_time = time.time() - start_time
        
        # Test parallel loading
        data_loader_par = AudioDataLoader(
            dataset, batch_size=batch_size, num_workers=self.max_workers, shuffle=False
        )
        
        start_time = time.time()
        for i, batch in enumerate(data_loader_par):
            if i >= num_batches:
                break
        par_time = time.time() - start_time
        
        results = {
            'sequential_time_seconds': seq_time,
            'parallel_time_seconds': par_time,
            'speedup': seq_time / par_time if par_time > 0 else 0,
            'batches_tested': num_batches,
            'batch_size': batch_size,
            'num_workers': self.max_workers,
            'samples_per_second_sequential': (num_batches * batch_size) / seq_time if seq_time > 0 else 0,
            'samples_per_second_parallel': (num_batches * batch_size) / par_time if par_time > 0 else 0
        }
        
        logger.info(f"Loading benchmark completed: {results['speedup']:.2f}x speedup with {self.max_workers} workers")
        return results


class DatasetSplitter:
    """Utility for splitting datasets into train/validation/test sets."""
    
    @staticmethod
    def split_dataset(dataset: AudioDataset, 
                     train_ratio: float = 0.8,
                     val_ratio: float = 0.1,
                     test_ratio: float = 0.1,
                     shuffle: bool = True,
                     random_seed: Optional[int] = None) -> Dict[str, AudioDataset]:
        """Split dataset into train/validation/test sets.
        
        Args:
            dataset: Dataset to split
            train_ratio: Proportion for training set
            val_ratio: Proportion for validation set  
            test_ratio: Proportion for test set
            shuffle: Whether to shuffle before splitting
            random_seed: Random seed for reproducibility
            
        Returns:
            Dictionary with 'train', 'val', 'test' datasets
        """
        # Validate ratios
        total_ratio = train_ratio + val_ratio + test_ratio
        if abs(total_ratio - 1.0) > 1e-6:
            raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")
        
        # Get indices
        indices = list(range(len(dataset)))
        
        if shuffle:
            if random_seed is not None:
                random.seed(random_seed)
            random.shuffle(indices)
        
        # Calculate split points
        total_samples = len(dataset)
        train_size = int(total_samples * train_ratio)
        val_size = int(total_samples * val_ratio)
        
        # Split indices
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
        
        # Create subset datasets
        train_samples = [dataset.samples[i] for i in train_indices]
        val_samples = [dataset.samples[i] for i in val_indices]
        test_samples = [dataset.samples[i] for i in test_indices]
        
        train_dataset = AudioDataset(train_samples, dataset.processor)
        val_dataset = AudioDataset(val_samples, dataset.processor)
        test_dataset = AudioDataset(test_samples, dataset.processor)
        
        logger.info(f"Dataset split: train={len(train_dataset)}, "
                   f"val={len(val_dataset)}, test={len(test_dataset)}")
        
        return {
            'train': train_dataset,
            'val': val_dataset,
            'test': test_dataset
        }
    
    @staticmethod
    def stratified_split(dataset: AudioDataset,
                        stratify_by: str = 'duration',
                        train_ratio: float = 0.8,
                        val_ratio: float = 0.1,
                        test_ratio: float = 0.1) -> Dict[str, AudioDataset]:
        """Split dataset with stratification to maintain distribution.
        
        Args:
            dataset: Dataset to split
            stratify_by: Field to stratify by ('duration', 'model', 'tags')
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
            test_ratio: Test set ratio
            
        Returns:
            Dictionary with stratified splits
        """
        # Create strata based on the stratification field
        if stratify_by == 'duration':
            # Create duration buckets
            durations = [sample.duration_seconds for sample in dataset.samples]
            percentiles = np.percentile(durations, [33, 66])
            
            def get_stratum(sample):
                if sample.duration_seconds <= percentiles[0]:
                    return 'short'
                elif sample.duration_seconds <= percentiles[1]:
                    return 'medium'
                else:
                    return 'long'
        
        elif stratify_by == 'tags':
            # Use most common tag as stratum
            def get_stratum(sample):
                if sample.tags:
                    return sample.tags[0]
                else:
                    return 'no_tags'
        
        else:
            raise ValueError(f"Unsupported stratification field: {stratify_by}")
        
        # Group samples by stratum
        strata = {}
        for sample in dataset.samples:
            stratum = get_stratum(sample)
            if stratum not in strata:
                strata[stratum] = []
            strata[stratum].append(sample)
        
        # Split each stratum
        train_samples = []
        val_samples = []
        test_samples = []
        
        for stratum_samples in strata.values():
            stratum_dataset = AudioDataset(stratum_samples, dataset.processor)
            splits = DatasetSplitter.split_dataset(
                stratum_dataset, train_ratio, val_ratio, test_ratio, shuffle=True
            )
            
            train_samples.extend(splits['train'].samples)
            val_samples.extend(splits['val'].samples)
            test_samples.extend(splits['test'].samples)
        
        # Create final datasets
        train_dataset = AudioDataset(train_samples, dataset.processor)
        val_dataset = AudioDataset(val_samples, dataset.processor)
        test_dataset = AudioDataset(test_samples, dataset.processor)
        
        logger.info(f"Stratified split by {stratify_by}: train={len(train_dataset)}, "
                   f"val={len(val_dataset)}, test={len(test_dataset)}")
        
        return {
            'train': train_dataset,
            'val': val_dataset,
            'test': test_dataset
        }