"""Advanced data processing and transformation pipelines."""

import logging
import numpy as np
import json
from typing import Dict, Any, List, Optional, Tuple, Iterator, Union
from pathlib import Path
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
import queue
import time
from datetime import datetime, timedelta

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    pd = None
    PANDAS_AVAILABLE = False

try:
    import librosa
    import soundfile as sf
    AUDIO_LIBS_AVAILABLE = True
except ImportError:
    librosa = None
    sf = None
    AUDIO_LIBS_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class ProcessingJob:
    """Represents a data processing job."""
    job_id: str
    job_type: str
    input_data: Any
    parameters: Dict[str, Any]
    priority: int = 0
    created_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: str = 'pending'
    result: Any = None
    error: Optional[str] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


class AudioDataProcessor:
    """Advanced audio data processing with batch capabilities."""
    
    def __init__(self, sample_rate: int = 48000, max_workers: int = 4):
        """Initialize audio data processor.
        
        Args:
            sample_rate: Target sample rate for processing
            max_workers: Maximum number of worker threads
        """
        self.sample_rate = sample_rate
        self.max_workers = max_workers
        self.processing_stats = {
            'files_processed': 0,
            'total_duration': 0.0,
            'errors': 0,
            'processing_time': 0.0
        }
        
        logger.info(f"AudioDataProcessor initialized: {sample_rate}Hz, {max_workers} workers")
    
    def process_audio_dataset(self, audio_paths: List[Path], 
                            output_dir: Path,
                            processing_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process a dataset of audio files with parallel processing.
        
        Args:
            audio_paths: List of paths to audio files
            output_dir: Directory to save processed files
            processing_config: Processing configuration
            
        Returns:
            Processing results and statistics
        """
        if not AUDIO_LIBS_AVAILABLE:
            logger.error("Audio libraries not available for dataset processing")
            return {'error': 'Audio libraries not available'}
        
        config = processing_config or {
            'normalize': True,
            'trim_silence': True,
            'target_sample_rate': self.sample_rate,
            'output_format': 'wav'
        }
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        start_time = time.time()
        results = []
        
        logger.info(f"Processing {len(audio_paths)} audio files")
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all processing jobs
            future_to_path = {
                executor.submit(self._process_single_audio, path, output_dir, config): path
                for path in audio_paths
            }
            
            # Collect results
            for future in future_to_path:
                path = future_to_path[future]
                try:
                    result = future.result(timeout=300)  # 5 minute timeout per file
                    result['input_path'] = str(path)
                    results.append(result)
                    
                    if result.get('success', False):
                        self.processing_stats['files_processed'] += 1
                        self.processing_stats['total_duration'] += result.get('duration', 0)
                    else:
                        self.processing_stats['errors'] += 1
                        
                except Exception as e:
                    logger.error(f"Error processing {path}: {e}")
                    results.append({
                        'input_path': str(path),
                        'success': False,
                        'error': str(e)
                    })
                    self.processing_stats['errors'] += 1
        
        processing_time = time.time() - start_time
        self.processing_stats['processing_time'] += processing_time
        
        # Generate processing report
        successful = sum(1 for r in results if r.get('success', False))
        failed = len(results) - successful
        
        report = {
            'total_files': len(audio_paths),
            'successful': successful,
            'failed': failed,
            'processing_time_seconds': processing_time,
            'average_time_per_file': processing_time / len(audio_paths) if audio_paths else 0,
            'results': results,
            'statistics': self.processing_stats.copy()
        }
        
        # Save processing report
        report_path = output_dir / 'processing_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Dataset processing completed: {successful}/{len(audio_paths)} successful")
        return report
    
    def _process_single_audio(self, input_path: Path, output_dir: Path, 
                            config: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single audio file."""
        try:
            start_time = time.time()
            
            # Load audio
            audio, sr = librosa.load(str(input_path), sr=config.get('target_sample_rate', self.sample_rate))
            
            # Apply processing steps
            processed_audio = audio.copy()
            
            if config.get('normalize', True):
                processed_audio = self._normalize_audio(processed_audio)
            
            if config.get('trim_silence', True):
                processed_audio = self._trim_silence(processed_audio)
            
            if config.get('apply_filters', False):
                processed_audio = self._apply_audio_filters(processed_audio, config.get('filter_params', {}))
            
            # Generate output filename
            output_format = config.get('output_format', 'wav')
            output_filename = input_path.stem + f'_processed.{output_format}'
            output_path = output_dir / output_filename
            
            # Save processed audio
            sf.write(str(output_path), processed_audio, sr, format=output_format.upper())
            
            processing_time = time.time() - start_time
            
            # Extract features for metadata
            features = self._extract_audio_features(processed_audio, sr)
            
            return {
                'success': True,
                'output_path': str(output_path),
                'duration': len(processed_audio) / sr,
                'sample_rate': sr,
                'processing_time': processing_time,
                'features': features,
                'original_duration': len(audio) / sr,
                'size_reduction_ratio': len(processed_audio) / len(audio) if len(audio) > 0 else 1.0
            }
            
        except Exception as e:
            logger.error(f"Error processing {input_path}: {e}")
            return {
                'success': False,
                'error': str(e),
                'processing_time': time.time() - start_time if 'start_time' in locals() else 0
            }
    
    def _normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """Normalize audio to prevent clipping."""
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            return audio / max_val * 0.95  # Leave some headroom
        return audio
    
    def _trim_silence(self, audio: np.ndarray, threshold: float = 0.01) -> np.ndarray:
        """Remove leading and trailing silence."""
        non_silent = np.abs(audio) > threshold
        
        if not np.any(non_silent):
            return audio[:int(0.1 * self.sample_rate)]  # Return short segment if all silence
        
        first_sound = np.argmax(non_silent)
        last_sound = len(audio) - np.argmax(non_silent[::-1]) - 1
        
        # Add small padding
        padding = int(0.05 * self.sample_rate)
        start = max(0, first_sound - padding)
        end = min(len(audio), last_sound + padding)
        
        return audio[start:end]
    
    def _apply_audio_filters(self, audio: np.ndarray, filter_params: Dict[str, Any]) -> np.ndarray:
        """Apply audio filters based on parameters."""
        filtered = audio.copy()
        
        # High-pass filter to remove DC offset
        if filter_params.get('highpass', True):
            if len(filtered) > 1:
                alpha = 0.99
                for i in range(1, len(filtered)):
                    filtered[i] = alpha * filtered[i-1] + alpha * (audio[i] - audio[i-1])
        
        # Simple low-pass filter for noise reduction
        if filter_params.get('lowpass', False):
            cutoff_ratio = filter_params.get('lowpass_cutoff', 0.5)
            if len(filtered) > 2:
                for i in range(1, len(filtered) - 1):
                    filtered[i] = filtered[i] * cutoff_ratio + filtered[i-1] * (1 - cutoff_ratio) * 0.5 + filtered[i+1] * (1 - cutoff_ratio) * 0.5
        
        return filtered
    
    def _extract_audio_features(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """Extract basic audio features."""
        features = {
            'duration': len(audio) / sr,
            'rms': float(np.sqrt(np.mean(audio ** 2))),
            'peak': float(np.max(np.abs(audio))),
            'zero_crossings': int(np.sum(np.diff(np.sign(audio)) != 0) / 2)
        }
        
        if librosa is not None:
            try:
                # Spectral features
                spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
                features['spectral_centroid_mean'] = float(np.mean(spectral_centroids))
                
                # Tempo
                tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
                features['tempo'] = float(tempo)
                
            except Exception as e:
                logger.debug(f"Feature extraction error: {e}")
        
        return features
    
    def create_audio_manifest(self, dataset_dir: Path, output_path: Path) -> Dict[str, Any]:
        """Create a manifest file for an audio dataset.
        
        Args:
            dataset_dir: Directory containing audio files
            output_path: Path to save manifest
            
        Returns:
            Manifest data
        """
        logger.info(f"Creating audio manifest for {dataset_dir}")
        
        # Find all audio files
        audio_extensions = ['.wav', '.mp3', '.flac', '.ogg', '.m4a']
        audio_files = []
        
        for ext in audio_extensions:
            audio_files.extend(dataset_dir.glob(f'**/*{ext}'))
            audio_files.extend(dataset_dir.glob(f'**/*{ext.upper()}'))
        
        manifest_entries = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_file = {
                executor.submit(self._analyze_audio_file, file_path): file_path
                for file_path in audio_files
            }
            
            for future in future_to_file:
                file_path = future_to_file[future]
                try:
                    analysis = future.result(timeout=60)
                    if analysis:
                        manifest_entries.append(analysis)
                except Exception as e:
                    logger.warning(f"Failed to analyze {file_path}: {e}")
        
        # Create manifest
        manifest = {
            'dataset_dir': str(dataset_dir),
            'created_at': datetime.now().isoformat(),
            'total_files': len(manifest_entries),
            'total_duration': sum(entry.get('duration', 0) for entry in manifest_entries),
            'sample_rates': list(set(entry.get('sample_rate', 0) for entry in manifest_entries)),
            'files': manifest_entries
        }
        
        # Save manifest
        with open(output_path, 'w') as f:
            json.dump(manifest, f, indent=2, default=str)
        
        logger.info(f"Created manifest with {len(manifest_entries)} entries: {output_path}")
        return manifest
    
    def _analyze_audio_file(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Analyze a single audio file for manifest creation."""
        try:
            if not AUDIO_LIBS_AVAILABLE:
                # Basic file info without audio analysis
                stat = file_path.stat()
                return {
                    'file_path': str(file_path.relative_to(file_path.parents[1])),
                    'filename': file_path.name,
                    'size_bytes': stat.st_size,
                    'modified_at': datetime.fromtimestamp(stat.st_mtime).isoformat()
                }
            
            # Load audio info without loading full file
            info = sf.info(str(file_path))
            
            analysis = {
                'file_path': str(file_path.relative_to(file_path.parents[1])),
                'filename': file_path.name,
                'duration': info.duration,
                'sample_rate': info.samplerate,
                'channels': info.channels,
                'format': info.format,
                'subtype': info.subtype,
                'size_bytes': file_path.stat().st_size
            }
            
            # Load small sample for feature extraction
            sample_audio, sr = librosa.load(str(file_path), duration=10.0, sr=info.samplerate)
            features = self._extract_audio_features(sample_audio, sr)
            analysis['features'] = features
            
            return analysis
            
        except Exception as e:
            logger.debug(f"Error analyzing {file_path}: {e}")
            return None


class DatasetSplitter:
    """Split datasets for training, validation, and testing."""
    
    def __init__(self, random_seed: int = 42):
        """Initialize dataset splitter.
        
        Args:
            random_seed: Random seed for reproducible splits
        """
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
        logger.info(f"DatasetSplitter initialized with seed: {random_seed}")
    
    def split_dataset(self, manifest_path: Path, 
                     split_ratios: Dict[str, float] = None,
                     output_dir: Optional[Path] = None) -> Dict[str, List[Dict[str, Any]]]:
        """Split dataset based on manifest file.
        
        Args:
            manifest_path: Path to dataset manifest
            split_ratios: Ratios for train/val/test splits
            output_dir: Directory to save split manifests
            
        Returns:
            Dictionary with split datasets
        """
        if split_ratios is None:
            split_ratios = {'train': 0.7, 'validation': 0.2, 'test': 0.1}
        
        # Validate split ratios
        total_ratio = sum(split_ratios.values())
        if abs(total_ratio - 1.0) > 0.001:
            raise ValueError(f"Split ratios must sum to 1.0, got {total_ratio}")
        
        logger.info(f"Splitting dataset with ratios: {split_ratios}")
        
        # Load manifest
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        
        files = manifest['files']
        np.random.shuffle(files)  # Shuffle for random split
        
        # Calculate split indices
        n_files = len(files)
        splits = {}
        current_idx = 0
        
        for split_name, ratio in split_ratios.items():
            split_size = int(n_files * ratio)
            if split_name == list(split_ratios.keys())[-1]:  # Last split gets remaining files
                split_files = files[current_idx:]
            else:
                split_files = files[current_idx:current_idx + split_size]
            
            splits[split_name] = split_files
            current_idx += split_size
        
        # Create split manifests
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            
            for split_name, split_files in splits.items():
                split_manifest = manifest.copy()
                split_manifest['files'] = split_files
                split_manifest['split_name'] = split_name
                split_manifest['split_size'] = len(split_files)
                split_manifest['split_duration'] = sum(f.get('duration', 0) for f in split_files)
                
                split_path = output_dir / f'{split_name}_manifest.json'
                with open(split_path, 'w') as f:
                    json.dump(split_manifest, f, indent=2, default=str)
                
                logger.info(f"Created {split_name} split: {len(split_files)} files -> {split_path}")
        
        # Log split statistics
        for split_name, split_files in splits.items():
            duration = sum(f.get('duration', 0) for f in split_files)
            logger.info(f"{split_name}: {len(split_files)} files ({duration:.1f}s)")
        
        return splits
    
    def balance_dataset(self, manifest_path: Path, 
                       balance_key: str = 'duration',
                       target_distribution: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """Balance dataset based on specified criteria.
        
        Args:
            manifest_path: Path to dataset manifest
            balance_key: Key to balance on (e.g., 'duration', 'sample_rate')
            target_distribution: Target distribution for balancing
            
        Returns:
            Balanced dataset information
        """
        logger.info(f"Balancing dataset on key: {balance_key}")
        
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        
        files = manifest['files']
        
        # Group files by balance key
        groups = {}
        for file_info in files:
            if balance_key == 'duration':
                # Group by duration ranges
                duration = file_info.get('duration', 0)
                if duration < 5:
                    group = 'short'
                elif duration < 15:
                    group = 'medium'
                else:
                    group = 'long'
            else:
                group = str(file_info.get(balance_key, 'unknown'))
            
            if group not in groups:
                groups[group] = []
            groups[group].append(file_info)
        
        # Balance groups
        if target_distribution:
            balanced_files = self._apply_target_distribution(groups, target_distribution)
        else:
            balanced_files = self._auto_balance_groups(groups)
        
        # Create balanced manifest
        balanced_manifest = manifest.copy()
        balanced_manifest['files'] = balanced_files
        balanced_manifest['balanced_on'] = balance_key
        balanced_manifest['original_size'] = len(files)
        balanced_manifest['balanced_size'] = len(balanced_files)
        
        logger.info(f"Balanced dataset: {len(files)} -> {len(balanced_files)} files")
        return balanced_manifest
    
    def _apply_target_distribution(self, groups: Dict[str, List], 
                                 target_dist: Dict[str, float]) -> List[Dict[str, Any]]:
        """Apply target distribution to groups."""
        total_files = sum(len(group) for group in groups.values())
        balanced_files = []
        
        for group_name, target_ratio in target_dist.items():
            if group_name in groups:
                target_count = int(total_files * target_ratio)
                group_files = groups[group_name]
                
                if len(group_files) >= target_count:
                    # Randomly sample if we have more files than needed
                    sampled = np.random.choice(len(group_files), target_count, replace=False)
                    balanced_files.extend([group_files[i] for i in sampled])
                else:
                    # Use all files if we don't have enough
                    balanced_files.extend(group_files)
        
        return balanced_files
    
    def _auto_balance_groups(self, groups: Dict[str, List]) -> List[Dict[str, Any]]:
        """Automatically balance groups to have equal representation."""
        min_group_size = min(len(group) for group in groups.values())
        balanced_files = []
        
        for group_files in groups.values():
            if len(group_files) > min_group_size:
                # Randomly sample to match smallest group
                sampled = np.random.choice(len(group_files), min_group_size, replace=False)
                balanced_files.extend([group_files[i] for i in sampled])
            else:
                balanced_files.extend(group_files)
        
        return balanced_files


class ProcessingQueue:
    """Queue system for managing data processing jobs."""
    
    def __init__(self, max_workers: int = 4, max_queue_size: int = 1000):
        """Initialize processing queue.
        
        Args:
            max_workers: Maximum number of worker threads
            max_queue_size: Maximum number of jobs in queue
        """
        self.max_workers = max_workers
        self.max_queue_size = max_queue_size
        self.job_queue = queue.PriorityQueue(maxsize=max_queue_size)
        self.results = {}
        self.job_counter = 0
        self.workers = []
        self.shutdown_event = threading.Event()
        
        # Start worker threads
        for i in range(max_workers):
            worker = threading.Thread(target=self._worker, daemon=True)
            worker.start()
            self.workers.append(worker)
        
        logger.info(f"ProcessingQueue initialized: {max_workers} workers, queue size {max_queue_size}")
    
    def submit_job(self, job_type: str, input_data: Any, 
                  parameters: Dict[str, Any] = None, priority: int = 0) -> str:
        """Submit a processing job to the queue.
        
        Args:
            job_type: Type of processing job
            input_data: Input data for processing
            parameters: Processing parameters
            priority: Job priority (lower = higher priority)
            
        Returns:
            Job ID
        """
        self.job_counter += 1
        job_id = f"job_{self.job_counter:06d}"
        
        job = ProcessingJob(
            job_id=job_id,
            job_type=job_type,
            input_data=input_data,
            parameters=parameters or {},
            priority=priority
        )
        
        try:
            # Use negative priority for proper ordering (lower number = higher priority)
            self.job_queue.put((-priority, job), timeout=1.0)
            logger.info(f"Submitted job {job_id} (type: {job_type}, priority: {priority})")
            return job_id
        except queue.Full:
            logger.error("Job queue is full, cannot submit job")
            raise RuntimeError("Job queue is full")
    
    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a job.
        
        Args:
            job_id: Job identifier
            
        Returns:
            Job status information
        """
        if job_id in self.results:
            job = self.results[job_id]
            return {
                'job_id': job.job_id,
                'job_type': job.job_type,
                'status': job.status,
                'created_at': job.created_at,
                'started_at': job.started_at,
                'completed_at': job.completed_at,
                'error': job.error
            }
        return None
    
    def get_result(self, job_id: str, timeout: Optional[float] = None) -> Any:
        """Get result of a completed job.
        
        Args:
            job_id: Job identifier
            timeout: Maximum time to wait for result
            
        Returns:
            Job result
        """
        start_time = time.time()
        
        while True:
            if job_id in self.results:
                job = self.results[job_id]
                if job.status == 'completed':
                    return job.result
                elif job.status == 'failed':
                    raise RuntimeError(f"Job {job_id} failed: {job.error}")
            
            if timeout and (time.time() - start_time) > timeout:
                raise TimeoutError(f"Job {job_id} did not complete within {timeout} seconds")
            
            time.sleep(0.1)  # Check every 100ms
    
    def _worker(self):
        """Worker thread for processing jobs."""
        while not self.shutdown_event.is_set():
            try:
                # Get job from queue with timeout
                priority, job = self.job_queue.get(timeout=1.0)
                job.started_at = datetime.now()
                job.status = 'running'
                
                logger.info(f"Processing job {job.job_id} (type: {job.job_type})")
                
                try:
                    # Process job based on type
                    result = self._process_job(job)
                    job.result = result
                    job.status = 'completed'
                    
                except Exception as e:
                    logger.error(f"Job {job.job_id} failed: {e}")
                    job.error = str(e)
                    job.status = 'failed'
                
                finally:
                    job.completed_at = datetime.now()
                    self.results[job.job_id] = job
                    self.job_queue.task_done()
                    
            except queue.Empty:
                continue  # Timeout, check shutdown event
            except Exception as e:
                logger.error(f"Worker error: {e}")
    
    def _process_job(self, job: ProcessingJob) -> Any:
        """Process a job based on its type."""
        if job.job_type == 'audio_analysis':
            return self._process_audio_analysis(job.input_data, job.parameters)
        elif job.job_type == 'audio_conversion':
            return self._process_audio_conversion(job.input_data, job.parameters)
        elif job.job_type == 'feature_extraction':
            return self._process_feature_extraction(job.input_data, job.parameters)
        else:
            raise ValueError(f"Unknown job type: {job.job_type}")
    
    def _process_audio_analysis(self, input_data: Any, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Process audio analysis job."""
        # Mock implementation - would use actual audio analyzer
        time.sleep(parameters.get('processing_time', 1.0))  # Simulate processing
        return {
            'analysis_type': 'comprehensive',
            'features': {'duration': 10.0, 'rms': 0.5},
            'processing_time': parameters.get('processing_time', 1.0)
        }
    
    def _process_audio_conversion(self, input_data: Any, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Process audio conversion job.""" 
        # Mock implementation
        time.sleep(parameters.get('processing_time', 2.0))
        return {
            'conversion_type': parameters.get('format', 'wav'),
            'output_path': parameters.get('output_path', '/tmp/converted.wav'),
            'success': True
        }
    
    def _process_feature_extraction(self, input_data: Any, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Process feature extraction job."""
        # Mock implementation
        time.sleep(parameters.get('processing_time', 0.5))
        return {
            'features': {
                'mfcc': [0.1, 0.2, 0.3],
                'spectral_centroid': 1000.0,
                'tempo': 120.0
            },
            'extraction_method': parameters.get('method', 'librosa')
        }
    
    def shutdown(self, timeout: float = 10.0):
        """Shutdown the processing queue.
        
        Args:
            timeout: Maximum time to wait for workers to finish
        """
        logger.info("Shutting down processing queue")
        
        self.shutdown_event.set()
        
        # Wait for workers to finish
        for worker in self.workers:
            worker.join(timeout=timeout/len(self.workers))
        
        logger.info("Processing queue shutdown complete")
    
    def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        completed_jobs = sum(1 for job in self.results.values() if job.status == 'completed')
        failed_jobs = sum(1 for job in self.results.values() if job.status == 'failed')
        
        return {
            'queue_size': self.job_queue.qsize(),
            'max_queue_size': self.max_queue_size,
            'total_jobs_submitted': self.job_counter,
            'completed_jobs': completed_jobs,
            'failed_jobs': failed_jobs,
            'active_workers': len(self.workers),
            'results_cached': len(self.results)
        }