"""Tests for data processing components."""

import pytest
import numpy as np
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch

from fugatto_lab.data import AudioDataset, DatasetPreprocessor, AudioDataLoader, BatchProcessor
from fugatto_lab.data.dataset import AudioSample
from fugatto_lab.data.loaders import DatasetSplitter
from fugatto_lab.core import AudioProcessor


class TestAudioDataset:
    """Test the AudioDataset class."""
    
    @pytest.fixture
    def sample_audio_files(self):
        """Create temporary audio files for testing."""
        files = []
        
        for i in range(3):
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                # Write fake audio data
                tmp.write(b"fake audio data " + str(i).encode())
                files.append(tmp.name)
        
        yield files
        
        # Cleanup
        for file in files:
            Path(file).unlink(missing_ok=True)
    
    @pytest.fixture
    def mock_processor(self):
        """Mock audio processor for testing."""
        processor = Mock(spec=AudioProcessor)
        processor.sample_rate = 48000
        processor.load_audio.return_value = np.random.randn(48000).astype(np.float32)
        return processor
    
    @pytest.fixture
    def sample_dataset(self, sample_audio_files, mock_processor):
        """Create a sample dataset for testing."""
        samples = []
        captions = ["A cat meowing", "A dog barking", "Birds singing"]
        tags_list = [["cat", "animal"], ["dog", "animal"], ["bird", "nature"]]
        
        for i, (file, caption, tags) in enumerate(zip(sample_audio_files, captions, tags_list)):
            sample = AudioSample(
                audio_path=file,
                caption=caption,
                duration_seconds=float(i + 1),
                sample_rate=48000,
                tags=tags,
                metadata={"index": i}
            )
            samples.append(sample)
        
        return AudioDataset(samples, mock_processor)
    
    def test_dataset_initialization(self, sample_dataset):
        """Test dataset initialization."""
        assert len(sample_dataset) == 3
        assert sample_dataset.cache_enabled == True
    
    def test_dataset_getitem(self, sample_dataset):
        """Test getting items from dataset."""
        item = sample_dataset[0]
        
        assert isinstance(item, dict)
        assert 'audio' in item
        assert 'caption' in item
        assert 'audio_path' in item
        assert 'duration_seconds' in item
        assert 'tags' in item
        assert 'metadata' in item
        assert 'index' in item
        
        assert item['caption'] == "A cat meowing"
        assert item['tags'] == ["cat", "animal"]
        assert item['index'] == 0
    
    def test_dataset_getitem_out_of_range(self, sample_dataset):
        """Test getting out of range item."""
        with pytest.raises(IndexError):
            sample_dataset[10]
    
    def test_dataset_get_batch(self, sample_dataset):
        """Test getting multiple items."""
        batch = sample_dataset.get_batch([0, 2])
        
        assert len(batch) == 2
        assert batch[0]['caption'] == "A cat meowing"
        assert batch[1]['caption'] == "Birds singing"
    
    def test_filter_by_duration(self, sample_dataset):
        """Test filtering by duration."""
        # Original dataset has durations 1.0, 2.0, 3.0
        filtered = sample_dataset.filter_by_duration(min_duration=1.5, max_duration=2.5)
        
        assert len(filtered) == 1
        assert filtered[0]['duration_seconds'] == 2.0
    
    def test_filter_by_tags(self, sample_dataset):
        """Test filtering by tags."""
        # Filter for samples with "animal" tag
        animal_dataset = sample_dataset.filter_by_tags(["animal"], any_tag=True)
        
        assert len(animal_dataset) == 2  # cat and dog
        
        # Filter for samples with both "animal" and "cat" tags
        cat_dataset = sample_dataset.filter_by_tags(["animal", "cat"], any_tag=False)
        
        assert len(cat_dataset) == 1  # only cat
    
    def test_get_statistics(self, sample_dataset):
        """Test dataset statistics."""
        stats = sample_dataset.get_statistics()
        
        assert stats['total_samples'] == 3
        assert stats['avg_duration_seconds'] == 2.0  # (1+2+3)/3
        assert stats['min_duration_seconds'] == 1.0
        assert stats['max_duration_seconds'] == 3.0
        assert stats['unique_tags'] == 3  # cat, dog, bird, animal, nature
        assert stats['most_common_sample_rate'] == 48000
    
    def test_save_and_load_json(self, sample_dataset):
        """Test saving and loading dataset to/from JSON."""
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
            json_path = tmp.name
        
        try:
            # Save dataset
            sample_dataset.save_to_json(json_path)
            
            # Verify file exists and has content
            assert Path(json_path).exists()
            
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            assert data['total_samples'] == 3
            assert len(data['samples']) == 3
            
            # Load dataset
            loaded_dataset = AudioDataset.from_json(json_path, sample_dataset.processor)
            
            assert len(loaded_dataset) == 3
            assert loaded_dataset[0]['caption'] == "A cat meowing"
            
        finally:
            Path(json_path).unlink(missing_ok=True)
    
    def test_cache_operations(self, sample_dataset):
        """Test cache operations."""
        # Access items to populate cache
        sample_dataset[0]
        sample_dataset[1]
        
        # Clear cache
        sample_dataset.clear_cache()
        
        # Should still work after cache clear
        item = sample_dataset[0]
        assert item['caption'] == "A cat meowing"


class TestDatasetPreprocessor:
    """Test the DatasetPreprocessor class."""
    
    @pytest.fixture
    def mock_audio_dir(self):
        """Create mock audio directory structure."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            audio_dir = Path(tmp_dir) / "audio"
            audio_dir.mkdir()
            
            # Create fake audio files
            for i in range(3):
                audio_file = audio_dir / f"audio_{i}.wav"
                audio_file.write_bytes(b"fake audio data")
            
            yield audio_dir
    
    @pytest.fixture
    def mock_captions_file(self, mock_audio_dir):
        """Create mock captions file."""
        captions_data = [
            {
                "audio_file": "audio_0.wav",
                "caption": "First audio",
                "tags": ["test"],
                "metadata": {"quality": "high"}
            },
            {
                "audio_file": "audio_1.wav", 
                "caption": "Second audio",
                "tags": ["test", "sample"],
                "metadata": {"quality": "medium"}
            },
            {
                "audio_file": "audio_2.wav",
                "caption": "Third audio", 
                "tags": ["sample"],
                "metadata": {"quality": "low"}
            }
        ]
        
        captions_file = mock_audio_dir.parent / "captions.json"
        with open(captions_file, 'w') as f:
            json.dump(captions_data, f)
        
        yield captions_file
    
    @patch('fugatto_lab.data.dataset.AudioProcessor')
    def test_preprocessor_initialization(self, mock_processor_class):
        """Test preprocessor initialization."""
        preprocessor = DatasetPreprocessor(
            sample_rate=44100,
            normalize_loudness=True,
            target_lufs=-16.0,
            max_duration=25.0
        )
        
        assert preprocessor.sample_rate == 44100
        assert preprocessor.normalize_loudness == True
        assert preprocessor.target_lufs == -16.0
        assert preprocessor.max_duration == 25.0
    
    @patch('fugatto_lab.data.dataset.AudioProcessor')
    def test_prepare_dataset(self, mock_processor_class, mock_audio_dir, mock_captions_file):
        """Test dataset preparation."""
        # Mock processor
        mock_processor = Mock()
        mock_processor.sample_rate = 48000
        mock_processor.load_audio.return_value = np.random.randn(48000).astype(np.float32)
        mock_processor.preprocess.return_value = np.random.randn(48000).astype(np.float32)
        mock_processor.save_audio.return_value = None
        mock_processor_class.return_value = mock_processor
        
        preprocessor = DatasetPreprocessor()
        
        # Prepare dataset
        dataset = preprocessor.prepare_dataset(
            audio_dir=mock_audio_dir,
            captions_file=mock_captions_file,
            augment=False
        )
        
        assert len(dataset) == 3
        assert dataset[0]['caption'] == "First audio"
        assert dataset[0]['tags'] == ["test"]
    
    @patch('fugatto_lab.data.dataset.AudioProcessor')
    def test_prepare_dataset_with_output_dir(self, mock_processor_class, mock_audio_dir, mock_captions_file):
        """Test dataset preparation with output directory."""
        mock_processor = Mock()
        mock_processor.sample_rate = 48000
        mock_processor.load_audio.return_value = np.random.randn(48000).astype(np.float32)
        mock_processor.preprocess.return_value = np.random.randn(48000).astype(np.float32)
        mock_processor.save_audio.return_value = None
        mock_processor_class.return_value = mock_processor
        
        preprocessor = DatasetPreprocessor()
        
        with tempfile.TemporaryDirectory() as output_dir:
            dataset = preprocessor.prepare_dataset(
                audio_dir=mock_audio_dir,
                captions_file=mock_captions_file,
                output_dir=output_dir,
                augment=False
            )
            
            assert len(dataset) == 3
            # Audio paths should point to output directory
            assert output_dir in dataset[0]['audio_path']
    
    @patch('fugatto_lab.data.dataset.AudioProcessor')
    def test_validate_dataset(self, mock_processor_class, sample_dataset):
        """Test dataset validation."""
        mock_processor_class.return_value = Mock()
        
        preprocessor = DatasetPreprocessor()
        
        validation = preprocessor.validate_dataset(sample_dataset)
        
        assert isinstance(validation, dict)
        assert 'dataset_size' in validation
        assert 'issues' in validation
        assert 'warnings' in validation
        assert 'is_valid' in validation
        assert 'statistics' in validation
        
        assert validation['dataset_size'] == 3


class TestAudioDataLoader:
    """Test the AudioDataLoader class."""
    
    @pytest.fixture
    def mock_dataset(self):
        """Create mock dataset for data loader testing."""
        dataset = Mock()
        dataset.__len__.return_value = 10
        
        def mock_getitem(idx):
            return {
                'audio': np.random.randn(48000).astype(np.float32),
                'caption': f"Sample {idx}",
                'audio_path': f"/tmp/sample_{idx}.wav",
                'duration_seconds': 1.0,
                'sample_rate': 48000,
                'tags': ["test"],
                'metadata': {},
                'index': idx
            }
        
        dataset.__getitem__.side_effect = mock_getitem
        dataset.processor = Mock()
        dataset.processor.sample_rate = 48000
        
        return dataset
    
    def test_data_loader_initialization(self, mock_dataset):
        """Test data loader initialization."""
        loader = AudioDataLoader(
            dataset=mock_dataset,
            batch_size=4,
            shuffle=True,
            num_workers=0
        )
        
        assert loader.batch_size == 4
        assert loader.shuffle == True
        assert loader.num_workers == 0
        assert len(loader) == 3  # 10 samples / 4 batch_size = 2.5 -> 3 batches
    
    def test_data_loader_iteration(self, mock_dataset):
        """Test data loader iteration."""
        loader = AudioDataLoader(
            dataset=mock_dataset,
            batch_size=3,
            shuffle=False,
            num_workers=0
        )
        
        batches = list(loader)
        
        assert len(batches) == 4  # 10 samples / 3 batch_size = 3.33 -> 4 batches
        
        # Check first batch
        first_batch = batches[0]
        assert len(first_batch['audio']) == 3
        assert len(first_batch['captions']) == 3
        assert first_batch['captions'][0] == "Sample 0"
    
    def test_data_loader_drop_last(self, mock_dataset):
        """Test data loader with drop_last=True."""
        loader = AudioDataLoader(
            dataset=mock_dataset,
            batch_size=3,
            drop_last=True,
            shuffle=False,
            num_workers=0
        )
        
        batches = list(loader)
        
        assert len(batches) == 3  # 10 samples / 3 batch_size = 3 complete batches
        
        # All batches should have exactly 3 samples
        for batch in batches:
            assert len(batch['audio']) == 3
    
    def test_data_loader_statistics(self, mock_dataset):
        """Test data loader statistics."""
        loader = AudioDataLoader(mock_dataset, batch_size=4)
        
        stats = loader.get_statistics()
        
        assert stats['dataset_size'] == 10
        assert stats['batch_size'] == 4
        assert stats['num_batches'] == 3


class TestBatchProcessor:
    """Test the BatchProcessor class."""
    
    @pytest.fixture
    def mock_processor(self):
        """Mock audio processor for batch processor."""
        processor = Mock()
        processor.sample_rate = 48000
        processor.normalize_loudness.return_value = np.random.randn(48000).astype(np.float32)
        processor._trim_silence.return_value = np.random.randn(48000).astype(np.float32)
        processor._apply_basic_filter.return_value = np.random.randn(48000).astype(np.float32)
        processor._resample_audio.return_value = np.random.randn(44100).astype(np.float32)
        return processor
    
    def test_batch_processor_initialization(self, mock_processor):
        """Test batch processor initialization."""
        batch_processor = BatchProcessor(processor=mock_processor, max_workers=2)
        
        assert batch_processor.max_workers == 2
        assert batch_processor.processor == mock_processor
    
    def test_process_batch_sequential(self, mock_processor):
        """Test sequential batch processing."""
        batch_processor = BatchProcessor(processor=mock_processor, max_workers=1)
        
        audio_batch = [
            np.random.randn(48000).astype(np.float32),
            np.random.randn(48000).astype(np.float32),
            np.random.randn(48000).astype(np.float32)
        ]
        
        operations = ['normalize', 'filter']
        
        processed_batch = batch_processor.process_batch(audio_batch, operations)
        
        assert len(processed_batch) == 3
        assert all(isinstance(audio, np.ndarray) for audio in processed_batch)
    
    def test_process_batch_operations(self, mock_processor):
        """Test different processing operations."""
        batch_processor = BatchProcessor(processor=mock_processor, max_workers=1)
        
        audio = np.random.randn(48000).astype(np.float32)
        
        # Test normalize operation
        result = batch_processor.process_batch([audio], ['normalize'])
        assert len(result) == 1
        mock_processor.normalize_loudness.assert_called()
        
        # Test trim_silence operation
        mock_processor.reset_mock()
        result = batch_processor.process_batch([audio], ['trim_silence'])
        mock_processor._trim_silence.assert_called()
        
        # Test filter operation
        mock_processor.reset_mock()
        result = batch_processor.process_batch([audio], ['filter'])
        mock_processor._apply_basic_filter.assert_called()
    
    def test_collate_batch_with_padding(self, mock_processor):
        """Test batch collation with padding."""
        batch_processor = BatchProcessor(processor=mock_processor)
        
        batch_data = {
            'audio': [
                np.random.randn(48000).astype(np.float32),  # 1 second
                np.random.randn(96000).astype(np.float32),  # 2 seconds
                np.random.randn(72000).astype(np.float32)   # 1.5 seconds
            ],
            'captions': ["Audio 1", "Audio 2", "Audio 3"],
            'durations': [1.0, 2.0, 1.5],
            'sample_rates': [48000, 48000, 48000]
        }
        
        collated = batch_processor.collate_batch(batch_data, pad_audio=True)
        
        assert 'audio' in collated
        assert collated['audio'].shape[0] == 3  # batch size
        assert collated['audio'].shape[1] == 96000  # max length
        assert collated['batch_size'] == 3
        assert collated['max_length'] == 96000
    
    def test_collate_batch_without_padding(self, mock_processor):
        """Test batch collation without padding."""
        batch_processor = BatchProcessor(processor=mock_processor)
        
        batch_data = {
            'audio': [np.random.randn(48000), np.random.randn(96000)],
            'captions': ["Audio 1", "Audio 2"],
            'durations': [1.0, 2.0],
            'sample_rates': [48000, 48000]
        }
        
        collated = batch_processor.collate_batch(batch_data, pad_audio=False)
        
        assert isinstance(collated['audio'], list)
        assert len(collated['audio']) == 2
        assert collated['max_length'] is None
    
    def test_create_data_loader(self, mock_processor, mock_dataset):
        """Test data loader creation."""
        batch_processor = BatchProcessor(processor=mock_processor)
        
        loader = batch_processor.create_data_loader(
            dataset=mock_dataset,
            batch_size=4,
            shuffle=True
        )
        
        assert isinstance(loader, AudioDataLoader)
        assert loader.batch_size == 4
        assert loader.shuffle == True
    
    def test_benchmark_loading(self, mock_processor, mock_dataset):
        """Test data loading benchmark."""
        batch_processor = BatchProcessor(processor=mock_processor, max_workers=1)
        
        results = batch_processor.benchmark_loading(
            dataset=mock_dataset,
            num_batches=2,
            batch_size=3
        )
        
        assert 'sequential_time_seconds' in results
        assert 'parallel_time_seconds' in results
        assert 'speedup' in results
        assert 'batches_tested' in results
        assert results['batches_tested'] == 2


class TestDatasetSplitter:
    """Test the DatasetSplitter class."""
    
    @pytest.fixture
    def large_dataset(self):
        """Create a larger dataset for splitting tests."""
        samples = []
        
        for i in range(100):
            sample = AudioSample(
                audio_path=f"/tmp/audio_{i}.wav",
                caption=f"Audio sample {i}",
                duration_seconds=float(i % 10 + 1),  # Durations 1-10
                sample_rate=48000,
                tags=[f"tag_{i % 5}"],  # 5 different tags
                metadata={"index": i}
            )
            samples.append(sample)
        
        processor = Mock()
        processor.sample_rate = 48000
        
        return AudioDataset(samples, processor)
    
    def test_split_dataset_basic(self, large_dataset):
        """Test basic dataset splitting."""
        splits = DatasetSplitter.split_dataset(
            dataset=large_dataset,
            train_ratio=0.7,
            val_ratio=0.2,
            test_ratio=0.1,
            shuffle=True,
            random_seed=42
        )
        
        assert 'train' in splits
        assert 'val' in splits
        assert 'test' in splits
        
        assert len(splits['train']) == 70
        assert len(splits['val']) == 20
        assert len(splits['test']) == 10
        
        # Total should equal original
        total = len(splits['train']) + len(splits['val']) + len(splits['test'])
        assert total == len(large_dataset)
    
    def test_split_dataset_no_shuffle(self, large_dataset):
        """Test dataset splitting without shuffling."""
        splits = DatasetSplitter.split_dataset(
            dataset=large_dataset,
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1,
            shuffle=False
        )
        
        # Without shuffling, first samples should be in train set
        train_sample = splits['train'][0]
        assert "Audio sample 0" in train_sample['caption']
    
    def test_split_dataset_invalid_ratios(self, large_dataset):
        """Test dataset splitting with invalid ratios."""
        with pytest.raises(ValueError):
            DatasetSplitter.split_dataset(
                dataset=large_dataset,
                train_ratio=0.6,
                val_ratio=0.3,
                test_ratio=0.2  # Total = 1.1 > 1.0
            )
    
    def test_stratified_split_by_duration(self, large_dataset):
        """Test stratified splitting by duration."""
        splits = DatasetSplitter.stratified_split(
            dataset=large_dataset,
            stratify_by='duration',
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1
        )
        
        assert 'train' in splits
        assert 'val' in splits
        assert 'test' in splits
        
        # Check that we have samples from different duration ranges
        train_durations = [s['duration_seconds'] for s in splits['train'].samples]
        assert min(train_durations) < 5.0  # Some short samples
        assert max(train_durations) > 5.0  # Some long samples
    
    def test_stratified_split_by_tags(self, large_dataset):
        """Test stratified splitting by tags."""
        splits = DatasetSplitter.stratified_split(
            dataset=large_dataset,
            stratify_by='tags',
            train_ratio=0.7,
            val_ratio=0.2,
            test_ratio=0.1
        )
        
        # Each split should have samples from different tag groups
        train_tags = set()
        for sample in splits['train'].samples:
            train_tags.update(sample.tags)
        
        # Should have multiple different tags
        assert len(train_tags) > 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])