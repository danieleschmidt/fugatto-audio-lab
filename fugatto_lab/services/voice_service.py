"""Voice cloning and synthesis service."""

import logging
import numpy as np
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
import time

from ..core import FugattoModel, AudioProcessor


logger = logging.getLogger(__name__)


class VoiceCloneService:
    """Service for voice cloning and speaker synthesis."""
    
    def __init__(self, model_name: str = "nvidia/fugatto-base"):
        """Initialize voice cloning service.
        
        Args:
            model_name: Base model for voice synthesis
        """
        self.model_name = model_name
        self._model = None
        self._processor = None
        self.speaker_embeddings = {}
        
        logger.info("VoiceCloneService initialized")
    
    @property
    def model(self) -> FugattoModel:
        """Get or create model instance."""
        if self._model is None:
            self._model = FugattoModel.from_pretrained(self.model_name)
        return self._model
    
    @property
    def processor(self) -> AudioProcessor:
        """Get or create audio processor."""
        if self._processor is None:
            self._processor = AudioProcessor()
        return self._processor
    
    def extract_speaker_embedding(self, reference_audio: Union[np.ndarray, str, Path],
                                 speaker_id: Optional[str] = None) -> Dict[str, Any]:
        """Extract speaker embedding from reference audio.
        
        Args:
            reference_audio: Reference audio data or file path
            speaker_id: Optional identifier for caching
            
        Returns:
            Speaker embedding information
        """
        start_time = time.time()
        
        # Load audio if path provided
        if isinstance(reference_audio, (str, Path)):
            audio_data = self.processor.load_audio(reference_audio)
            ref_path = str(reference_audio)
        else:
            audio_data = reference_audio
            ref_path = None
        
        logger.info(f"Extracting speaker embedding from {len(audio_data)/self.processor.sample_rate:.2f}s audio")
        
        try:
            # Preprocess reference audio
            processed_audio = self.processor.preprocess(
                audio_data, 
                normalize=True, 
                trim_silence=True
            )
            
            # Extract speaker characteristics (mock implementation)
            embedding = self._extract_voice_features(processed_audio)
            
            result = {
                'speaker_id': speaker_id,
                'embedding': embedding,
                'reference_path': ref_path,
                'reference_duration_seconds': len(processed_audio) / self.processor.sample_rate,
                'extraction_time_ms': (time.time() - start_time) * 1000,
                'audio_stats': self.processor.get_audio_stats(processed_audio)
            }
            
            # Cache embedding if speaker_id provided
            if speaker_id:
                self.speaker_embeddings[speaker_id] = result
                logger.info(f"Cached speaker embedding: {speaker_id}")
            
            logger.info(f"Speaker embedding extracted in {result['extraction_time_ms']:.1f}ms")
            return result
            
        except Exception as e:
            logger.error(f"Speaker embedding extraction failed: {e}")
            raise VoiceCloneError(f"Embedding extraction failed: {e}") from e
    
    def clone_voice(self, reference_audio: Union[np.ndarray, str, Path],
                   text: str, speaker_id: Optional[str] = None,
                   prosody_transfer: bool = True, 
                   emotion_control: Optional[str] = None,
                   output_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """Clone voice from reference and synthesize new speech.
        
        Args:
            reference_audio: Reference audio for voice characteristics
            text: Text to synthesize in cloned voice
            speaker_id: Optional speaker identifier
            prosody_transfer: Whether to transfer prosody patterns
            emotion_control: Optional emotion modification
            output_path: Optional save path
            
        Returns:
            Cloned speech result
        """
        start_time = time.time()
        
        logger.info(f"Cloning voice for text: '{text[:50]}{'...' if len(text) > 50 else ''}'")
        
        try:
            # Extract or get cached speaker embedding
            if speaker_id and speaker_id in self.speaker_embeddings:
                embedding_result = self.speaker_embeddings[speaker_id]
                logger.info(f"Using cached speaker embedding: {speaker_id}")
            else:
                embedding_result = self.extract_speaker_embedding(reference_audio, speaker_id)
            
            # Prepare voice cloning prompt
            voice_prompt = self._create_voice_prompt(text, embedding_result, emotion_control)
            
            # Generate cloned speech
            duration_estimate = self._estimate_speech_duration(text)
            cloned_audio = self.model.generate(
                prompt=voice_prompt,
                duration_seconds=duration_estimate,
                temperature=0.6  # Lower temperature for more consistent voice
            )
            
            # Apply voice transfer if reference audio available
            if prosody_transfer and isinstance(reference_audio, (str, Path, np.ndarray)):
                cloned_audio = self._apply_prosody_transfer(
                    cloned_audio, reference_audio, embedding_result
                )
            
            # Post-process for voice consistency
            cloned_audio = self._post_process_voice(cloned_audio)
            
            result = {
                'cloned_audio': cloned_audio,
                'original_text': text,
                'speaker_id': speaker_id,
                'sample_rate': self.processor.sample_rate,
                'duration_seconds': len(cloned_audio) / self.processor.sample_rate,
                'cloning_time_ms': (time.time() - start_time) * 1000,
                'prosody_transfer': prosody_transfer,
                'emotion_control': emotion_control,
                'audio_stats': self.processor.get_audio_stats(cloned_audio)
            }
            
            # Save to file if requested
            if output_path:
                self.processor.save_audio(cloned_audio, output_path)
                result['output_path'] = str(output_path)
            
            logger.info(f"Voice cloning completed in {result['cloning_time_ms']:.1f}ms")
            return result
            
        except Exception as e:
            logger.error(f"Voice cloning failed: {e}")
            raise VoiceCloneError(f"Voice cloning failed: {e}") from e
    
    def batch_clone_voices(self, reference_audio: Union[np.ndarray, str, Path],
                          texts: List[str], speaker_id: Optional[str] = None,
                          output_dir: Optional[Union[str, Path]] = None) -> List[Dict[str, Any]]:
        """Clone voice for multiple text inputs.
        
        Args:
            reference_audio: Reference audio for voice
            texts: List of texts to synthesize
            speaker_id: Optional speaker identifier
            output_dir: Directory to save outputs
            
        Returns:
            List of cloning results
        """
        logger.info(f"Batch cloning voice for {len(texts)} texts")
        
        # Extract speaker embedding once
        embedding_result = self.extract_speaker_embedding(reference_audio, speaker_id)
        
        results = []
        output_dir = Path(output_dir) if output_dir else None
        
        for i, text in enumerate(texts):
            try:
                # Prepare output path
                output_path = None
                if output_dir:
                    output_dir.mkdir(parents=True, exist_ok=True)
                    safe_filename = "".join(c for c in text[:30] if c.isalnum() or c in (' ', '-', '_')).rstrip()
                    output_path = output_dir / f"{i:03d}_{safe_filename}.wav"
                
                # Clone voice (reuse embedding)
                result = self.clone_voice(
                    reference_audio=embedding_result['embedding'],  # Use cached embedding
                    text=text,
                    speaker_id=speaker_id,
                    output_path=output_path
                )
                
                result['batch_index'] = i
                results.append(result)
                
            except Exception as e:
                logger.error(f"Batch voice cloning failed for text {i}: {e}")
                results.append({
                    'batch_index': i,
                    'text': text,
                    'error': str(e),
                    'success': False
                })
        
        successful = sum(1 for r in results if 'error' not in r)
        logger.info(f"Batch voice cloning completed: {successful}/{len(texts)} successful")
        
        return results
    
    def convert_voice_realtime(self, input_audio: Union[np.ndarray, str, Path],
                              target_speaker_id: str, 
                              chunk_size_seconds: float = 2.0) -> Dict[str, Any]:
        """Convert voice in real-time chunks.
        
        Args:
            input_audio: Input speech audio
            target_speaker_id: Target speaker for conversion
            chunk_size_seconds: Size of processing chunks
            
        Returns:
            Voice conversion result
        """
        if target_speaker_id not in self.speaker_embeddings:
            raise VoiceCloneError(f"Speaker not found: {target_speaker_id}")
        
        # Load input audio
        if isinstance(input_audio, (str, Path)):
            audio_data = self.processor.load_audio(input_audio)
        else:
            audio_data = input_audio
        
        logger.info(f"Converting voice to speaker: {target_speaker_id}")
        
        try:
            chunk_samples = int(chunk_size_seconds * self.processor.sample_rate)
            converted_chunks = []
            
            # Process in chunks
            for i in range(0, len(audio_data), chunk_samples):
                chunk = audio_data[i:i + chunk_samples]
                
                # Convert chunk using speaker embedding
                converted_chunk = self._convert_voice_chunk(
                    chunk, self.speaker_embeddings[target_speaker_id]
                )
                converted_chunks.append(converted_chunk)
            
            # Combine chunks
            converted_audio = np.concatenate(converted_chunks)
            
            result = {
                'converted_audio': converted_audio,
                'target_speaker_id': target_speaker_id,
                'sample_rate': self.processor.sample_rate,
                'duration_seconds': len(converted_audio) / self.processor.sample_rate,
                'chunk_size_seconds': chunk_size_seconds,
                'num_chunks': len(converted_chunks),
                'audio_stats': self.processor.get_audio_stats(converted_audio)
            }
            
            logger.info(f"Voice conversion completed: {result['num_chunks']} chunks")
            return result
            
        except Exception as e:
            logger.error(f"Voice conversion failed: {e}")
            raise VoiceCloneError(f"Voice conversion failed: {e}") from e
    
    def _extract_voice_features(self, audio: np.ndarray) -> Dict[str, Any]:
        """Extract voice characteristics from audio (mock implementation)."""
        # Mock voice feature extraction
        # In a real implementation, this would use advanced voice analysis
        
        # Basic audio characteristics
        rms = np.sqrt(np.mean(audio ** 2))
        spectral_centroid = np.mean(audio)  # Simplified
        pitch_estimate = 440.0 * (1 + spectral_centroid)  # Mock pitch
        
        # Voice timbre features (mock)
        formants = [800, 1200, 2400]  # Mock formant frequencies
        fundamental_freq = pitch_estimate
        
        return {
            'rms': float(rms),
            'pitch_mean': float(pitch_estimate),
            'formants': formants,
            'fundamental_freq': float(fundamental_freq),
            'spectral_centroid': float(spectral_centroid),
            'voice_type': 'unknown',  # Would be classified in real implementation
            'embedding_vector': audio[:128].tolist()  # Mock embedding vector
        }
    
    def _create_voice_prompt(self, text: str, embedding_result: Dict[str, Any],
                           emotion: Optional[str] = None) -> str:
        """Create prompt for voice cloning."""
        base_prompt = f"Synthesize speech: '{text}'"
        
        # Add voice characteristics
        voice_features = embedding_result['embedding']
        pitch = voice_features.get('pitch_mean', 440)
        
        if pitch > 180:
            voice_desc = "with a higher-pitched voice"
        else:
            voice_desc = "with a lower-pitched voice"
        
        prompt = f"{base_prompt} {voice_desc}"
        
        # Add emotion if specified
        if emotion:
            prompt += f" in a {emotion} tone"
        
        return prompt
    
    def _estimate_speech_duration(self, text: str) -> float:
        """Estimate speech duration from text length."""
        # Rough estimate: ~5 characters per second of speech
        words = len(text.split())
        estimated_seconds = max(2.0, words * 0.6)  # ~0.6 seconds per word
        return min(estimated_seconds, 30.0)  # Cap at 30 seconds
    
    def _apply_prosody_transfer(self, generated_audio: np.ndarray, 
                              reference_audio: Union[np.ndarray, str, Path],
                              embedding_result: Dict[str, Any]) -> np.ndarray:
        """Apply prosody patterns from reference to generated audio."""
        # Load reference if needed
        if isinstance(reference_audio, (str, Path)):
            ref_audio = self.processor.load_audio(reference_audio)
        else:
            ref_audio = reference_audio
        
        # Simple prosody transfer (mock implementation)
        # Real implementation would analyze and transfer rhythm, stress patterns, etc.
        
        # Apply reference audio's amplitude envelope
        if len(ref_audio) > 0:
            # Normalize and apply some characteristics
            ref_envelope = np.abs(ref_audio)
            if len(ref_envelope) > len(generated_audio):
                ref_envelope = ref_envelope[:len(generated_audio)]
            else:
                # Repeat pattern if reference is shorter
                repeats = len(generated_audio) // len(ref_envelope) + 1
                ref_envelope = np.tile(ref_envelope, repeats)[:len(generated_audio)]
            
            # Apply envelope with reduced strength
            envelope_strength = 0.3
            modified_audio = generated_audio * (1 - envelope_strength)
            modified_audio += generated_audio * ref_envelope * envelope_strength
            
            return modified_audio
        
        return generated_audio
    
    def _post_process_voice(self, audio: np.ndarray) -> np.ndarray:
        """Post-process cloned voice for consistency."""
        # Apply gentle smoothing
        processed = self.processor.preprocess(
            audio, 
            normalize=True, 
            trim_silence=False,  # Keep natural pauses
            apply_filter=True
        )
        
        return processed
    
    def _convert_voice_chunk(self, chunk: np.ndarray, 
                           target_embedding: Dict[str, Any]) -> np.ndarray:
        """Convert voice characteristics for a single chunk."""
        # Mock voice conversion
        # Real implementation would use advanced voice conversion techniques
        
        target_features = target_embedding['embedding']
        target_pitch = target_features.get('pitch_mean', 440)
        
        # Simple pitch shifting effect
        # This is a very simplified version - real voice conversion is much more complex
        converted = chunk.copy()
        
        # Apply some spectral modifications based on target voice
        if len(converted) > 1:
            # Simple frequency domain modification
            converted = converted * (1 + 0.1 * np.sin(np.linspace(0, 4*np.pi, len(converted))))
        
        return converted
    
    def get_speaker_list(self) -> List[Dict[str, Any]]:
        """Get list of cached speakers."""
        return [
            {
                'speaker_id': speaker_id,
                'reference_duration': info['reference_duration_seconds'],
                'extraction_time': info['extraction_time_ms'],
                'reference_path': info.get('reference_path')
            }
            for speaker_id, info in self.speaker_embeddings.items()
        ]
    
    def remove_speaker(self, speaker_id: str) -> bool:
        """Remove cached speaker embedding."""
        if speaker_id in self.speaker_embeddings:
            del self.speaker_embeddings[speaker_id]
            logger.info(f"Removed speaker: {speaker_id}")
            return True
        return False
    
    def get_service_stats(self) -> Dict[str, Any]:
        """Get service statistics."""
        return {
            'service': 'VoiceCloneService',
            'model_name': self.model_name,
            'cached_speakers': len(self.speaker_embeddings),
            'speaker_list': list(self.speaker_embeddings.keys())
        }


class VoiceCloneError(Exception):
    """Exception raised for voice cloning errors."""
    pass