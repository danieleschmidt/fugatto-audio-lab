"""Fugatto Audio Lab: Toolkit for Controllable Audio Generation.

A plug-and-play generative audio playground with live "prompt → sound" preview
for NVIDIA's Fugatto transformer with text+audio multi-conditioning.
"""

__version__ = "0.1.0"
__author__ = "Daniel Schmidt"
__email__ = "daniel@example.com"

from fugatto_lab.core import FugattoModel, AudioProcessor

__all__ = ["FugattoModel", "AudioProcessor", "__version__"]