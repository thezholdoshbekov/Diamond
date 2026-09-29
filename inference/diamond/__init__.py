"""Diamond — sequence-to-sequence speech restoration (v10 RQ-Transformer).

    from diamond import Diamond
    model = Diamond.from_pretrained("data/pretrained/diamond.safetensors")
    model.restore_path("degraded.mp3", "restored.wav")
"""
from .infer import Diamond, SampleConfig, restore_file
from .model.model import SpeechRestorationModel

__all__ = ["Diamond", "SampleConfig", "restore_file", "SpeechRestorationModel"]
__version__ = "1.0.0"
