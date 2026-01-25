from dataclasses import dataclass
from typing import Optional

@dataclass
class Prediction:
    label: str
    confidence: float
    source: str

@dataclass
class JobText:
    text: str
    position: Optional[str] = None
    organization: Optional[str] = None