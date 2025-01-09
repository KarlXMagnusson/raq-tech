from pydantic import BaseModel
from typing import List

class SlideData(BaseModel):
    number: int
    content: str

class DeckData(BaseModel):
    deck: str
    model: str
    slides: List[SlideData]
