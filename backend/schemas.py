from pydantic import BaseModel
from typing import Optional


class PredictionRequest(BaseModel):

    Component_ID: Optional[str] = None

    Lot_ID: Optional[str] = None

    Component_Type: Optional[str] = None

    Value_0h: float

    Value_24h: float

    Lot_Mean_0h: Optional[float] = None