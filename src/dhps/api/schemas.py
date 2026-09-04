"""Request/response contracts for the API — Sprint C.

Structural constraints (positivity, the policy's [0, 1] position range)
live here as pydantic fields; the pricer's training box is checked in
``service.validate_pricer_box`` so violations carry the box limits in
the message.
"""

from pydantic import BaseModel, Field


class OptionQuery(BaseModel):
    spot: float = Field(gt=0.0, description="current spot")
    strike: float = Field(gt=0.0)
    t_maturity: float = Field(gt=0.0, description="time to maturity, years")
    sigma: float = Field(gt=0.0, description="annualized volatility")


class PriceOut(BaseModel):
    price: float


class GreeksOut(BaseModel):
    price: float
    delta: float
    gamma: float
    vega: float
    theta: float
    dual_delta: float


class HedgeQuery(BaseModel):
    spot: float = Field(gt=0.0)
    strike: float = Field(gt=0.0)
    time_to_maturity: float = Field(gt=0.0, le=1.0)
    position: float = Field(ge=0.0, le=1.0,
                            description="current stock position")


class HedgeOut(BaseModel):
    target_position: float
