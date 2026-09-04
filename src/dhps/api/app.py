"""FastAPI wiring for the pricing/hedging endpoints — Sprint C.

    uv run --extra api uvicorn dhps.api.app:app --port 8000

The lifespan trains-or-loads both learners BEFORE the server accepts
requests, so a reachable server is a warm server. See scripts/
benchmark_api.py for the latency gate and scripts/export_models.py to
pre-build a budget (the Docker image does that at build time).
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from dhps.api import service
from dhps.api.schemas import GreeksOut, HedgeOut, HedgeQuery, OptionQuery, PriceOut


def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        pricer, policy = service.get_models()
        print(f"[api] budget={service.active_budget()} "
              f"pricer_mae={pricer.metrics['price_mae']:.4f} "
              f"policy_cvar95={policy.metrics['cvar95']:+.2f}")
        yield

    app = FastAPI(
        title="Deep Hedging & Pricing Simulator",
        version="1.0.0",
        description="Learned European pricing (differential ML) and "
                    "deep-hedging positions under transaction costs. "
                    "GET /meta for the models, budgets, and input box.",
        lifespan=lifespan,
    )

    def _checked(q: OptionQuery) -> None:
        try:
            service.validate_pricer_box(q.spot, q.strike, q.t_maturity, q.sigma)
        except ValueError as err:
            raise HTTPException(status_code=422, detail=str(err)) from err

    @app.post("/price", response_model=PriceOut)
    def price(q: OptionQuery) -> PriceOut:
        _checked(q)
        return PriceOut(price=service.price_one(q.spot, q.strike,
                                                q.t_maturity, q.sigma))

    @app.post("/greeks", response_model=GreeksOut)
    def greeks(q: OptionQuery) -> GreeksOut:
        _checked(q)
        return GreeksOut(**service.greeks_one(q.spot, q.strike,
                                              q.t_maturity, q.sigma))

    @app.post("/hedge", response_model=HedgeOut)
    def hedge(q: HedgeQuery) -> HedgeOut:
        return HedgeOut(target_position=service.hedge_one(
            q.spot, q.strike, q.time_to_maturity, q.position))

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/meta")
    def get_meta() -> dict:
        return service.meta()

    return app


app = create_app()
