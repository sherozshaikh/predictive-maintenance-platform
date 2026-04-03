import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger

from apps.api.routes import router, scoring_service
from configs.settings import get_infra_settings
from storage.sqlite_store import init_tables

logger.remove()
logger.add(sys.stderr, serialize=True)

STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(application: FastAPI):
    init_tables()
    scoring_service.load_models()
    logger.info("api_startup_complete")
    yield


app = FastAPI(
    title="Predictive Maintenance Platform",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(router)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def dashboard() -> FileResponse:
    return FileResponse(str(STATIC_DIR / "index.html"))


def run() -> None:
    import uvicorn

    settings = get_infra_settings()
    uvicorn.run(
        "apps.api.main:app",
        host=settings.api.host,
        port=settings.api.port,
        workers=settings.api.workers,
    )


if __name__ == "__main__":
    run()
