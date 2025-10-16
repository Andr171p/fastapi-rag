import logging

import uvicorn

from fastapi_rag.app import app
from fastapi_rag.settings import settings

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host="0.0.0.0", port=settings.app.port, log_level="info")  # noqa: S104
