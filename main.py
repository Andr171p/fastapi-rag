import logging

from fastapi import FastAPI

from corp_website_ai.routers import router

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Сервис - AI ассистент для сайта визитки компании")

app.include_router(router)
