import logging

from fastapi import FastAPI

from ai_business_card.routers import router

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Сервис - AI ассистент для сайта визитки компании")

app.include_router(router)
