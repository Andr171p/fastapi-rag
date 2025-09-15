import asyncio
import logging

from corp_website_ai.documents import store_file

logging.basicConfig(level=logging.INFO)

file_path = r"C:\\Users\\andre\\dio-ai-business-card\\.assets\\Услуги.md"


async def main() -> None:
    await store_file(file_path)


if __name__ == "__main__":
    asyncio.run(main())
