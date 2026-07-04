import asyncio
import cognee

async def remember(text):
    return await cognee.remember(text)

async def recall(text):
    return await cognee.recall(text)

def save_memory(text):
    return asyncio.run(remember(text))

def search_memory(text):
    return asyncio.run(recall(text))
