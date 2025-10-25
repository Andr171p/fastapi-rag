from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, WebSocket, WebSocketDisconnect

from ..agent import execute_agent
from ..broker import broker
from ..schemas import Message, Role
from ..websockets import connection_manager

router = APIRouter(prefix="/ws", tags=["Websockets"])


@router.websocket("/chat/{chat_id}")
async def chat(
        chat_id: UUID, websocket: WebSocket, background_tasks: BackgroundTasks
) -> None:
    await connection_manager.connect(websocket, chat_id)
    try:
        payload = await websocket.receive_json()
        user_message = Message.model_validate(payload)
        response = await execute_agent(user_message.chat_id, user_message.text)
        ai_message = Message(chat_id=chat_id, role=Role.AI, text=response)
        await connection_manager.send(chat_id, ai_message)
        background_tasks.add_task(
            broker.publish, [user_message, ai_message], queue="messages_persisting"
        )
    except WebSocketDisconnect:
        await connection_manager.disconnect(chat_id)
