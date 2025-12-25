"""ChatKit server that streams responses from a single assistant."""

from __future__ import annotations

from pathlib import Path
from typing import Any, AsyncIterator

from agents import Runner, function_tool
from chatkit.agents import AgentContext, simple_to_agent_input, stream_agent_response
from chatkit.server import ChatKitServer
from chatkit.types import ThreadMetadata, ThreadStreamEvent, UserMessageItem

from .memory_store import MemoryStore
from .knowledge_base import KnowledgeBase, build_snippet
from agents import Agent


MAX_RECENT_ITEMS = 30
MODEL = "gpt-4.1-mini"
DATA_PATH = Path(__file__).resolve().parent / "data" / "knowledge_base.jsonl"

knowledge_base = KnowledgeBase(db_path=DATA_PATH)


@function_tool
async def search_knowledge_base(query: str, top_k: int = 5) -> list[dict[str, Any]]:
    """
    Search the curated perinatal mental health knowledge base. Use this to ground answers and cite
    relevant sources. Returns the most similar chunks with snippets and URLs.
    """

    results = await knowledge_base.search(query, top_k=top_k)
    payload: list[dict[str, Any]] = []
    for result in results:
        payload.append(
            {
                "url": result.record.url,
                "title": result.record.title or "Unknown title",
                "snippet": build_snippet(result.record.text),
                "score": round(result.score, 4),
                "chunk_index": result.record.chunk_index,
                "source_type": result.record.source_type or "unknown",
            }
        )
    return payload


assistant_agent = Agent[AgentContext[dict[str, Any]]](
    model=MODEL,
    name="Starter Assistant",
    tools=[search_knowledge_base],
    instructions=(
        "You are a concise, helpful assistant grounded in the curated perinatal mental health "
        "knowledge base. Always search the knowledge base before answering. Cite sources with "
        "their URL or title when you use them. If the knowledge base has no relevant information, "
        "say so briefly and avoid speculation."
    ),
)


class StarterChatServer(ChatKitServer[dict[str, Any]]):
    """Server implementation that keeps conversation state in memory."""

    def __init__(self) -> None:
        self.store: MemoryStore = MemoryStore()
        knowledge_base.ensure_loaded()
        super().__init__(self.store)

    async def respond(
        self,
        thread: ThreadMetadata,
        item: UserMessageItem | None,
        context: dict[str, Any],
    ) -> AsyncIterator[ThreadStreamEvent]:
        items_page = await self.store.load_thread_items(
            thread.id,
            after=None,
            limit=MAX_RECENT_ITEMS,
            order="desc",
            context=context,
        )
        items = list(reversed(items_page.data))
        agent_input = await simple_to_agent_input(items)

        agent_context = AgentContext(
            thread=thread,
            store=self.store,
            request_context=context,
        )

        result = Runner.run_streamed(
            assistant_agent,
            agent_input,
            context=agent_context,
        )

        async for event in stream_agent_response(agent_context, result):
            yield event
