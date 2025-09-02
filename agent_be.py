from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_core.models import UserMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient
from dotenv import load_dotenv
load_dotenv()
import os
import arxiv
from typing import List, Dict, AsyncGenerator
import asyncio

model_client = OpenAIChatCompletionClient(
        model = "gemini-1.5-flash-8b"
)

# Tool:
def arxiv_search(query: str, max_results: int = 5) -> List[Dict]:
    """
    Return a compact list of arXiv papers matching `query`.

    Each element contains: `title`, `authors`, `published`, `summary` and
    `pdf_url`. The helper is wrapped as an AutoGen FunctionTool below so it
    can be invoked by agents through the normal tool use mechanism.
    """
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )

    papers: List[Dict] = []
    for result in client.results(search):
        papers.append(
            {
                "title": result.title,
                "authors": [a.name for a in result.authors],
                "published": result.published.strftime("%Y-%m-%d"),
                "summary": result.summary,
                "pdf_url": result.pdf_url,
            }
        )
    return papers


# Agent-1:
arxiv_search_agent = AssistantAgent(
        name = "arxiv_search_agent",
        model_client = model_client,
        tools = [arxiv_search],
        description = "Create arXiv queries and retrieves candidates papers.",
        system_message = ("""
                        Given a user topic, think of the best arXiv query. When the tool returns, choose exactly the number of papers requested and pass them as concise JSON to the summarizer.
                        """)
)


# Agent-2:
summarizer_agent = AssistantAgent(
        name = "summarizer_agent",
        model_client = model_client,
        description = "An agent that can summarize the result",
        system_message = ("""
                        You are an expert summarizer. When you recieve the JSON list of papers, write a literature-review style report in Markdown : 
                        1. Start with 2-3 sentence of introduction of the topic.
                        2. Then include one bullet per paper with : title (as Markdown link), authors, the specific problem tackled and its key contribution.
                        3. Close with a single sentence takaway.
                        """
                        ),
)

team = RoundRobinGroupChat(
        participants = [arxiv_search_agent, summarizer_agent],
        max_turns = 2
        )


async def run_team():
    task = "Conduct a literature review on the topic - Autogen and return exactly 5 papers."

    async for msg in team.run_stream(task=task):
        print(msg)


if __name__ == '__main__':
    asyncio.run(run_team())