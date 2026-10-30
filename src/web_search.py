

from ddgs import DDGS


def search_web(query: str, max_results: int = 5) -> list[dict]:
    """Search the web and return titles, snippets, and URLs."""
    if not query or not query.strip():
        return []

    try:
        results = DDGS().text(
            query.strip(),
            max_results=max_results,
        )

        return [
            {
                "title": item.get("title", "Untitled"),
                "snippet": item.get("body", ""),
                "url": item.get("href", ""),
            }
            for item in results
        ]

    except Exception as exc:
        print(f"Web search failed: {exc}")
        return []
