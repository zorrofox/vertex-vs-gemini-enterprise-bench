from typing import Any

Citation = tuple[str, str]


def _dedup(items: list[Citation]) -> list[Citation]:
    seen: set[str] = set()
    out: list[Citation] = []
    for title, uri in items:
        if uri and uri not in seen:
            seen.add(uri)
            out.append((title or "Source", uri))
    return out


def extract_grounding_citations(metadata: dict[str, Any] | None) -> list[Citation]:
    """兼容 Vertex AI groundingChunks 与 Discovery Engine supportChunks 两种结构。"""
    if not metadata:
        return []
    found: list[Citation] = []

    for chunk in metadata.get("groundingChunks", []):
        web = chunk.get("web") or {}
        if web.get("uri"):
            found.append((web.get("title", "Web Source"), web["uri"]))

    for chunk in metadata.get("supportChunks", []):
        if chunk.get("uri"):
            found.append((chunk.get("title", "Web Source"), chunk["uri"]))
        web = chunk.get("webSource") or {}
        if web.get("uri"):
            found.append((web.get("title", "Web Source"), web["uri"]))
        src = chunk.get("sourceMetadata") or {}
        if src.get("uri"):
            found.append((src.get("title", "Web Source"), src["uri"]))

    return _dedup(found)


def format_citations_markdown(citations: list[Citation]) -> str:
    if not citations:
        return ""
    lines = ["\n\n---\n**Sources:**"]
    lines += [f"{i}. [{title}]({uri})" for i, (title, uri) in enumerate(citations, 1)]
    return "\n".join(lines) + "\n"
