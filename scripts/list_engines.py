"""列出当前 GCP 项目下可用的 Agent Search (Gemini Enterprise) App，用于填写 .env。"""

import sys

import httpx

sys.path.insert(0, ".")
from discovery_api.auth import token_cache  # noqa: E402
from discovery_api.config import settings  # noqa: E402


def main() -> None:
    proj = sys.argv[1] if len(sys.argv) > 1 else settings.gcp_project_id
    token = token_cache.get_token()
    url = (
        f"https://discoveryengine.googleapis.com/v1/projects/{proj}"
        f"/locations/global/collections/default_collection/engines"
    )
    r = httpx.get(
        url,
        headers={"Authorization": f"Bearer {token}", "X-Goog-User-Project": proj},
        timeout=20,
    )
    r.raise_for_status()
    engines = r.json().get("engines", [])

    usable = [
        e
        for e in engines
        if e.get("solutionType") == "SOLUTION_TYPE_SEARCH"
        and (e.get("searchEngineConfig") or {}).get("searchTier") == "SEARCH_TIER_ENTERPRISE"
    ]
    print(f"Project: {proj}")
    print(f"Search Enterprise engines: {len(usable)} / {len(engines)} total\n")
    for e in usable:
        eid = e["name"].split("/")[-1]
        addons = (e.get("searchEngineConfig") or {}).get("searchAddOns", [])
        ds = e.get("dataStoreIds") or []
        print(f"  AGENT_SEARCH_ENGINE={eid}")
        print(f"    displayName={e.get('displayName')}  addons={addons}  dataStores={ds}")


if __name__ == "__main__":
    main()
