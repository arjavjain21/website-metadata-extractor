import asyncio
import sys
import types
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from domain_meta_extractor_consolidated import ConsolidatedExtractor


async def _measure_max_active(concurrency: int, domains):
    extractor = ConsolidatedExtractor(concurrency=concurrency)
    state = {"active": 0, "max_active": 0}

    async def fake_extract_domain(self, domain, session, progress_tracker=None):
        state["active"] += 1
        try:
            state["max_active"] = max(state["max_active"], state["active"])
            await asyncio.sleep(0.01)
            return {
                "domain": domain,
                "meta_title": "",
                "meta_description": "",
                "extraction_method": "fake",
                "status_code": 200,
                "extraction_time": 0.0,
                "error_message": "",
            }
        finally:
            state["active"] -= 1

    extractor.extract_domain = types.MethodType(fake_extract_domain, extractor)
    await extractor.process_domains(domains)
    return state["max_active"]


def test_process_domains_respects_configured_concurrency():
    domains = [f"example{i}.com" for i in range(6)]
    max_active = asyncio.run(_measure_max_active(2, domains))
    assert max_active <= 2


def test_adjusting_concurrency_changes_parallelism():
    domains = [f"example{i}.com" for i in range(6)]

    max_active_low = asyncio.run(_measure_max_active(1, domains))
    max_active_high = asyncio.run(_measure_max_active(3, domains))

    assert max_active_low == 1
    assert max_active_high > max_active_low
    assert max_active_high <= 3
