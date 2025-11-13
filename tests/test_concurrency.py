import asyncio
from unittest import mock
import sys
import types
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from domain_meta_extractor_consolidated import ConsolidatedExtractor


class DummyProgress:
    def __init__(self, total: int):
        self.total = total
        self.updates = []

    def update(self, success: bool, domain: str = "", method: str = ""):
        self.updates.append((success, domain, method))

    def finish(self):
        pass


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


def test_process_domains_preserves_input_order_and_updates_progress():
    domains = [f"example{i}.com" for i in range(5)]
    delays = {
        domains[0]: 0.05,
        domains[1]: 0.01,
        domains[2]: 0.03,
        domains[3]: 0.02,
        domains[4]: 0.04,
    }

    async def run_test():
        extractor = ConsolidatedExtractor(concurrency=2)
        progress = DummyProgress(len(domains))

        async def fake_extract_domain(self, domain, session, progress_tracker=None):
            await asyncio.sleep(delays[domain])
            if progress_tracker:
                progress_tracker.update(True, domain, "fake")
            return {
                "domain": domain,
                "meta_title": domain,
                "meta_description": domain,
                "extraction_method": "fake",
                "status_code": 200,
                "extraction_time": 0.0,
                "error_message": "",
            }

        extractor.extract_domain = types.MethodType(fake_extract_domain, extractor)
        results = await extractor.process_domains(domains, progress)

        assert [r["domain"] for r in results] == domains
        assert len(progress.updates) == len(domains)

    asyncio.run(run_test())


def test_process_domains_limits_in_flight_task_count():
    domains = [f"example{i}.com" for i in range(100)]
    concurrency = 5

    async def run_test():
        extractor = ConsolidatedExtractor(concurrency=concurrency)
        progress = DummyProgress(len(domains))
        state = {"count": 0, "max": 0}
        original_create_task = asyncio.create_task

        def tracking_create_task(coro, *args, **kwargs):
            task = original_create_task(coro, *args, **kwargs)
            state["count"] += 1
            state["max"] = max(state["max"], state["count"])

            def _done(_):
                state["count"] -= 1

            task.add_done_callback(_done)
            return task

        async def fake_extract_domain(self, domain, session, progress_tracker=None):
            await asyncio.sleep(0.001)
            if progress_tracker:
                progress_tracker.update(True, domain, "fake")
            return {
                "domain": domain,
                "meta_title": domain,
                "meta_description": domain,
                "extraction_method": "fake",
                "status_code": 200,
                "extraction_time": 0.0,
                "error_message": "",
            }

        extractor.extract_domain = types.MethodType(fake_extract_domain, extractor)

        with mock.patch("asyncio.create_task", side_effect=tracking_create_task):
            results = await extractor.process_domains(domains, progress)

        assert len(results) == len(domains)
        assert state["max"] <= concurrency
        assert state["count"] == 0

    asyncio.run(run_test())
