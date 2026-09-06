"""Tasks 9.1-9.4: detail panel fields, efficiency figures, GPU rows, absent samples."""

from datetime import timedelta

from slurm_monitor.tui.format import format_local
from tests.tui_support import GPU_NO_ACCOUNTING_USAGE, T0, make_app, standard_session


async def _select(app, pilot, table, key):
    table.move_cursor(row=table.get_row_index(key))
    await pilot.pause()
    await app.workers.wait_for_complete()
    await pilot.pause()


async def test_detail_panel_for_active_and_historical_job():
    session = standard_session()
    app = make_app(session)
    async with app.run_test(size=(120, 40)) as pilot:
        await app.ready.wait()
        await app.workers.wait_for_complete()
        await pilot.pause()
        text = app.detail_panel.text
        assert app.selected_job_id == 1001
        for expected in (
            "account     lab-account",
            "qos         normal",
            "nodes       gl1520",
            "requested   4 cpus, 1 nodes, 15.6 GB, 1 gpus, gres/gpu:a100=1",
            "time limit  2:00:00",
            # Rendered in the machine's local zone, so derive the expectation the same way.
            f"submitted   {format_local(T0 - timedelta(hours=2))}",
            "workdir     /home/tester/run1001",
            "stdout      /home/tester/run1001/slurm-1001.out",
            "stderr      /home/tester/run1001/slurm-1001.err",
        ):
            assert expected in text, expected

        await pilot.press("h")
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert app.selected_job_id == 900
        text = app.detail_panel.text
        assert "stdout      /home/tester/run900/slurm-900.out" in text
        assert "stderr      /home/tester/run900/slurm-900.err" in text
        assert "state       COMPLETED" in text
        assert session.calls["job_detail"] == [1001, 900]


async def test_usage_panel_percentages_are_hand_checked():
    app = make_app(standard_session())
    async with app.run_test(size=(120, 40)) as pilot:
        await app.ready.wait()
        await app.workers.wait_for_complete()
        await pilot.pause()
        text = app.usage_panel.text
        # 1800 cpu-s consumed of 4 cpus * 900 s = 3600 -> 50.0%
        assert "cpu         1800 cpu-s / 3600 cpu-s (50.0%)" in text
        # 4000 MB of 16000 MB -> 25.0%
        assert "memory      3.9 GB / 15.6 GB (25.0%)" in text
        # 900 s of 7200 s -> 12.5%
        assert "wall time   0:15:00 / 2:00:00 (12.5%)" in text


async def test_gpu_rows_only_for_gpu_jobs():
    app = make_app(standard_session())
    async with app.run_test(size=(120, 40)) as pilot:
        await app.ready.wait()
        await app.workers.wait_for_complete()
        await pilot.pause()
        # 1001 holds one GPU.
        text = app.usage_panel.text
        assert "gpu util    87.5%" in text
        assert "gpu memory  20.0 GB / 40.0 GB (50.0%)" in text

        # 1002 has no GPU allocation: rows omitted, not shown as 0.
        await _select(app, pilot, app.active_table, "1002")
        text = app.usage_panel.text
        assert app.selected_job_id == 1002
        assert "gpu" not in text
        assert "(50.0%)" in text


async def test_gpu_without_accounting_reads_unavailable():
    session = standard_session()
    session.usages[1001] = GPU_NO_ACCOUNTING_USAGE
    app = make_app(session)
    async with app.run_test(size=(120, 40)) as pilot:
        await app.ready.wait()
        await app.workers.wait_for_complete()
        await pilot.pause()
        text = app.usage_panel.text
        assert "gpu util    unavailable" in text
        assert "gpu memory  unavailable" in text
        assert "0.0%" not in text.split("gpu util")[1]


async def test_sub_interval_job_reports_no_samples_not_zero():
    app = make_app(standard_session())
    async with app.run_test(size=(120, 40)) as pilot:
        await app.ready.wait()
        await pilot.pause()
        await _select(app, pilot, app.active_table, "1003")
        text = app.usage_panel.text
        assert app.selected_job_id == 1003
        flat = " ".join(text.split())
        assert "cpu no samples yet / 80 cpu-s" in flat
        assert "memory no samples yet / 3.9 GB" in flat
        assert "samples no samples yet" in flat
        assert "0%" not in text
        assert "0.0" not in text


async def test_usage_polls_only_while_selected_job_is_running():
    session = standard_session()
    app = make_app(session)
    async with app.run_test(size=(120, 40)) as pilot:
        await app.ready.wait()
        await app.workers.wait_for_complete()
        await pilot.pause()
        # Running job: every usage tick re-fetches.
        assert session.calls["job_usage"].count(1001) == 1
        await app.refresh_usage()
        await app.refresh_usage()
        assert session.calls["job_usage"].count(1001) == 3

        # Terminal job in History: fetched once with the detail, then left alone.
        await pilot.press("h")
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert session.calls["job_usage"].count(900) == 1
        await app.refresh_usage()
        await app.refresh_usage()
        assert session.calls["job_usage"].count(900) == 1


async def test_missing_detail_is_reported_not_fatal():
    session = standard_session()
    del session.details[1001]
    del session.usages[1001]
    app = make_app(session)
    async with app.run_test(size=(120, 40)) as pilot:
        await app.ready.wait()
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert "detail unavailable" in app.detail_panel.text
        assert app.active_table.row_count == 4
