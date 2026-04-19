"""WebLearnCrawler options (robots, throttle)."""

from __future__ import annotations


from engine.WebLearnCrawler import WebLearnCrawler


def test_crawler_stats_include_robots_flags():
    c = WebLearnCrawler(
        {
            "enabled": False,
            "respect_robots_txt": True,
            "min_delay_sec_per_host": 2.5,
        }
    )
    st = c.stats()
    assert st["respect_robots_txt"] is True
    assert st["min_delay_sec_per_host"] == 2.5


def test_robots_disallow_skips_get(monkeypatch):
    c = WebLearnCrawler(
        {
            "enabled": True,
            "seed_urls": ["https://example.com/page"],
            "allowed_hosts": ["example.com"],
            "respect_robots_txt": True,
            "max_urls_per_tick": 1,
        }
    )

    monkeypatch.setattr(c, "_url_allowed_by_robots", lambda u: False)
    monkeypatch.setattr(c, "_throttle_host", lambda h: None)
    monkeypatch.setattr(c, "_touch_host", lambda h: None)

    calls = {"n": 0}

    def no_get(*a, **k):
        calls["n"] += 1
        raise AssertionError("should not GET when robots disallows")

    monkeypatch.setattr("engine.WebLearnCrawler.requests.get", no_get)

    class Sh:
        memory = None

    sh = Sh()

    class Mem:
        def add_node(self, *a, **k):
            raise AssertionError("no memory write")

    sh.memory = Mem()

    out = c.run_tick(sh, 1)
    assert "robots.txt" in (out.get("errors") or [""])[0]
    assert calls["n"] == 0
