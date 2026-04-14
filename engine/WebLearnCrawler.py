"""
Allowlist-only HTTP fetch for BackgroundAgent: pull text from configured URLs,
optionally discover same-host links one hop deep, and store condensed snippets
in MemoryForest (thought_tree / web_learn).

Not a general-purpose crawler: hosts must be listed explicitly in config. Uses
stdlib HTML parsing plus ``requests`` (already a project dependency).

Optional ``respect_robots_txt`` and ``min_delay_sec_per_host`` reduce load and
align with common crawler etiquette (see config / ``stats()``).
"""

from __future__ import annotations

import hashlib
import re
import time
from collections import deque
from html.parser import HTMLParser
from typing import Any, Deque, Dict, List, Optional, Set, TYPE_CHECKING, Union
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import requests

from core.Memory import MemoryNode

if TYPE_CHECKING:
    from agents.shared_resources import SharedResources


class _HTMLTextExtractor(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self._chunks: List[str] = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "noscript"):
            self._skip += 1

    def handle_endtag(self, tag):
        if tag in ("script", "style", "noscript") and self._skip > 0:
            self._skip -= 1

    def handle_data(self, data):
        if self._skip > 0:
            return
        t = data.strip()
        if t:
            self._chunks.append(t)

    def get_text(self) -> str:
        raw = " ".join(self._chunks)
        raw = re.sub(r"\s+", " ", raw).strip()
        return raw


_HREF_RE = re.compile(
    r"""href\s*=\s*(['"])(.*?)\1""",
    re.I | re.DOTALL,
)


def _normalize_url(url: str) -> Optional[str]:
    url = (url or "").strip()
    if not url or url.startswith(("#", "javascript:", "mailto:", "tel:")):
        return None
    try:
        p = urlparse(url)
        if p.scheme not in ("http", "https"):
            return None
        if not p.netloc:
            return None
        # Drop fragment; keep query for Wikipedia etc.
        return p._replace(fragment="").geturl()
    except Exception:
        return None


def _host_of(url: str) -> str:
    try:
        return (urlparse(url).hostname or "").lower()
    except Exception:
        return ""


def _allowed_url(url: str, allowed_hosts: Set[str]) -> bool:
    nu = _normalize_url(url)
    if not nu:
        return False
    host = _host_of(nu)
    return bool(host) and host in allowed_hosts


def html_to_text(html: str, max_chars: int) -> str:
    parser = _HTMLTextExtractor()
    try:
        parser.feed(html)
        parser.close()
    except Exception as _e:
        print(f"[WebLearnCrawler] html parse error: {_e}", flush=True)
    text = parser.get_text()
    if max_chars > 0 and len(text) > max_chars:
        text = text[: max_chars - 3].rstrip() + "..."
    return text


def extract_same_host_links(
    html: str,
    base_url: str,
    allowed_hosts: Set[str],
    limit: int = 12,
) -> List[str]:
    out: List[str] = []
    seen: Set[str] = set()
    try:
        for m in _HREF_RE.finditer(html):
            href = (m.group(2) or "").strip()
            joined = urljoin(base_url, href)
            nu = _normalize_url(joined)
            if not nu or nu in seen:
                continue
            if not _allowed_url(nu, allowed_hosts):
                continue
            seen.add(nu)
            out.append(nu)
            if len(out) >= limit:
                break
    except Exception as _e:
        print(f"[WebLearnCrawler] link extract error: {_e}", flush=True)
    return out


def _content_fingerprint(text: str) -> str:
    sample = re.sub(r"\s+", " ", text)[:4000].lower()
    return hashlib.sha256(sample.encode("utf-8", errors="ignore")).hexdigest()[:24]


class WebLearnCrawler:
    """Background tick helper: fetch allowlisted URLs and write memories."""

    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        cfg = cfg if isinstance(cfg, dict) else {}
        self.enabled: bool = bool(cfg.get("enabled", False))
        self.every_n_ticks: int = max(1, int(cfg.get("every_n_ticks", 12)))
        self.seed_urls: List[str] = [
            str(u).strip() for u in cfg.get("seed_urls", []) if str(u).strip()
        ]
        raw_hosts = cfg.get("allowed_hosts", [])
        self.allowed_hosts: Set[str] = {
            str(h).strip().lower()
            for h in (raw_hosts if isinstance(raw_hosts, list) else [])
            if str(h).strip()
        }
        self.max_chars: int = max(400, min(int(cfg.get("max_chars", 6000)), 100_000))
        self.max_urls_per_tick: int = max(1, min(int(cfg.get("max_urls_per_tick", 1)), 5))
        self.follow_links: bool = bool(cfg.get("follow_links", True))
        self.max_links_per_page: int = max(1, min(int(cfg.get("max_links_per_page", 12)), 40))
        self.max_queue_len: int = max(20, min(int(cfg.get("max_queue_len", 400)), 5000))
        self.timeout_sec: float = max(3.0, min(float(cfg.get("timeout_sec", 18)), 120.0))
        ua = cfg.get("user_agent")
        if isinstance(ua, str) and ua.strip():
            self.user_agent = ua.strip()
        else:
            self.user_agent = "HaromaX6-WebLearn/1.0 (allowlist-only research fetcher)"

        self.respect_robots_txt: bool = bool(cfg.get("respect_robots_txt", False))
        _md = float(cfg.get("min_delay_sec_per_host", 0.0) or 0.0)
        self.min_delay_sec_per_host: float = max(0.0, min(120.0, _md))

        self._queue: Deque[str] = deque()
        self._seen_fingerprints: Set[str] = set()
        self._queued_urls: Set[str] = set()
        self._fetched_urls: Set[str] = set()
        self._seeded = False
        self._last_stats: Dict[str, Any] = {}
        self._last_fetch_monotonic: Dict[str, float] = {}
        self._robots_by_host: Dict[str, Union[RobotFileParser, bool, None]] = {}

    def _throttle_host(self, host: str) -> None:
        if self.min_delay_sec_per_host <= 0 or not host:
            return
        now = time.monotonic()
        last = self._last_fetch_monotonic.get(host, 0.0)
        if last > 0:
            gap = now - last
            if gap < self.min_delay_sec_per_host:
                time.sleep(self.min_delay_sec_per_host - gap)

    def _touch_host(self, host: str) -> None:
        if host:
            self._last_fetch_monotonic[host] = time.monotonic()

    def _robots_for_host(self, host: str) -> Optional[RobotFileParser]:
        if not host:
            return None
        cached = self._robots_by_host.get(host)
        if isinstance(cached, RobotFileParser):
            return cached
        if cached is False:
            return None
        robots_url = f"https://{host}/robots.txt"
        try:
            r = requests.get(
                robots_url,
                timeout=min(12.0, self.timeout_sec),
                headers={"User-Agent": self.user_agent},
            )
            if r.status_code != 200 or not (r.text or "").strip():
                self._robots_by_host[host] = False
                return None
            rp = RobotFileParser()
            rp.set_url(robots_url)
            rp.parse(r.text.splitlines())
            self._robots_by_host[host] = rp
            return rp
        except Exception:
            self._robots_by_host[host] = False
            return None

    def _url_allowed_by_robots(self, url: str) -> bool:
        if not self.respect_robots_txt:
            return True
        host = _host_of(url)
        rp = self._robots_for_host(host)
        if rp is None:
            return True
        try:
            return rp.can_fetch(self.user_agent, url)
        except Exception:
            return True

    def _memory_available(self, shared: SharedResources) -> bool:
        m = getattr(shared, "memory", None)
        return m is not None and hasattr(m, "add_node")

    def _seed_queue(self) -> None:
        if self._seeded:
            return
        self._seeded = True
        ah = self.allowed_hosts
        for u in self.seed_urls:
            if _allowed_url(u, ah):
                nu = _normalize_url(u)
                if nu and nu not in self._queued_urls:
                    self._queued_urls.add(nu)
                    self._queue.append(nu)

    def run_tick(self, shared: SharedResources, cycle_count: int) -> Dict[str, Any]:
        stats: Dict[str, Any] = {
            "ok": False,
            "fetched": 0,
            "skipped_duplicate": 0,
            "enqueued_links": 0,
            "errors": [],
        }
        if not self.enabled:
            stats["reason"] = "disabled"
            self._last_stats = stats
            return stats
        if not self.seed_urls or not self.allowed_hosts:
            stats["reason"] = "missing_seed_urls_or_allowed_hosts"
            self._last_stats = stats
            return stats
        if not self._memory_available(shared):
            stats["reason"] = "no_memory"
            self._last_stats = stats
            return stats

        self._seed_queue()
        if not self._queue:
            stats["reason"] = "queue_empty"
            self._last_stats = stats
            return stats

        headers = {"User-Agent": self.user_agent, "Accept": "text/html,text/plain,*/*"}
        fetched = 0

        while fetched < self.max_urls_per_tick and self._queue:
            url = self._queue.popleft()
            self._queued_urls.discard(url)
            if url in self._fetched_urls:
                continue
            _host = _host_of(url)
            self._throttle_host(_host)
            if not self._url_allowed_by_robots(url):
                stats["errors"].append(f"{url} disallowed by robots.txt")
                self._touch_host(_host)
                continue
            try:
                r = requests.get(
                    url,
                    timeout=self.timeout_sec,
                    headers=headers,
                    allow_redirects=True,
                )
            except requests.RequestException as exc:
                self._touch_host(_host)
                stats["errors"].append(f"{url} {exc!r}")
                continue
            self._touch_host(_host)
            try:
                ctype = (r.headers.get("Content-Type") or "").lower()
                if r.status_code >= 400:
                    stats["errors"].append(f"{url} HTTP {r.status_code}")
                    continue
                if "text/html" not in ctype and "text/plain" not in ctype and ctype:
                    # Skip binary etc.
                    stats["errors"].append(f"{url} skip content-type {ctype[:40]}")
                    continue
                body = r.text or ""
                treat_as_html = "html" in ctype or (
                    not ctype
                    and body[:800].lstrip().lower().startswith(("<!doctype", "<html", "<head"))
                )
                text = (
                    html_to_text(body, self.max_chars)
                    if treat_as_html
                    else ((body[: self.max_chars] + "…") if len(body) > self.max_chars else body)
                )
                text = re.sub(r"\s+", " ", text).strip()
                if len(text) < 80:
                    stats["errors"].append(f"{url} text too short")
                    continue
                fp = _content_fingerprint(text)
                if fp in self._seen_fingerprints:
                    stats["skipped_duplicate"] += 1
                    continue
                self._seen_fingerprints.add(fp)
                title = (
                    _host_of(url)
                    + " — "
                    + (urlparse(url).path.strip("/").replace("/", " ") or "root")
                )
                excerpt_title = title[:120]
                content = f"[web_learn] URL: {url}\nTitle hint: {excerpt_title}\n---\n{text}"
                node = MemoryNode(
                    content=content[: min(len(content), self.max_chars + 400)],
                    emotion="curiosity",
                    confidence=min(0.92, 0.55 + min(0.35, len(text) / 15000.0)),
                    tags=[
                        "web_learn",
                        "external_text",
                        f"url:{url[:180]}",
                        f"cycle:{cycle_count}",
                    ],
                )
                shared.memory.add_node("thought_tree", "web_learn", node)

                # Optional shallow crawl: same-host links only
                if self.follow_links and treat_as_html and len(self._queue) < self.max_queue_len:
                    links = extract_same_host_links(
                        body,
                        url,
                        self.allowed_hosts,
                        limit=self.max_links_per_page,
                    )
                    for link in links:
                        if link not in self._queued_urls and len(self._queue) < self.max_queue_len:
                            self._queued_urls.add(link)
                            self._queue.append(link)
                            stats["enqueued_links"] += 1

                self._fetched_urls.add(url)
                fetched += 1
                stats["ok"] = True
            except Exception as exc:
                stats["errors"].append(f"{url} {exc!r}")

        stats["fetched"] = fetched
        self._last_stats = stats
        return stats

    def stats(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "every_n_ticks": self.every_n_ticks,
            "respect_robots_txt": self.respect_robots_txt,
            "min_delay_sec_per_host": self.min_delay_sec_per_host,
            "queue_len": len(self._queue),
            "fingerprints_cached": len(self._seen_fingerprints),
            "last": dict(self._last_stats),
        }
