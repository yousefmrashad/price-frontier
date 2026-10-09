#!/usr/bin/env python3
"""Fetch OpenRouter model metadata and compute blended $/Mtok pricing.

Blended cost model (defaults):
  - 1,000,000 input tokens per unit of work
  - --cache-hit-rate (0.85) of input tokens priced at input_cache_read
  - remaining input tokens priced at prompt, unless --cache-write-share
    routes a portion of them at input_cache_write
  - output tokens = --output-ratio (0.01) * input tokens, priced at completion

The top --limit models are then expanded per provider via /models/{id}/endpoints
and joined on provider name, so each model/provider row carries both the
model-level and provider-level blended price.
"""

from __future__ import annotations

import json
import re
import threading
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Annotated, Any, NoReturn

import requests
import typer

API_BASE = "https://openrouter.ai/api/v1"
MODELS_URL = f"{API_BASE}/models"
TOKENS = 1_000_000
USER_AGENT = "openrouter-blended-price/1.0"


class SortKey(str, Enum):
    BLENDED = "blended"
    INTELLIGENCE = "intelligence"
    PROMPT = "prompt"
    CONTEXT = "context"


class TpsMode(str, Enum):
    AVG = "avg"
    PERCENTILES = "percentiles"


class ProviderSort(str, Enum):
    PROVIDER = "provider"
    DELTA = "delta"
    MODEL = "model"
    NAME = "name"


def fail(message: str) -> NoReturn:
    """Exit with a clean one-line error.

    typer.BadParameter is for argument validation and renders every failure as
    "Invalid value for <flag>", which is wrong for a bad URL or a 5xx.
    """
    typer.secho(f"error: {message}", fg=typer.colors.RED, err=True)
    raise typer.Exit(1)


class SessionFactory:
    """Hands out one requests.Session per thread.

    Session keeps mutable connection-pool state and is not documented as
    thread-safe, so the provider fan-out cannot share a single instance.
    """

    def __init__(self, api_key: str | None) -> None:
        self._api_key = api_key
        self._local = threading.local()

    def get(self) -> requests.Session:
        session: requests.Session | None = getattr(self._local, "session", None)
        if session is None:
            session = requests.Session()
            session.headers.update(
                {"Accept": "application/json", "User-Agent": USER_AGENT}
            )
            if self._api_key:
                session.headers["Authorization"] = f"Bearer {self._api_key}"
            self._local.session = session
        return session


def _get_json(session: requests.Session, url: str, timeout: float) -> dict[str, Any]:
    resp = session.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def fetch_models(
    session: requests.Session, url: str = MODELS_URL, timeout: float = 30.0
) -> list[dict[str, Any]]:
    try:
        payload = _get_json(session, url, timeout)
    except requests.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else "?"
        fail(f"HTTP {status} from {url}")
    except requests.RequestException as exc:
        fail(f"request to {url} failed: {exc}")
    data = payload.get("data")
    if not isinstance(data, list):
        fail(f"unexpected response from {url}: missing 'data' array")
    return data


def base_model_id(model_id: str) -> str:
    return model_id.split(":", 1)[0]


def fetch_endpoints(
    session: requests.Session,
    model_id: str,
    base: str = API_BASE,
    timeout: float = 30.0,
) -> list[dict[str, Any]]:
    quoted = urllib.parse.quote(base_model_id(model_id), safe="/")
    try:
        payload = _get_json(session, f"{base}/models/{quoted}/endpoints", timeout)
    except (requests.RequestException, ValueError, OSError):
        return []
    data = payload.get("data")
    if not isinstance(data, dict):
        return []
    endpoints = data.get("endpoints")
    return endpoints if isinstance(endpoints, list) else []


def to_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def intelligence_index(model: dict[str, Any]) -> float | None:
    """Find an intelligence score anywhere in the model payload.

    The key sits at a different depth depending on the OpenRouter response, so
    walk the whole structure. Lists must be descended into explicitly: the AA
    scores arrive nested under things like {"architectures": [...]}.
    """
    stack: list[Any] = [model]
    seen: set[int] = set()
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            if id(node) in seen:
                continue
            seen.add(id(node))
            for key, value in node.items():
                lowered = key.lower().replace("-", "_")
                if "intelligence" in lowered and "index" in lowered:
                    score = to_float(value)
                    if score is not None:
                        return score
            stack.extend(node.values())
        elif isinstance(node, list):
            if id(node) in seen:
                continue
            seen.add(id(node))
            stack.extend(node)
    return None


def blended_per_mtok(
    pricing: dict[str, Any] | None,
    cache_hit_rate: float,
    output_ratio: float,
    cache_write_share: float,
) -> tuple[float | None, dict[str, Any]]:
    pricing = pricing or {}
    prompt = to_float(pricing.get("prompt"))
    completion = to_float(pricing.get("completion"))
    if prompt is None or completion is None:
        return None, {
            "prompt": None,
            "completion": None,
            "cache_read": None,
            "cache_write": None,
            "cache_read_priced": False,
        }

    cache_read = to_float(pricing.get("input_cache_read"))
    cache_read_priced = cache_read is not None
    if cache_read is None:
        cache_read = prompt

    cache_write = to_float(pricing.get("input_cache_write"))
    if cache_write is None:
        cache_write = cache_read

    uncached_share = 1.0 - cache_hit_rate
    write_share = min(max(cache_write_share, 0.0), uncached_share)
    write_tokens = uncached_share * write_share
    fresh_tokens = uncached_share - write_tokens

    blended = (
        cache_hit_rate * cache_read
        + write_tokens * cache_write
        + fresh_tokens * prompt
        + output_ratio * completion
    )

    return blended * TOKENS, {
        "prompt": prompt * TOKENS,
        "completion": completion * TOKENS,
        "cache_read": cache_read * TOKENS,
        "cache_write": cache_write * TOKENS,
        "cache_read_priced": cache_read_priced,
    }


PERCENTILE_KEYS = (50, 75, 90, 95, 99)


def throughput_percentiles(endpoint: dict[str, Any]) -> dict[int, float]:
    """Extract {percentile: tokens/sec} from an endpoint's throughput field.

    The field is either null or an object/list keyed by p50..p99; tolerate both
    shapes so a populated value is never dropped by a format difference.
    """
    raw = None
    for key in (
        "throughput_last_30m",
        "throughput_last_1h",
        "throughput_last_6h",
        "throughput_last_24h",
    ):
        if endpoint.get(key):
            raw = endpoint[key]
            break
    if raw is None:
        return {}

    out: dict[int, float] = {}
    if isinstance(raw, dict):
        for pct in PERCENTILE_KEYS:
            # Same value may be spelled "p50", "50", or "50%" depending on window.
            for key in (f"p{pct}", str(pct), f"{pct}%"):
                value = to_float(raw.get(key))
                if value is not None:
                    out[pct] = value
                    break
    elif isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue
            pct = parse_percentile(
                item.get("percentile", item.get("p", item.get("quantile")))
            )
            value = first_float(
                item, "value", "tokens_per_second", "tokens_per_sec", "throughput"
            )
            if pct is not None and value is not None:
                out[pct] = value
    return out


def parse_percentile(label: Any) -> int | None:
    """Normalize a percentile label to its integer.

    The dict branch keys off "p50", so the list branch accepts the same spelling
    rather than only a bare number.
    """
    if label is None:
        return None
    text = str(label).strip().lower().removeprefix("p").removesuffix("%")
    value = to_float(text)
    return int(value) if value is not None else None


def first_float(node: dict[str, Any], *keys: str) -> float | None:
    """First key that parses as a number.

    Uses explicit None checks rather than `or` chaining so a legitimate 0.0
    reading is not skipped in favour of a later key.
    """
    for key in keys:
        value = to_float(node.get(key))
        if value is not None:
            return value
    return None


def money(value: float | None) -> str:
    return "-" if value is None else f"${value:,.3f}"


def plot_points(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rows that can actually be placed: needs both throughput and price.

    Rows without an intelligence index are dropped rather than painted grey,
    since the color channel is the whole point of the chart.
    """
    return [
        r
        for r in rows
        if r["throughput_avg"] is not None
        and r["intelligence_index"] is not None
        and r["provider_blended_per_mtok"] is not None
    ]


def hover_text(row: dict[str, Any]) -> str:
    """Tooltip body. This is how you tell the points apart, so it carries
    every field the table would have shown plus the ones it doesn't."""
    pcts = row["throughput_percentiles"]
    lines = [
        f"<b>{row['id']}</b>",
        f"provider: {row['provider_name']}",
    ]
    if row.get("quantization") and row["quantization"] != "unknown":
        lines.append(f"quantization: {row['quantization']}")
    lines += [
        f"intelligence: {row['intelligence_index']:.1f}",
        f"context: {row['context_length']:,}",
        f"blended: {money(row['provider_blended_per_mtok'])}/Mtok",
        f"model blended: {money(row['model_blended_per_mtok'])}/Mtok",
        f"delta: {money(row['delta_vs_model'])}",
        f"throughput avg: {row['throughput_avg']:.0f} tok/s",
    ]
    if pcts:
        lines.append(
            "p50/75/90/95/99: "
            + "/".join(f"{pcts[p]:.0f}" for p in PERCENTILE_KEYS if p in pcts)
        )
    if row.get("uptime") is not None:
        lines.append(f"uptime 30m: {row['uptime'] * 100:.1f}%")
    if row.get("latency") is not None:
        lines.append(f"latency 30m: {row['latency']:.2f}s")
    return "<br>".join(lines)


def plot_path(raw: str) -> str:
    """Default to .html so `--plot chart` still opens as an interactive page."""
    return raw if Path(raw).suffix else f"{raw}.html"


def write_plot(
    rows: list[dict[str, Any]],
    path: str,
    cache_hit_rate: float,
    output_ratio: float,
    cache_write_share: float,
) -> None:
    path = plot_path(path)
    points = plot_points(rows)
    if not points:
        fail(
            "no provider row has both throughput and an intelligence index, "
            "so there is nothing to plot. Throughput needs an API key "
            "($OPENROUTER_API_KEY); try a wider --include or a higher --limit."
        )

    # Imported lazily so the table and JSON paths do not pay for it.
    import plotly.graph_objects as go

    fig = go.Figure(
        go.Scatter(
            x=[r["throughput_avg"] for r in points],
            y=[r["provider_blended_per_mtok"] for r in points],
            text=[hover_text(r) for r in points],
            hovertemplate="%{text}<extra></extra>",
            mode="markers",
            marker={
                # Price is already the y axis; spending marker size on it again
                # would duplicate the axis. Size by intelligence instead, which
                # is otherwise only carried by color.
                "size": [12 + r["intelligence_index"] / 4 for r in points],
                "sizemode": "diameter",
                "color": [r["intelligence_index"] for r in points],
                "colorscale": "Viridis",
                "colorbar": {"title": {"text": "intelligence<br>index"}},
                "line": {"color": "white", "width": 0.8},
                "opacity": 0.85,
                "showscale": True,
            },
            name="model@provider",
        )
    )

    fig.update_layout(
        title={
            "text": (
                "Cost vs throughput by model@provider"
                f"<br><sup>color and size = intelligence index | "
                f"cache_hit={cache_hit_rate:.0%}, output_ratio={output_ratio:.0%}, "
                f"cache_write_share={cache_write_share:.0%}</sup>"
            )
        },
        xaxis_title="Throughput (tokens/sec, avg of reported percentiles)",
        yaxis_title="Blended $/Mtok (provider, log scale)",
        yaxis_type="log",
        template="plotly_white",
        hovermode="closest",
        # Hover anywhere in the column, not just on the marker itself, so the
        # exact x/y can still be read in a crowded cluster.
        hoverdistance=-1,
        width=1100,
        height=700,
    )
    try:
        fig.write_html(path, include_plotlyjs="cdn")
    except (OSError, ValueError) as exc:
        fail(f"could not write {path}: {exc}")

    typer.secho(
        f"wrote {path} ({len(points)} of {len(rows)} provider rows had both "
        f"throughput and an intelligence index)",
        fg=typer.colors.GREEN,
    )
    typer.echo(
        "open it in a browser and hover a point for the model, provider, "
        "and full price/throughput breakdown"
    )


@dataclass
class Settings:
    api_base: str
    timeout: float
    cache_hit_rate: float
    output_ratio: float
    cache_write_share: float
    provider_workers: int
    sessions: SessionFactory


def build_provider_rows(
    models: list[dict[str, Any]], cfg: Settings
) -> list[dict[str, Any]]:
    def load(row: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        # Called on a pool worker thread, so this resolves that thread's own session.
        endpoints = fetch_endpoints(
            cfg.sessions.get(), row["id"], cfg.api_base, cfg.timeout
        )
        return row, endpoints

    rows: list[dict[str, Any]] = []
    workers = max(1, min(cfg.provider_workers, len(models)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for row, endpoints in pool.map(load, models):
            for ep in endpoints:
                if not isinstance(ep, dict):
                    continue
                pricing = (
                    ep.get("pricing") if isinstance(ep.get("pricing"), dict) else {}
                )
                provider_blended, pprices = blended_per_mtok(
                    pricing, cfg.cache_hit_rate, cfg.output_ratio, cfg.cache_write_share
                )
                if provider_blended is None:
                    continue
                provider_name = str(
                    ep.get("provider_name") or ep.get("name") or "unknown"
                )
                pcts = throughput_percentiles(ep)
                quantization = ep.get("quantization")
                ctx = to_float(ep.get("context_length")) or row["context_length"]
                rows.append(
                    {
                        "id": row["id"],
                        "provider_name": provider_name,
                        "provider_endpoint_name": ep.get("name"),
                        "quantization": quantization,
                        "context_length": int(ctx),
                        "max_completion_tokens": to_float(
                            ep.get("max_completion_tokens")
                        ),
                        "intelligence_index": row["intelligence_index"],
                        "model_blended_per_mtok": row["blended_per_mtok"],
                        "provider_blended_per_mtok": provider_blended,
                        "delta_vs_model": provider_blended - row["blended_per_mtok"],
                        "provider_prompt_per_mtok": pprices["prompt"],
                        "provider_completion_per_mtok": pprices["completion"],
                        "provider_cache_read_per_mtok": pprices["cache_read"],
                        "provider_cache_write_per_mtok": pprices["cache_write"],
                        "provider_cache_read_priced": pprices["cache_read_priced"],
                        "uptime": to_float(ep.get("uptime_last_30m")),
                        "latency": to_float(ep.get("latency_last_30m")),
                        "throughput_percentiles": pcts,
                        "throughput_avg": (sum(pcts.values()) / len(pcts))
                        if pcts
                        else None,
                        "pricing": pricing,
                    }
                )
    return rows


def main(
    min_intelligence: Annotated[
        float | None,
        typer.Option(
            "--min-intelligence",
            help="drop models below this Artificial Analysis intelligence index",
        ),
    ] = None,
    cache_hit_rate: Annotated[
        float,
        typer.Option(
            "--cache-hit-rate",
            help="fraction of input tokens served from cache (default 0.85)",
        ),
    ] = 0.85,
    output_ratio: Annotated[
        float,
        typer.Option(
            "--output-ratio", help="output tokens per input token (default 0.01)"
        ),
    ] = 0.01,
    cache_write_share: Annotated[
        float,
        typer.Option(
            "--cache-write-share",
            help="fraction of non-cached input tokens billed as cache writes (default 0.0)",
        ),
    ] = 0.0,
    max_blended: Annotated[
        float | None,
        typer.Option(
            "--max-blended",
            help="only show models at or below this $/Mtok blended price",
        ),
    ] = None,
    min_context: Annotated[
        int, typer.Option("--min-context", help="minimum context_length")
    ] = 0,
    include: Annotated[
        str | None,
        typer.Option(
            "--include", help="regex filter on model id/name, case-insensitive"
        ),
    ] = None,
    exclude: Annotated[
        str | None,
        typer.Option("--exclude", help="regex to drop, applied after --include"),
    ] = None,
    limit: Annotated[
        int,
        typer.Option("--limit", min=0, help="max models to expand (default 10)"),
    ] = 10,
    sort: Annotated[SortKey, typer.Option("--sort")] = SortKey.BLENDED,
    descending: Annotated[
        bool, typer.Option("--descending", help="reverse the sort order")
    ] = False,
    with_intelligence_only: Annotated[
        bool,
        typer.Option(
            "--with-intelligence-only",
            help="require an intelligence index to be present",
        ),
    ] = False,
    no_providers: Annotated[
        bool,
        typer.Option("--no-providers", help="skip the per-provider endpoints join"),
    ] = False,
    provider_limit: Annotated[
        int,
        typer.Option(
            "--provider-limit", min=0, help="max provider rows per model (0 = no cap)"
        ),
    ] = 0,
    provider_workers: Annotated[
        int,
        typer.Option(
            "--provider-workers",
            min=1,
            help="parallel endpoints requests (default 6)",
        ),
    ] = 6,
    tps: Annotated[TpsMode, typer.Option("--tps")] = TpsMode.AVG,
    provider_sort: Annotated[ProviderSort, typer.Option("--provider-sort")] = (
        ProviderSort.PROVIDER
    ),
    api_key: Annotated[
        str | None,
        typer.Option(
            "--api-key",
            envvar="OPENROUTER_API_KEY",
            help="OpenRouter API key; required for "
            "throughput percentiles (default $OPENROUTER_API_KEY)",
        ),
    ] = None,
    api_base: Annotated[
        str, typer.Option("--api-base", help="override the API base URL")
    ] = API_BASE,
    url: Annotated[
        str, typer.Option("--url", help="override the models endpoint")
    ] = MODELS_URL,
    timeout: Annotated[float, typer.Option("--timeout", min=0.1)] = 30.0,
    as_json: Annotated[
        bool, typer.Option("--json", help="emit JSON instead of a table")
    ] = False,
    plot: Annotated[
        str | None,
        typer.Option(
            "--plot",
            metavar="PATH",
            help="write an interactive blended $/Mtok vs throughput scatter to "
            "PATH (defaults to .html); hover a point for the model and provider. "
            "Throughput needs an API key",
        ),
    ] = None,
) -> None:
    if cache_hit_rate < 0 or cache_hit_rate > 1:
        raise typer.BadParameter(
            "must be between 0 and 1", param_hint="--cache-hit-rate"
        )
    if output_ratio < 0:
        raise typer.BadParameter("must be >= 0", param_hint="--output-ratio")

    include_re = re.compile(include, re.IGNORECASE) if include else None
    exclude_re = re.compile(exclude, re.IGNORECASE) if exclude else None

    sessions = SessionFactory(api_key)
    session = sessions.get()

    rows: list[dict[str, Any]] = []
    for model in fetch_models(session, url, timeout):
        model_id = str(model.get("id", ""))
        name = str(model.get("name", model_id))
        if include_re and not (include_re.search(model_id) or include_re.search(name)):
            continue
        if exclude_re and exclude_re.search(model_id):
            continue

        context = to_float(model.get("context_length")) or 0.0
        if context < min_context:
            continue

        index = intelligence_index(model)
        if min_intelligence is not None:
            if index is None or index < min_intelligence:
                continue
        elif with_intelligence_only and index is None:
            continue

        blended, prices = blended_per_mtok(
            model.get("pricing"), cache_hit_rate, output_ratio, cache_write_share
        )
        if blended is None:
            continue
        if max_blended is not None and blended > max_blended:
            continue

        rows.append(
            {
                "id": model_id,
                "name": name,
                "intelligence_index": index,
                "context_length": int(context),
                "blended_per_mtok": blended,
                "prompt_per_mtok": prices["prompt"],
                "completion_per_mtok": prices["completion"],
                "cache_read_per_mtok": prices["cache_read"],
                "cache_write_per_mtok": prices["cache_write"],
                "cache_read_priced": prices["cache_read_priced"],
                "pricing": model.get("pricing") or {},
            }
        )

    key = {
        SortKey.BLENDED: lambda r: r["blended_per_mtok"],
        SortKey.INTELLIGENCE: lambda r: -(r["intelligence_index"] or 0.0),
        SortKey.PROMPT: lambda r: r["prompt_per_mtok"] or 0.0,
        SortKey.CONTEXT: lambda r: r["context_length"],
    }[sort]
    rows.sort(key=key, reverse=not descending)
    models_shown = rows[:limit]

    provider_rows: list[dict[str, Any]] = []
    if not no_providers and models_shown:
        provider_rows = build_provider_rows(
            models_shown,
            Settings(
                api_base=api_base,
                timeout=timeout,
                cache_hit_rate=cache_hit_rate,
                output_ratio=output_ratio,
                cache_write_share=cache_write_share,
                provider_workers=provider_workers,
                sessions=sessions,
            ),
        )
        pkey = {
            ProviderSort.PROVIDER: lambda r: r["provider_blended_per_mtok"],
            ProviderSort.DELTA: lambda r: r["delta_vs_model"],
            ProviderSort.MODEL: lambda r: r["model_blended_per_mtok"],
            ProviderSort.NAME: lambda r: r["provider_name"],
        }[provider_sort]
        provider_rows.sort(key=pkey)
        if provider_limit > 0:
            kept: list[dict[str, Any]] = []
            seen: dict[str, int] = {}
            for r in provider_rows:
                count = seen.get(r["id"], 0)
                if count >= provider_limit:
                    continue
                seen[r["id"]] = count + 1
                kept.append(r)
            provider_rows = kept

    assumptions = {
        "cache_hit_rate": cache_hit_rate,
        "output_ratio": output_ratio,
        "cache_write_share": cache_write_share,
        "min_intelligence": min_intelligence,
        "limit": limit,
    }

    if plot:
        write_plot(provider_rows, plot, cache_hit_rate, output_ratio, cache_write_share)

    if as_json:
        print(
            json.dumps(
                {
                    "assumptions": assumptions,
                    "count": len(models_shown),
                    "models": models_shown,
                    "provider_count": len(provider_rows),
                    "providers": provider_rows,
                },
                indent=2,
            )
        )
        return

    print(
        f"models matched: {len(rows)}   showing: {len(models_shown)}   "
        f"(cache_hit={cache_hit_rate:.0%}, output_ratio={output_ratio:.0%}, "
        f"cache_write_share={cache_write_share:.0%})"
    )

    header = (
        f"{'ID':<52} {'INTL':>5} {'CTX':>9} {'BLEND':>11} {'PROMPT':>10} "
        f"{'COMPL':>10} {'CACHE_R':>9} {'CR?':>3}"
    )
    print(header)
    print("-" * len(header))
    for row in models_shown:
        intl = (
            ""
            if row["intelligence_index"] is None
            else format(row["intelligence_index"], ".1f")
        )
        print(
            f"{row['id'][:52]:<52} {intl:>5} {row['context_length']:>9,} "
            f"{money(row['blended_per_mtok']):>11} {money(row['prompt_per_mtok']):>10} "
            f"{money(row['completion_per_mtok']):>10} {money(row['cache_read_per_mtok']):>9} "
            f"{'' if row['cache_read_priced'] else '~':>3}"
        )

    if no_providers:
        print("\n(per-provider join skipped: --no-providers)")
    elif not provider_rows:
        print("\nno provider endpoint data returned")
    else:
        print(
            f"\njoined on provider name - {len(provider_rows)} rows across "
            f"{len({r['id'] for r in provider_rows})} models"
        )
        tps_col = "TPS_AVG" if tps is TpsMode.AVG else "TPS 50/75/90/95/99"
        tps_width = 8 if tps is TpsMode.AVG else 17
        pheader = (
            f"{'PROVIDER':<22} {'MODEL':<46} {'QUANT':<10} {tps_col:>{tps_width}} "
            f"{'MODEL_B':>11} {'PROV_B':>11} {'DELTA':>11} {'P_PROMPT':>10} "
            f"{'P_COMPL':>10} {'P_CR':>9} {'CR?':>3}"
        )
        print(pheader)
        print("-" * len(pheader))
        for r in provider_rows:
            quant = str(r["quantization"] or "-")[:10]
            intl = (
                ""
                if r["intelligence_index"] is None
                else format(r["intelligence_index"], ".1f")
            )
            if tps is TpsMode.AVG:
                avg = r["throughput_avg"]
                tps_val = "-" if avg is None else f"{avg:.0f}"
            else:
                pcts = r["throughput_percentiles"]
                tps_val = (
                    "/".join(f"{pcts[p]:.0f}" for p in PERCENTILE_KEYS if p in pcts)
                    if pcts
                    else "-"
                )
            print(
                f"{r['provider_name'][:22]:<22} {(r['id'] + ' ' + intl)[:46]:<46} "
                f"{quant:<10} {tps_val:>{tps_width}} "
                f"{money(r['model_blended_per_mtok']):>11} "
                f"{money(r['provider_blended_per_mtok']):>11} "
                f"{money(r['delta_vs_model']):>11} "
                f"{money(r['provider_prompt_per_mtok']):>10} "
                f"{money(r['provider_completion_per_mtok']):>10} "
                f"{money(r['provider_cache_read_per_mtok']):>9} "
                f"{'' if r['provider_cache_read_priced'] else '~':>3}"
            )

    print(
        "\nBLEND = $/Mtok blended. CR? ~ means no input_cache_read price; prompt price used."
    )
    print(
        "Blended = hit*cache_read + (1-hit)*[write*cache_write + (1-write)*prompt] + out*completion"
    )
    print(
        "DELTA = provider_blended - model_blended (negative = provider is cheaper than the model row)"
    )
    if any(not r["throughput_percentiles"] for r in provider_rows) and not api_key:
        print(
            "TPS = throughput tokens/sec percentiles; all '-' because no --api-key/"
            "$OPENROUTER_API_KEY was set (endpoint stats require auth)."
        )


app = typer.Typer(
    add_completion=False, help="OpenRouter blended $/Mtok price with provider join."
)
app.command()(main)

if __name__ == "__main__":
    app()
