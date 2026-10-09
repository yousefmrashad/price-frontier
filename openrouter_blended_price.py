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
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Annotated, Any

import requests
import typer

API_BASE = "https://openrouter.ai/api/v1"
MODELS_URL = f"{API_BASE}/models"
TOKENS = 1_000_000
USER_AGENT = "openrouter-blended-price/1.0"


def build_session(api_key: str | None) -> requests.Session:
    session = requests.Session()
    session.headers.update({"Accept": "application/json", "User-Agent": USER_AGENT})
    if api_key:
        session.headers["Authorization"] = f"Bearer {api_key}"
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
        raise typer.BadParameter(f"HTTP {status} from {url}") from exc
    except requests.RequestException as exc:
        raise typer.BadParameter(f"request failed: {exc}") from exc
    data = payload.get("data")
    if not isinstance(data, list):
        raise typer.BadParameter("unexpected response: missing 'data' array")
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
    stack: list[Any] = [model]
    seen: set[int] = set()
    while stack:
        node = stack.pop()
        if not isinstance(node, dict) or id(node) in seen:
            continue
        seen.add(id(node))
        for key, value in node.items():
            lowered = key.lower().replace("-", "_")
            if "intelligence" in lowered and "index" in lowered:
                score = to_float(value)
                if score is not None:
                    return score
        stack.extend(node.values())
    return None


def blended_per_mtok(
    pricing: dict[str, Any] | None,
    cache_hit_rate: float,
    output_ratio: float,
    cache_write_share: float,
) -> tuple[float | None, dict[str, float | None]]:
    pricing = pricing or {}
    prompt = to_float(pricing.get("prompt"))
    completion = to_float(pricing.get("completion"))
    if prompt is None or completion is None:
        return None, {
            "prompt": None,
            "completion": None,
            "cache_read": None,
            "cache_write": None,
            "cache_read_priced": 0.0,
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
        "cache_read_priced": 1.0 if cache_read_priced else 0.0,
    }


PERCENTILES = (("p50", 50), ("p75", 75), ("p90", 90), ("p95", 95), ("p99", 99))


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
        for name, pct in PERCENTILES:
            for key in (name, f"p{pct}", str(pct)):
                if key in raw:
                    value = to_float(raw[key])
                    if value is not None:
                        out[pct] = value
                        break
    elif isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue
            label = item.get("percentile") or item.get("p") or item.get("quantile")
            value = to_float(
                item.get("value")
                or item.get("tokens_per_second")
                or item.get("tokens_per_sec")
                or item.get("throughput")
            )
            pct = to_float(label)
            if pct is not None and value is not None:
                out[int(pct)] = value
    return out


def money(value: float | None) -> str:
    return "-" if value is None else f"${value:,.3f}"


@dataclass
class Settings:
    api_base: str
    timeout: float
    cache_hit_rate: float
    output_ratio: float
    cache_write_share: float
    provider_workers: int
    session: requests.Session


def build_provider_rows(
    models: list[dict[str, Any]], cfg: Settings
) -> list[dict[str, Any]]:
    def load(row: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        endpoints = fetch_endpoints(cfg.session, row["id"], cfg.api_base, cfg.timeout)
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
                        "provider_cache_read_priced": bool(
                            pprices["cache_read_priced"]
                        ),
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
        int, typer.Option("--limit", help="max models to expand (default 10)")
    ] = 10,
    sort: Annotated[
        str, typer.Option("--sort", help="blended | intelligence | prompt | context")
    ] = "blended",
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
            "--provider-limit", help="max provider rows per model (0 = no cap)"
        ),
    ] = 0,
    provider_workers: Annotated[
        int,
        typer.Option(
            "--provider-workers", help="parallel endpoints requests (default 6)"
        ),
    ] = 6,
    tps: Annotated[
        str,
        typer.Option(
            "--tps", help="throughput column: avg | percentiles (default avg)"
        ),
    ] = "avg",
    provider_sort: Annotated[
        str, typer.Option("--provider-sort", help="provider | delta | model | name")
    ] = "provider",
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
    timeout: Annotated[float, typer.Option("--timeout")] = 30.0,
    as_json: Annotated[
        bool, typer.Option("--json", help="emit JSON instead of a table")
    ] = False,
) -> None:
    if cache_hit_rate < 0 or cache_hit_rate > 1:
        raise typer.BadParameter(
            "must be between 0 and 1", param_hint="--cache-hit-rate"
        )
    if output_ratio < 0:
        raise typer.BadParameter("must be >= 0", param_hint="--output-ratio")
    if sort not in ("blended", "intelligence", "prompt", "context"):
        raise typer.BadParameter(
            "must be one of: blended, intelligence, prompt, context",
            param_hint="--sort",
        )
    if tps not in ("avg", "percentiles"):
        raise typer.BadParameter("must be one of: avg, percentiles", param_hint="--tps")
    if provider_sort not in ("provider", "delta", "model", "name"):
        raise typer.BadParameter(
            "must be one of: provider, delta, model, name", param_hint="--provider-sort"
        )

    include_re = re.compile(include, re.IGNORECASE) if include else None
    exclude_re = re.compile(exclude, re.IGNORECASE) if exclude else None

    session = build_session(api_key)

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
                "cache_read_priced": bool(prices["cache_read_priced"]),
                "pricing": model.get("pricing") or {},
            }
        )

    key = {
        "blended": lambda r: r["blended_per_mtok"],
        "intelligence": lambda r: -(r["intelligence_index"] or 0.0),
        "prompt": lambda r: r["prompt_per_mtok"] or 0.0,
        "context": lambda r: r["context_length"],
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
                session=session,
            ),
        )
        pkey = {
            "provider": lambda r: r["provider_blended_per_mtok"],
            "delta": lambda r: r["delta_vs_model"],
            "model": lambda r: r["model_blended_per_mtok"],
            "name": lambda r: r["provider_name"],
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
        tps_col = "TPS_AVG" if tps == "avg" else "TPS 50/75/90/95/99"
        tps_width = 8 if tps == "avg" else 17
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
            if tps == "avg":
                avg = r["throughput_avg"]
                tps_val = "-" if avg is None else f"{avg:.0f}"
            else:
                pcts = r["throughput_percentiles"]
                tps_val = (
                    "/".join(f"{pcts[p]:.0f}" for _, p in PERCENTILES if p in pcts)
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
