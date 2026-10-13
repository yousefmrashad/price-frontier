"""Tests for the pure functions. No network, no CLI."""

from typing import Any, ClassVar

import pytest

from price_frontier import (
    PERCENTILE_KEYS,
    Retention,
    SessionFactory,
    Settings,
    base_model_id,
    blended_per_mtok,
    build_provider_rows,
    endpoint_label,
    fetch_zdr_keys,
    hover_text,
    intelligence_index,
    parse_percentile,
    plot_path,
    plot_points,
    retention_label,
    retention_mark,
    throughput_percentiles,
    to_float,
    zdr_key,
)


class TestToFloat:
    def test_numeric_strings_and_floats(self):
        assert to_float("1.5") == 1.5
        assert to_float(2) == 2.0

    def test_rejects_none_bool_and_junk(self):
        assert to_float(None) is None
        assert to_float(True) is None
        assert to_float("not a number") is None
        assert to_float([]) is None

    def test_blank_string_is_none(self):
        assert to_float("") is None


class TestBaseModelId:
    def test_strips_variant_suffix(self):
        assert base_model_id("openai/gpt-5.5:free") == "openai/gpt-5.5"

    def test_no_suffix_is_identity(self):
        assert base_model_id("openai/gpt-5.5") == "openai/gpt-5.5"

    def test_only_splits_on_first_colon(self):
        assert base_model_id("a:b:c") == "a"


class TestIntelligenceIndex:
    def test_finds_top_level_score(self):
        assert intelligence_index({"intelligence_index": 42.0}) == 42.0

    def test_normalizes_hyphens_and_case(self):
        assert intelligence_index({"Intelligence-Index": 7}) == 7.0

    def test_finds_nested_score(self):
        model = {"a": {"b": [{"artificial_analysis": {"intelligence_index": 55}}]}}
        assert intelligence_index(model) == 55.0

    def test_absent_returns_none(self):
        assert intelligence_index({"pricing": {"prompt": "1"}}) is None

    def test_non_numeric_returns_none(self):
        assert intelligence_index({"intelligence_index": "n/a"}) is None

    def test_cycle_does_not_hang(self):
        node: dict[str, Any] = {"intelligence_index": 3.0}
        node["self"] = node
        assert intelligence_index(node) == 3.0

    def test_requires_both_words(self):
        assert intelligence_index({"intelligence": 10}) is None
        assert intelligence_index({"index": 10}) is None


class TestBlendedPerMtok:
    pricing: ClassVar[dict[str, str]] = {
        "prompt": "0.000003",
        "completion": "0.000015",
    }

    def test_missing_pricing_is_none(self):
        blended, prices = blended_per_mtok(None, 0.85, 0.01, 0.0)
        assert blended is None
        assert prices["cache_read_priced"] is False

    def test_partial_pricing_is_none(self):
        assert blended_per_mtok({"prompt": "1"}, 0.85, 0.01, 0.0)[0] is None

    def test_falls_back_to_prompt_when_no_cache_price(self):
        blended, prices = blended_per_mtok(self.pricing, 0.85, 0.01, 0.0)
        # every input token at prompt, plus 1% of them at completion
        assert blended == pytest.approx(0.85 * 3 + 0.15 * 3 + 0.01 * 15)
        assert prices["cache_read_priced"] is False

    def test_uses_cache_read_when_priced(self):
        prices_in = {**self.pricing, "input_cache_read": "0.000001"}
        _, prices = blended_per_mtok(prices_in, 0.85, 0.01, 0.0)
        assert prices["cache_read_priced"] is True
        assert prices["cache_read"] == pytest.approx(1.0)

    def test_cache_write_share_split(self):
        prices_in = {
            **self.pricing,
            "input_cache_read": "0.000001",
            "input_cache_write": "0.000004",
        }
        blended, _ = blended_per_mtok(prices_in, 0.5, 0.0, 0.5)
        # 0.5 hit at 1, 0.5 uncached split evenly between write(4) and prompt(3)
        assert blended == pytest.approx(0.5 * 1 + 0.25 * 4 + 0.25 * 3)

    def test_write_share_is_clamped_to_uncached_share(self):
        prices_in = {**self.pricing, "input_cache_write": "0.000009"}
        blended, _ = blended_per_mtok(prices_in, 0.8, 0.0, 5.0)
        # write_share is a fraction of the uncached tokens, so it clamps to 0.2:
        # 0.8 at prompt(3), and the 0.2 uncached split 0.2/0.8 -> 0.04 write(9) + 0.16 prompt
        assert blended == pytest.approx(0.8 * 3 + 0.04 * 9 + 0.16 * 3)

    def test_negative_write_share_clamped_to_zero(self):
        _, prices = blended_per_mtok(self.pricing, 0.5, 0.0, -1.0)
        assert prices["prompt"] == pytest.approx(3.0)

    def test_zero_output_ratio_excludes_completion(self):
        with_out, without = (
            blended_per_mtok(self.pricing, 0.0, 0.01, 0.0)[0],
            blended_per_mtok(self.pricing, 0.0, 0.0, 0.0)[0],
        )
        assert with_out != without


class TestThroughputPercentiles:
    def test_absent_returns_empty(self):
        assert throughput_percentiles({}) == {}

    def test_explicit_null_returns_empty(self):
        assert throughput_percentiles({"throughput_last_30m": None}) == {}

    def test_dict_shape_with_p_keys(self):
        endpoint = {"throughput_last_30m": {"p50": 10, "p99": 100}}
        assert throughput_percentiles(endpoint) == {50: 10, 99: 100}

    def test_dict_shape_with_bare_percentile_keys(self):
        endpoint = {"throughput_last_30m": {"50": 10, "95": 90}}
        assert throughput_percentiles(endpoint) == {50: 10, 95: 90}

    def test_prefers_30m_over_1h(self):
        endpoint = {
            "throughput_last_1h": {"p50": 1},
            "throughput_last_30m": {"p50": 2},
        }
        assert throughput_percentiles(endpoint) == {50: 2}

    def test_falls_back_to_longer_window(self):
        endpoint = {"throughput_last_24h": {"p75": 5}}
        assert throughput_percentiles(endpoint) == {75: 5}

    def test_list_shape(self):
        endpoint = {"throughput_last_1h": [{"percentile": "p50", "value": 12.5}]}
        assert throughput_percentiles(endpoint) == {50: 12.5}

    def test_list_shape_alt_field_names(self):
        endpoint = {"throughput_last_1h": [{"p": 90, "tokens_per_second": 42}]}
        assert throughput_percentiles(endpoint) == {90: 42}

    def test_list_entries_missing_value_dropped(self):
        endpoint = {"throughput_last_1h": [{"percentile": "p50"}]}
        assert throughput_percentiles(endpoint) == {}

    def test_non_dict_entries_ignored(self):
        endpoint = {"throughput_last_1h": ["junk", 7, {"p": 50, "value": 1}]}
        assert throughput_percentiles(endpoint) == {50: 1}

    def test_empty_dict_yields_empty(self):
        assert throughput_percentiles({"throughput_last_30m": {}}) == {}

    def test_unparseable_values_skipped(self):
        endpoint = {"throughput_last_30m": {"p50": "fast"}}
        assert throughput_percentiles(endpoint) == {}

    def test_zero_is_preserved(self):
        # 0 is falsy upstream but a real reading, so it must survive to_float
        endpoint = {"throughput_last_6h": {"p50": 0}}
        assert throughput_percentiles(endpoint) == {50: 0.0}


class TestPlotPoints:
    def row(self, **overrides: Any) -> dict[str, Any]:
        base: dict[str, Any] = {
            "throughput_avg": 100.0,
            "intelligence_index": 40.0,
            "provider_blended_per_mtok": 5.0,
        }
        return {**base, **overrides}

    def test_keeps_complete_rows(self):
        assert len(plot_points([self.row()])) == 1

    def test_drops_rows_without_throughput(self):
        assert plot_points([self.row(throughput_avg=None)]) == []

    def test_drops_rows_without_intelligence(self):
        assert plot_points([self.row(intelligence_index=None)]) == []

    def test_keeps_zero_price_and_zero_tps(self):
        point = self.row(provider_blended_per_mtok=0.0, throughput_avg=0.0)
        assert plot_points([point]) == [point]


def test_percentile_keys_match_display_order():
    assert PERCENTILE_KEYS == (50, 75, 90, 95, 99)


class TestParsePercentile:
    def test_accepts_bare_number(self):
        assert parse_percentile(50) == 50
        assert parse_percentile("50") == 50

    def test_accepts_p_prefixed(self):
        assert parse_percentile("p50") == 50
        assert parse_percentile("P99") == 99

    def test_accepts_percent_sign(self):
        assert parse_percentile("95%") == 95

    def test_tolerates_surrounding_space(self):
        assert parse_percentile(" p75 ") == 75

    def test_rejects_unparseable(self):
        assert parse_percentile("median") is None
        assert parse_percentile(None) is None


class TestHoverText:
    def row(self, **overrides: Any) -> dict[str, Any]:
        base: dict[str, Any] = {
            "id": "openai/gpt-6-astra",
            "provider_name": "OpenAI",
            "quantization": "fp8",
            "intelligence_index": 48.0,
            "context_length": 1050000,
            "provider_blended_per_mtok": 1.5,
            "model_blended_per_mtok": 3.0,
            "delta_vs_model": -1.5,
            "throughput_avg": 230.4,
            "throughput_percentiles": {50: 210.0, 95: 280.0},
            "uptime": 0.997,
            "latency": 0.42,
        }
        return {**base, **overrides}

    def test_always_identifies_model_and_provider(self):
        text = hover_text(self.row())
        assert "openai/gpt-6-astra" in text
        assert "OpenAI" in text

    def test_includes_prices_and_percentiles(self):
        text = hover_text(self.row())
        assert "$1.500/Mtok" in text
        assert "$-1.500" in text
        assert "210/280" in text

    def test_omits_unknown_quantization(self):
        assert "quantization" not in hover_text(self.row(quantization="unknown"))

    def test_omits_absent_uptime_and_latency(self):
        text = hover_text(self.row(uptime=None, latency=None))
        assert "uptime" not in text
        assert "latency" not in text

    def test_omits_percentile_line_when_empty(self):
        text = hover_text(self.row(throughput_percentiles={}))
        assert "p50/75" not in text


class TestPlotPath:
    def test_appends_html_when_no_suffix(self):
        assert plot_path("chart") == "chart.html"

    def test_respects_explicit_suffix(self):
        assert plot_path("chart.png") == "chart.png"

    def test_dotted_name_is_not_treated_as_suffixless(self):
        assert plot_path("plot.v2") == "plot.v2"

    def test_handles_directory_prefix(self):
        assert plot_path("out/chart") == "out/chart.html"


class TestZdrKey:
    def test_strips_variant_suffix(self):
        assert zdr_key("openai/gpt-5-mini:free", "OpenAI", "openai") == (
            "openai/gpt-5-mini",
            "OpenAI",
            "openai",
        )

    def test_missing_tag_normalizes_to_empty(self):
        assert zdr_key("m", "P", None)[2] == ""
        assert zdr_key("m", "P", "")[2] == ""

    def test_missing_provider_normalizes_to_unknown(self):
        assert zdr_key("m", None, "t")[1] == "unknown"


class TestEndpointLabel:
    def test_qualified_tag_keeps_provider_and_qualifier(self):
        assert endpoint_label("OpenAI", "openai/flex") == "OpenAI/flex"
        assert endpoint_label("BaseTen", "baseten/fp4") == "BaseTen/fp4"

    def test_nested_tag_uses_last_segment(self):
        assert endpoint_label("Azure", "azure/swedencentral") == "Azure/swedencentral"

    def test_bare_tag_matching_provider_collapses(self):
        assert endpoint_label("Azure", "azure") == "Azure"
        assert endpoint_label("OpenAI", "openai") == "OpenAI"

    def test_bare_tag_not_matching_provider_is_kept(self):
        assert endpoint_label("Foo", "fast") == "Foo/fast"

    def test_missing_tag_is_provider_only(self):
        assert endpoint_label("Phala", None) == "Phala"
        assert endpoint_label("Phala", "") == "Phala"

    def test_missing_provider_is_unknown(self):
        assert endpoint_label(None, "openai/flex") == "unknown/flex"


class FakeResponse:
    def __init__(self, payload: Any) -> None:
        self._payload: Any = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> Any:
        return self._payload


class FakeSession:
    def __init__(self, payload: Any = None, exc: Exception | None = None) -> None:
        self._payload: Any = payload
        self._exc: Exception | None = exc
        self.urls: list[str] = []

    def get(self, url: str, **_kwargs: Any) -> FakeResponse:
        self.urls.append(url)
        if self._exc is not None:
            raise self._exc
        return FakeResponse(self._payload)


class TestFetchZdrKeys:
    def test_builds_triple_set(self):
        session = FakeSession(
            {
                "data": [
                    {
                        "model_id": "openai/gpt-5-mini",
                        "provider_name": "OpenAI",
                        "tag": "openai",
                    }
                ]
            }
        )
        assert fetch_zdr_keys(session, "https://x.test", 5.0) == {
            ("openai/gpt-5-mini", "OpenAI", "openai")
        }
        assert session.urls == ["https://x.test/endpoints/zdr"]

    def test_skips_non_dict_entries(self):
        session = FakeSession(
            {"data": ["junk", 7, {"model_id": "m", "provider_name": "P"}]}
        )
        assert fetch_zdr_keys(session) == {("m", "P", "")}

    def test_failure_returns_none(self):
        import requests

        session = FakeSession(exc=requests.RequestException("down"))
        assert fetch_zdr_keys(session) is None

    def test_missing_data_returns_none(self):
        assert fetch_zdr_keys(FakeSession({})) is None
        assert fetch_zdr_keys(FakeSession({"data": {}})) is None


class TestRetentionLabels:
    def test_label(self):
        assert retention_label(True) == "zero-retention"
        assert retention_label(False) == "retains"
        assert retention_label(None) == "unknown"

    def test_mark(self):
        assert retention_mark(True) == "yes"
        assert retention_mark(False) == "no"
        assert retention_mark(None) == "?"

    def test_hover_shows_retention(self):
        base: dict[str, Any] = {
            "id": "m",
            "provider_name": "P",
            "intelligence_index": 40.0,
            "context_length": 1000,
            "provider_blended_per_mtok": 1.0,
            "model_blended_per_mtok": 2.0,
            "delta_vs_model": -1.0,
            "throughput_avg": 10.0,
            "throughput_percentiles": {},
        }
        assert "zero-retention" in hover_text({**base, "zdr": True})
        assert "retains" in hover_text({**base, "zdr": False})


class TestBuildProviderRowsZdr:
    def _model(self) -> dict[str, Any]:
        return {
            "id": "openai/gpt-5-mini",
            "intelligence_index": 50.0,
            "context_length": 1000,
            "blended_per_mtok": 1.0,
        }

    def _endpoint(self, **overrides: Any) -> dict[str, Any]:
        base: dict[str, Any] = {
            "provider_name": "OpenAI",
            "tag": "openai",
            "model_id": "openai/gpt-5-mini",
            "pricing": {"prompt": "0.000001", "completion": "0.000002"},
        }
        return {**base, **overrides}

    def _cfg(
        self,
        monkeypatch: pytest.MonkeyPatch,
        endpoint: dict[str, Any],
        zdr_keys: set[tuple[str, str, str]] | None,
    ) -> Settings:
        import price_frontier as pf

        def fake_fetch(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
            return [endpoint]

        monkeypatch.setattr(pf, "fetch_endpoints", fake_fetch)
        return Settings(
            api_base="https://x.test",
            timeout=5.0,
            cache_hit_rate=0.0,
            output_ratio=0.0,
            cache_write_share=0.0,
            provider_workers=1,
            sessions=SessionFactory(None),
            zdr_keys=zdr_keys,
        )

    def test_marks_zdr_true_when_listed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        rows = build_provider_rows(
            [self._model()],
            self._cfg(
                monkeypatch,
                self._endpoint(),
                {("openai/gpt-5-mini", "OpenAI", "openai")},
            ),
        )
        assert rows[0]["zdr"] is True

    def test_marks_zdr_false_when_absent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        rows = build_provider_rows(
            [self._model()], self._cfg(monkeypatch, self._endpoint(), set())
        )
        assert rows[0]["zdr"] is False

    def test_marks_zdr_none_when_list_unreachable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        rows = build_provider_rows(
            [self._model()], self._cfg(monkeypatch, self._endpoint(), None)
        )
        assert rows[0]["zdr"] is None

    def test_endpoint_label_and_tag_stored(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        rows = build_provider_rows(
            [self._model()],
            self._cfg(
                monkeypatch,
                self._endpoint(provider_name="OpenAI", tag="openai/flex"),
                set(),
            ),
        )
        assert rows[0]["endpoint"] == "OpenAI/flex"
        assert rows[0]["provider_tag"] == "openai/flex"

    def test_uptime_percent_scale_normalized_to_fraction(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        rows = build_provider_rows(
            [self._model()],
            self._cfg(monkeypatch, self._endpoint(uptime_last_30m=100), set()),
        )
        assert rows[0]["uptime"] == pytest.approx(1.0)

    def test_uptime_fraction_scale_left_alone(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        rows = build_provider_rows(
            [self._model()],
            self._cfg(monkeypatch, self._endpoint(uptime_last_30m=0.997), set()),
        )
        assert rows[0]["uptime"] == pytest.approx(0.997)

    def test_hover_shows_endpoint_label(self) -> None:
        base: dict[str, Any] = {
            "id": "m",
            "provider_name": "OpenAI",
            "endpoint": "OpenAI/flex",
            "intelligence_index": 40.0,
            "context_length": 1000,
            "provider_blended_per_mtok": 1.0,
            "model_blended_per_mtok": 2.0,
            "delta_vs_model": -1.0,
            "throughput_avg": 10.0,
            "throughput_percentiles": {},
        }
        assert "endpoint: OpenAI/flex" in hover_text({**base, "zdr": None})

    def test_hover_renders_full_uptime_as_100_percent(self) -> None:
        base: dict[str, Any] = {
            "id": "m",
            "provider_name": "P",
            "intelligence_index": 40.0,
            "context_length": 1000,
            "provider_blended_per_mtok": 1.0,
            "model_blended_per_mtok": 2.0,
            "delta_vs_model": -1.0,
            "throughput_avg": 10.0,
            "throughput_percentiles": {},
            "uptime": 1.0,
            "latency": None,
        }
        assert "uptime 30m: 100.0%" in hover_text(base)

    def test_hover_falls_back_to_provider_without_endpoint(self) -> None:
        base: dict[str, Any] = {
            "id": "m",
            "provider_name": "P",
            "intelligence_index": 40.0,
            "context_length": 1000,
            "provider_blended_per_mtok": 1.0,
            "model_blended_per_mtok": 2.0,
            "delta_vs_model": -1.0,
            "throughput_avg": 10.0,
            "throughput_percentiles": {},
        }
        assert "endpoint: P" in hover_text({**base, "zdr": None})

    def test_retention_enum_values(self):
        assert {r.value for r in Retention} == {"all", "zdr", "non-zdr"}
