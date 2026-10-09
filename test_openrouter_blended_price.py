"""Tests for the pure functions. No network, no CLI."""

from typing import ClassVar

import pytest

from openrouter_blended_price import (
    PERCENTILE_KEYS,
    base_model_id,
    blended_per_mtok,
    intelligence_index,
    plot_points,
    throughput_percentiles,
    to_float,
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
        node: dict = {"intelligence_index": 3.0}
        node["self"] = node
        assert intelligence_index(node) == 3.0

    def test_requires_both_words(self):
        assert intelligence_index({"intelligence": 10}) is None
        assert intelligence_index({"index": 10}) is None


class TestBlendedPerMtok:
    pricing: ClassVar = {"prompt": "0.000003", "completion": "0.000015"}

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
    def row(self, **overrides):
        base = {
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
