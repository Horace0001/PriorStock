"""Feature schema for the OHLCV-124 grouped-fusion variant."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FeatureGroupSpec:
    """One named feature group and its projection width."""

    group_name: str
    feature_names: tuple[str, ...]
    projection_dim: int


A_BASELINE_FEATURE_NAMES = (
    "ret_1",
    "ret_3",
    "ret_5",
    "ret_10",
    "ret_20",
    "ret_60",
    "open_gap",
    "close_open_ret",
    "intraday_range",
    "body_signed_rel",
    "upper_shadow_rel",
    "lower_shadow_rel",
    "close_location",
)

B_TREND_FEATURE_NAMES = (
    "sma_gap_5",
    "sma_gap_20",
    "sma_gap_60",
    "ema_gap_5",
    "ema_gap_20",
    "ema_gap_60",
    "sma_spread_5_20",
    "sma_spread_20_60",
    "ema_spread_5_20",
    "ema_spread_20_60",
    "dif_scaled_12_26",
    "dea_scaled_12_26_9",
    "macd_hist_scaled_12_26_9",
    "trix_20",
)

C_MOMENTUM_FEATURE_NAMES = (
    "rsi_scaled_6",
    "rsi_scaled_14",
    "rsi_scaled_30",
    "kdj_k_scaled_9",
    "kdj_d_scaled_9",
    "kdj_j_scaled_9",
    "stochrsi_scaled_14",
    "willr_scaled_14",
    "cci_scaled_20",
    "cmo_scaled_14",
    "roc_5",
    "roc_10",
    "roc_20",
    "ultosc_scaled_7_14_28",
    "dpo_scaled_20",
)

D_VOLATILITY_FEATURE_NAMES = (
    "tr_rel",
    "atr_rel_14",
    "atr_rel_20",
    "ret_vol_5",
    "ret_vol_20",
    "ret_vol_60",
    "bb_upper_gap_20",
    "bb_lower_gap_20",
    "bb_width_20",
    "bb_percent_b_scaled_20",
    "donchian_position_scaled_20",
    "donchian_width_20",
    "keltner_width_20",
    "parkinson_vol_20",
    "garman_klass_vol_20",
)

E_VOLUME_FEATURE_NAMES = (
    "volume_log_ratio_5",
    "volume_log_ratio_20",
    "volume_log_ratio_60",
    "obv_scaled_20",
    "obv_scaled_60",
    "adl_scaled_20",
    "adosc_scaled_3_10",
    "mfi_scaled_14",
    "mfi_scaled_20",
    "pvt_scaled_20",
    "emv_scaled_14",
    "force_scaled_13",
    "force_scaled_20",
    "vwap_gap_5",
    "vwap_gap_20",
)

F_DMI_FEATURE_NAMES = (
    "plus_dm_rel_14",
    "minus_dm_rel_14",
    "plus_di_scaled_14",
    "minus_di_scaled_14",
    "dx_scaled_14",
    "adx_scaled_14",
    "adxr_scaled_14",
    "dmi_spread_scaled_14",
    "dmi_ratio_log_14",
)

G_SUPPORT_RESISTANCE_FEATURE_NAMES = (
    "range_position_scaled_20",
    "range_position_scaled_60",
    "dist_to_high_20",
    "dist_to_high_60",
    "dist_to_low_20",
    "dist_to_low_60",
    "breakout_high_20",
    "breakdown_low_20",
    "breakout_strength_20",
    "breakdown_strength_20",
    "pivot_gap",
    "r1_gap",
    "s1_gap",
    "confirmed_fractal_high",
    "confirmed_fractal_low",
)

H_CANDLE_FEATURE_NAMES = (
    "body_ratio",
    "upper_shadow_ratio",
    "lower_shadow_ratio",
    "open_pos",
    "close_pos",
    "doji_flag",
    "hammer_flag",
    "shooting_star_flag",
    "bullish_engulfing_flag",
    "bearish_engulfing_flag",
)

I_STATISTICAL_FEATURE_NAMES = (
    "mean_ret_5",
    "mean_ret_20",
    "mean_ret_60",
    "ret_zscore_5",
    "ret_zscore_20",
    "skew_ret_20",
    "kurt_ret_20",
    "mdd_20",
    "mdd_60",
    "up_ratio_20",
    "avg_up_20",
    "avg_down_20",
    "autocorr_20_lag1",
    "trend_consistency_20",
)

J_CYCLE_FEATURE_NAMES = (
    "dominant_cycle_scaled_20",
    "dominant_cycle_scaled_60",
    "hilbert_phase_sin",
    "hilbert_phase_cos",
)

FEATURE_GROUP_SPECS = (
    FeatureGroupSpec("A", A_BASELINE_FEATURE_NAMES, 128),
    FeatureGroupSpec("B", B_TREND_FEATURE_NAMES, 16),
    FeatureGroupSpec("C", C_MOMENTUM_FEATURE_NAMES, 16),
    FeatureGroupSpec("D", D_VOLATILITY_FEATURE_NAMES, 16),
    FeatureGroupSpec("E", E_VOLUME_FEATURE_NAMES, 16),
    FeatureGroupSpec("F", F_DMI_FEATURE_NAMES, 12),
    FeatureGroupSpec("G", G_SUPPORT_RESISTANCE_FEATURE_NAMES, 16),
    FeatureGroupSpec("H", H_CANDLE_FEATURE_NAMES, 12),
    FeatureGroupSpec("I", I_STATISTICAL_FEATURE_NAMES, 16),
    FeatureGroupSpec("J", J_CYCLE_FEATURE_NAMES, 8),
)

ALL_FEATURE_NAMES = tuple(
    feature_name
    for group_spec in FEATURE_GROUP_SPECS
    for feature_name in group_spec.feature_names
)

GROUP_NAME_TO_FEATURE_NAMES = {
    group_spec.group_name: group_spec.feature_names
    for group_spec in FEATURE_GROUP_SPECS
}

GROUP_NAME_TO_PROJECTION_DIM = {
    group_spec.group_name: group_spec.projection_dim
    for group_spec in FEATURE_GROUP_SPECS
}

B_TO_J_GROUP_NAMES = tuple(group_spec.group_name for group_spec in FEATURE_GROUP_SPECS if group_spec.group_name != "A")

BASE_INCREMENTAL_PROJECTION_DIM = sum(
    GROUP_NAME_TO_PROJECTION_DIM[group_name]
    for group_name in B_TO_J_GROUP_NAMES
)


def resolve_incremental_group_projection_dims(target_d_model: int) -> dict[str, int]:
    """Scale B-J group projection widths so their concatenation matches `target_d_model`."""

    if target_d_model <= 0:
        raise ValueError("target_d_model must be positive.")
    resolved_projection_dims: dict[str, int] = {}
    accumulated_projection_dim = 0
    for group_name in B_TO_J_GROUP_NAMES[:-1]:
        base_projection_dim = GROUP_NAME_TO_PROJECTION_DIM[group_name]
        scaled_projection_dim = (base_projection_dim * target_d_model) // BASE_INCREMENTAL_PROJECTION_DIM
        if (base_projection_dim * target_d_model) % BASE_INCREMENTAL_PROJECTION_DIM != 0:
            raise ValueError(
                "target_d_model must allow integer-scaled OHLCV-124 B-J projection widths."
            )
        if scaled_projection_dim <= 0:
            raise ValueError("Scaled OHLCV-124 group projection dimensions must be positive.")
        resolved_projection_dims[group_name] = scaled_projection_dim
        accumulated_projection_dim += scaled_projection_dim

    final_group_name = B_TO_J_GROUP_NAMES[-1]
    final_projection_dim = target_d_model - accumulated_projection_dim
    if final_projection_dim <= 0:
        raise ValueError("Final OHLCV-124 group projection dimension must be positive.")
    resolved_projection_dims[final_group_name] = final_projection_dim
    return resolved_projection_dims
