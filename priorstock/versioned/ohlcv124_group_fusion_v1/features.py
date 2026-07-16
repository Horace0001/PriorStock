"""OHLCV-124 feature computation following the lightweight indicator document."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from priorstock.versioned.ohlcv124_group_fusion_v1.feature_schema import ALL_FEATURE_NAMES


warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

_EXTREME_COMPRESSION_FEATURE_NAMES = (
    "obv_scaled_20",
    "obv_scaled_60",
    "adl_scaled_20",
    "adosc_scaled_3_10",
    "pvt_scaled_20",
    "dmi_ratio_log_14",
)


def _safe_divide(numerator: pd.Series, denominator: pd.Series, epsilon: float) -> pd.Series:
    """Divide two aligned series with a configured denominator stabilizer."""

    return numerator / (denominator + epsilon)


def _simple_moving_average(series: pd.Series, window_size: int) -> pd.Series:
    """Compute an expanding-available-history simple moving average."""

    return series.rolling(window=window_size, min_periods=1).mean()


def _exponential_moving_average(series: pd.Series, window_size: int) -> pd.Series:
    """Compute the document-defined EMA with alpha equal to 2 divided by n plus 1."""

    return series.ewm(span=window_size, adjust=False).mean()


def _rolling_standard_deviation(series: pd.Series, window_size: int) -> pd.Series:
    """Compute a population rolling standard deviation with expanding available history."""

    return series.rolling(window=window_size, min_periods=1).std(ddof=0).fillna(0.0)


def _rolling_sum(series: pd.Series, window_size: int) -> pd.Series:
    """Compute an expanding-available-history rolling sum."""

    return series.rolling(window=window_size, min_periods=1).sum()


def _rolling_maximum(series: pd.Series, window_size: int) -> pd.Series:
    """Compute an expanding-available-history rolling maximum."""

    return series.rolling(window=window_size, min_periods=1).max()


def _rolling_minimum(series: pd.Series, window_size: int) -> pd.Series:
    """Compute an expanding-available-history rolling minimum."""

    return series.rolling(window=window_size, min_periods=1).min()


def _compute_return(close_series: pd.Series, lag_size: int) -> pd.Series:
    """Compute C_t divided by lagged close minus one."""

    return (close_series / close_series.shift(lag_size)) - 1.0


def _compute_rsi(close_series: pd.Series, window_size: int, epsilon: float) -> pd.Series:
    """Compute the document EMA-based RSI series."""

    close_change = close_series.diff().fillna(0.0)
    upward_move = close_change.clip(lower=0.0)
    downward_move = (-close_change).clip(lower=0.0)
    relative_strength = _exponential_moving_average(upward_move, window_size) / (
        _exponential_moving_average(downward_move, window_size) + epsilon
    )
    return 100.0 - (100.0 / (1.0 + relative_strength))


def _compute_kdj(
    high_series: pd.Series,
    low_series: pd.Series,
    close_series: pd.Series,
    window_size: int,
    epsilon: float,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Compute recursive K, D, and J stochastic oscillator series."""

    lowest_low = _rolling_minimum(low_series, window_size)
    highest_high = _rolling_maximum(high_series, window_size)
    raw_stochastic_value = 100.0 * (close_series - lowest_low) / (highest_high - lowest_low + epsilon)

    previous_k_value = 50.0
    previous_d_value = 50.0
    k_values: list[float] = []
    d_values: list[float] = []
    for raw_value in raw_stochastic_value.tolist():
        current_k_value = ((2.0 / 3.0) * previous_k_value) + ((1.0 / 3.0) * raw_value)
        current_d_value = ((2.0 / 3.0) * previous_d_value) + ((1.0 / 3.0) * current_k_value)
        k_values.append(current_k_value)
        d_values.append(current_d_value)
        previous_k_value = current_k_value
        previous_d_value = current_d_value

    k_series = pd.Series(k_values, index=close_series.index)
    d_series = pd.Series(d_values, index=close_series.index)
    j_series = (3.0 * k_series) - (2.0 * d_series)
    return k_series, d_series, j_series


def _rolling_mean_absolute_deviation(series: pd.Series, moving_average_series: pd.Series, window_size: int) -> pd.Series:
    """Compute rolling mean absolute deviation from the provided rolling mean."""

    return (series - moving_average_series).abs().rolling(window=window_size, min_periods=1).mean()


def _compute_money_flow_index(
    typical_price_series: pd.Series,
    volume_series: pd.Series,
    window_size: int,
    epsilon: float,
) -> pd.Series:
    """Compute money flow index with rolling positive and negative money flow."""

    raw_money_flow = typical_price_series * volume_series
    typical_price_change = typical_price_series.diff().fillna(0.0)
    positive_money_flow = raw_money_flow.where(typical_price_change > 0.0, 0.0)
    negative_money_flow = raw_money_flow.where(typical_price_change < 0.0, 0.0)
    positive_sum = _rolling_sum(positive_money_flow, window_size)
    negative_sum = _rolling_sum(negative_money_flow, window_size)
    money_flow_ratio = positive_sum / (negative_sum + epsilon)
    return 100.0 - (100.0 / (1.0 + money_flow_ratio))


def _compute_rolling_skew(series: pd.Series, window_size: int, epsilon: float) -> pd.Series:
    """Compute rolling skewness using the document population-moment formula."""

    def _skew(values: np.ndarray) -> float:
        mean_value = float(np.mean(values))
        centered_values = values - mean_value
        standard_deviation = float(np.sqrt(np.mean(centered_values**2)))
        return float(np.mean(centered_values**3) / ((standard_deviation**3) + epsilon))

    return series.rolling(window=window_size, min_periods=1).apply(_skew, raw=True)


def _compute_rolling_kurtosis(series: pd.Series, window_size: int, epsilon: float) -> pd.Series:
    """Compute rolling kurtosis using the document population-moment formula."""

    def _kurtosis(values: np.ndarray) -> float:
        mean_value = float(np.mean(values))
        centered_values = values - mean_value
        standard_deviation = float(np.sqrt(np.mean(centered_values**2)))
        return float(np.mean(centered_values**4) / ((standard_deviation**4) + epsilon))

    return series.rolling(window=window_size, min_periods=1).apply(_kurtosis, raw=True)


def _compute_rolling_maximum_drawdown(close_series: pd.Series, window_size: int, epsilon: float) -> pd.Series:
    """Compute the minimum drawdown within each rolling close-price window."""

    def _maximum_drawdown(values: np.ndarray) -> float:
        running_maximum = np.maximum.accumulate(values)
        drawdown_values = values / (running_maximum + epsilon) - 1.0
        return float(np.min(drawdown_values))

    return close_series.rolling(window=window_size, min_periods=1).apply(_maximum_drawdown, raw=True)


def _compute_rolling_lag_one_autocorrelation(return_series: pd.Series, window_size: int, epsilon: float) -> pd.Series:
    """Compute the document lag-one rolling autocorrelation approximation."""

    def _autocorrelation(values: np.ndarray) -> float:
        if values.size < 2:
            return 0.0
        mean_value = float(np.mean(values))
        centered_values = values - mean_value
        numerator = float(np.sum(centered_values[:-1] * centered_values[1:]))
        denominator = float(np.sum(centered_values**2) + epsilon)
        return numerator / denominator

    return return_series.rolling(window=window_size, min_periods=1).apply(_autocorrelation, raw=True)


def _compute_dominant_cycle_scaled(return_series: pd.Series, window_size: int, epsilon: float) -> pd.Series:
    """Estimate a dominant cycle by the lag with the largest rolling autocorrelation."""

    def _dominant_cycle(values: np.ndarray) -> float:
        if values.size < 3:
            return 0.0
        mean_value = float(np.mean(values))
        centered_values = values - mean_value
        best_lag = 2
        best_score = float("-inf")
        largest_valid_lag = min(window_size - 1, values.size - 1)
        for lag_value in range(2, largest_valid_lag + 1):
            numerator = float(np.sum(centered_values[:-lag_value] * centered_values[lag_value:]))
            denominator = float(np.sum(centered_values**2) + epsilon)
            score = numerator / denominator
            if score > best_score:
                best_score = score
                best_lag = lag_value
        return float(best_lag / window_size)

    return return_series.rolling(window=window_size, min_periods=1).apply(_dominant_cycle, raw=True)


def _compute_hilbert_phase(close_series: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Compute sine and cosine of the analytic-signal phase using an FFT Hilbert transform."""

    close_values = close_series.to_numpy(dtype=np.float64)
    if close_values.size == 0:
        empty_series = pd.Series([], index=close_series.index, dtype=np.float64)
        return empty_series, empty_series

    frequency_values = np.fft.fft(close_values)
    hilbert_multiplier = np.zeros(close_values.size, dtype=np.float64)
    if close_values.size % 2 == 0:
        hilbert_multiplier[0] = 1.0
        hilbert_multiplier[close_values.size // 2] = 1.0
        hilbert_multiplier[1 : close_values.size // 2] = 2.0
    else:
        hilbert_multiplier[0] = 1.0
        hilbert_multiplier[1 : (close_values.size + 1) // 2] = 2.0
    analytic_signal = np.fft.ifft(frequency_values * hilbert_multiplier)
    phase_values = np.angle(analytic_signal)
    return (
        pd.Series(np.sin(phase_values), index=close_series.index),
        pd.Series(np.cos(phase_values), index=close_series.index),
    )


def _sanitize_feature_frame(feature_frame: pd.DataFrame) -> pd.DataFrame:
    """Replace non-finite values and return float32 features in schema order."""

    ordered_feature_frame = feature_frame.loc[:, list(ALL_FEATURE_NAMES)].replace([np.inf, -np.inf], np.nan)
    for feature_name in _EXTREME_COMPRESSION_FEATURE_NAMES:
        ordered_feature_frame[feature_name] = np.tanh(ordered_feature_frame[feature_name])
    ordered_feature_frame = ordered_feature_frame.fillna(0.0)
    return ordered_feature_frame.astype("float32").copy()


def build_ohlcv124_feature_frame(price_frame: pd.DataFrame, epsilon: float) -> pd.DataFrame:
    """Build all 124 document-specified OHLCV features from adjusted OHLCV columns."""

    open_series = price_frame["effective_open"].astype("float64")
    high_series = price_frame["effective_high"].astype("float64")
    low_series = price_frame["effective_low"].astype("float64")
    close_series = price_frame["effective_close"].astype("float64")
    volume_series = price_frame["volume"].astype("float64")

    feature_frame = pd.DataFrame(index=price_frame.index)
    previous_close = close_series.shift(1)
    high_low_range = high_series - low_series
    typical_price = (high_series + low_series + close_series) / 3.0
    ret_1 = _compute_return(close_series, 1).fillna(0.0)

    for lag_size in (1, 3, 5, 10, 20, 60):
        feature_frame[f"ret_{lag_size}"] = _compute_return(close_series, lag_size).fillna(0.0)
    feature_frame["open_gap"] = (open_series / previous_close) - 1.0
    feature_frame["close_open_ret"] = (close_series / open_series) - 1.0
    feature_frame["intraday_range"] = _safe_divide(high_series - low_series, close_series, epsilon)
    feature_frame["body_signed_rel"] = _safe_divide(close_series - open_series, close_series, epsilon)
    feature_frame["upper_shadow_rel"] = _safe_divide(high_series - pd.concat([open_series, close_series], axis=1).max(axis=1), close_series, epsilon)
    feature_frame["lower_shadow_rel"] = _safe_divide(pd.concat([open_series, close_series], axis=1).min(axis=1) - low_series, close_series, epsilon)
    feature_frame["close_location"] = (2.0 * (close_series - low_series) / (high_low_range + epsilon)) - 1.0

    sma_by_window = {window_size: _simple_moving_average(close_series, window_size) for window_size in (5, 20, 60)}
    ema_by_window = {window_size: _exponential_moving_average(close_series, window_size) for window_size in (5, 12, 20, 26, 60)}
    for window_size in (5, 20, 60):
        feature_frame[f"sma_gap_{window_size}"] = (sma_by_window[window_size] / close_series) - 1.0
        feature_frame[f"ema_gap_{window_size}"] = (ema_by_window[window_size] / close_series) - 1.0
    feature_frame["sma_spread_5_20"] = _safe_divide(sma_by_window[5] - sma_by_window[20], close_series, epsilon)
    feature_frame["sma_spread_20_60"] = _safe_divide(sma_by_window[20] - sma_by_window[60], close_series, epsilon)
    feature_frame["ema_spread_5_20"] = _safe_divide(ema_by_window[5] - ema_by_window[20], close_series, epsilon)
    feature_frame["ema_spread_20_60"] = _safe_divide(ema_by_window[20] - ema_by_window[60], close_series, epsilon)
    dif_series = ema_by_window[12] - ema_by_window[26]
    dea_series = _exponential_moving_average(dif_series, 9)
    macd_histogram = dif_series - dea_series
    feature_frame["dif_scaled_12_26"] = _safe_divide(dif_series, close_series, epsilon)
    feature_frame["dea_scaled_12_26_9"] = _safe_divide(dea_series, close_series, epsilon)
    feature_frame["macd_hist_scaled_12_26_9"] = _safe_divide(macd_histogram, close_series, epsilon)
    triple_ema = _exponential_moving_average(_exponential_moving_average(_exponential_moving_average(close_series, 20), 20), 20)
    feature_frame["trix_20"] = (triple_ema / triple_ema.shift(1)) - 1.0

    rsi_by_window = {window_size: _compute_rsi(close_series, window_size, epsilon) for window_size in (6, 14, 30)}
    for window_size, rsi_series in rsi_by_window.items():
        feature_frame[f"rsi_scaled_{window_size}"] = (rsi_series - 50.0) / 50.0
    k_series, d_series, j_series = _compute_kdj(high_series, low_series, close_series, 9, epsilon)
    feature_frame["kdj_k_scaled_9"] = (k_series - 50.0) / 50.0
    feature_frame["kdj_d_scaled_9"] = (d_series - 50.0) / 50.0
    feature_frame["kdj_j_scaled_9"] = np.tanh((j_series - 50.0) / 50.0)
    rsi_14 = rsi_by_window[14]
    stochrsi = (rsi_14 - _rolling_minimum(rsi_14, 14)) / (_rolling_maximum(rsi_14, 14) - _rolling_minimum(rsi_14, 14) + epsilon)
    feature_frame["stochrsi_scaled_14"] = (2.0 * stochrsi) - 1.0
    highest_high_14 = _rolling_maximum(high_series, 14)
    lowest_low_14 = _rolling_minimum(low_series, 14)
    willr = -100.0 * (highest_high_14 - close_series) / (highest_high_14 - lowest_low_14 + epsilon)
    feature_frame["willr_scaled_14"] = (willr + 50.0) / 50.0
    typical_price_sma_20 = _simple_moving_average(typical_price, 20)
    mean_deviation_20 = _rolling_mean_absolute_deviation(typical_price, typical_price_sma_20, 20)
    cci_20 = (typical_price - typical_price_sma_20) / ((0.015 * mean_deviation_20) + epsilon)
    feature_frame["cci_scaled_20"] = np.tanh(cci_20 / 200.0)
    upward_move = close_series.diff().fillna(0.0).clip(lower=0.0)
    downward_move = (-close_series.diff().fillna(0.0)).clip(lower=0.0)
    feature_frame["cmo_scaled_14"] = (_rolling_sum(upward_move, 14) - _rolling_sum(downward_move, 14)) / (
        _rolling_sum(upward_move, 14) + _rolling_sum(downward_move, 14) + epsilon
    )
    for window_size in (5, 10, 20):
        feature_frame[f"roc_{window_size}"] = _compute_return(close_series, window_size).fillna(0.0)
    buying_pressure = close_series - pd.concat([low_series, previous_close], axis=1).min(axis=1)
    true_range_for_ultosc = pd.concat([high_series, previous_close], axis=1).max(axis=1) - pd.concat([low_series, previous_close], axis=1).min(axis=1)
    average_7 = _rolling_sum(buying_pressure, 7) / (_rolling_sum(true_range_for_ultosc, 7) + epsilon)
    average_14 = _rolling_sum(buying_pressure, 14) / (_rolling_sum(true_range_for_ultosc, 14) + epsilon)
    average_28 = _rolling_sum(buying_pressure, 28) / (_rolling_sum(true_range_for_ultosc, 28) + epsilon)
    ultosc = 100.0 * ((4.0 * average_7) + (2.0 * average_14) + average_28) / 7.0
    feature_frame["ultosc_scaled_7_14_28"] = (ultosc - 50.0) / 50.0
    feature_frame["dpo_scaled_20"] = _safe_divide(close_series.shift(11) - _simple_moving_average(close_series, 20), close_series, epsilon)

    true_range_components = pd.concat(
        [
            high_series - low_series,
            (high_series - previous_close).abs(),
            (low_series - previous_close).abs(),
        ],
        axis=1,
    )
    true_range = true_range_components.max(axis=1)
    feature_frame["tr_rel"] = _safe_divide(true_range, close_series, epsilon)
    atr_14 = _exponential_moving_average(true_range, 14)
    atr_20 = _exponential_moving_average(true_range, 20)
    feature_frame["atr_rel_14"] = _safe_divide(atr_14, close_series, epsilon)
    feature_frame["atr_rel_20"] = _safe_divide(atr_20, close_series, epsilon)
    for window_size in (5, 20, 60):
        feature_frame[f"ret_vol_{window_size}"] = _rolling_standard_deviation(ret_1, window_size)
    bb_middle = _simple_moving_average(close_series, 20)
    bb_std = _rolling_standard_deviation(close_series, 20)
    bb_upper = bb_middle + (2.0 * bb_std)
    bb_lower = bb_middle - (2.0 * bb_std)
    feature_frame["bb_upper_gap_20"] = (bb_upper / close_series) - 1.0
    feature_frame["bb_lower_gap_20"] = (bb_lower / close_series) - 1.0
    feature_frame["bb_width_20"] = (bb_upper - bb_lower) / (bb_middle + epsilon)
    feature_frame["bb_percent_b_scaled_20"] = (2.0 * (close_series - bb_lower) / (bb_upper - bb_lower + epsilon)) - 1.0
    donchian_upper_20 = _rolling_maximum(high_series, 20)
    donchian_lower_20 = _rolling_minimum(low_series, 20)
    feature_frame["donchian_position_scaled_20"] = (2.0 * (close_series - donchian_lower_20) / (donchian_upper_20 - donchian_lower_20 + epsilon)) - 1.0
    feature_frame["donchian_width_20"] = (donchian_upper_20 - donchian_lower_20) / (close_series + epsilon)
    feature_frame["keltner_width_20"] = (4.0 * atr_20) / (_exponential_moving_average(close_series, 20) + epsilon)
    feature_frame["parkinson_vol_20"] = np.sqrt(
        _rolling_sum(np.log(high_series / (low_series + epsilon)) ** 2.0, 20) / (4.0 * 20.0 * np.log(2.0))
    )
    garman_klass_component = (
        0.5 * (np.log(high_series / (low_series + epsilon)) ** 2.0)
        - ((2.0 * np.log(2.0)) - 1.0) * (np.log(close_series / (open_series + epsilon)) ** 2.0)
    )
    feature_frame["garman_klass_vol_20"] = np.sqrt(_simple_moving_average(garman_klass_component.clip(lower=0.0), 20))

    for window_size in (5, 20, 60):
        feature_frame[f"volume_log_ratio_{window_size}"] = np.log((volume_series / (_simple_moving_average(volume_series, window_size) + epsilon)) + epsilon)
    close_direction = np.sign(close_series.diff().fillna(0.0))
    obv = (close_direction * volume_series).cumsum()
    feature_frame["obv_scaled_20"] = (obv - _simple_moving_average(obv, 20)) / (_simple_moving_average(volume_series, 20) + epsilon)
    feature_frame["obv_scaled_60"] = (obv - _simple_moving_average(obv, 60)) / (_simple_moving_average(volume_series, 60) + epsilon)
    money_flow_multiplier = ((2.0 * close_series) - high_series - low_series) / (high_low_range + epsilon)
    adl = (money_flow_multiplier * volume_series).cumsum()
    feature_frame["adl_scaled_20"] = (adl - _simple_moving_average(adl, 20)) / (_simple_moving_average(volume_series, 20) + epsilon)
    feature_frame["adosc_scaled_3_10"] = (_exponential_moving_average(adl, 3) - _exponential_moving_average(adl, 10)) / (_simple_moving_average(volume_series, 10) + epsilon)
    feature_frame["mfi_scaled_14"] = (_compute_money_flow_index(typical_price, volume_series, 14, epsilon) - 50.0) / 50.0
    feature_frame["mfi_scaled_20"] = (_compute_money_flow_index(typical_price, volume_series, 20, epsilon) - 50.0) / 50.0
    pvt = (volume_series * ((close_series - previous_close) / (previous_close + epsilon))).fillna(0.0).cumsum()
    feature_frame["pvt_scaled_20"] = (pvt - _simple_moving_average(pvt, 20)) / (_simple_moving_average(volume_series, 20) + epsilon)
    middle_price = (high_series + low_series) / 2.0
    emv = (middle_price - middle_price.shift(1)) / ((volume_series / (high_low_range + epsilon)) + epsilon)
    feature_frame["emv_scaled_14"] = np.tanh(_simple_moving_average(emv, 14) / (close_series + epsilon))
    force = (close_series - previous_close) * volume_series
    feature_frame["force_scaled_13"] = _exponential_moving_average(force.fillna(0.0), 13) / ((close_series * _simple_moving_average(volume_series, 13)) + epsilon)
    feature_frame["force_scaled_20"] = _exponential_moving_average(force.fillna(0.0), 20) / ((close_series * _simple_moving_average(volume_series, 20)) + epsilon)
    for window_size in (5, 20):
        vwap = _rolling_sum(typical_price * volume_series, window_size) / (_rolling_sum(volume_series, window_size) + epsilon)
        feature_frame[f"vwap_gap_{window_size}"] = (vwap / close_series) - 1.0

    up_move = high_series - high_series.shift(1)
    down_move = low_series.shift(1) - low_series
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0.0), up_move, 0.0), index=price_frame.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0.0), down_move, 0.0), index=price_frame.index)
    plus_dm_ema_14 = _exponential_moving_average(plus_dm, 14)
    minus_dm_ema_14 = _exponential_moving_average(minus_dm, 14)
    plus_di = 100.0 * plus_dm_ema_14 / (atr_14 + epsilon)
    minus_di = 100.0 * minus_dm_ema_14 / (atr_14 + epsilon)
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di + epsilon)
    adx = _exponential_moving_average(dx, 14)
    feature_frame["plus_dm_rel_14"] = plus_dm_ema_14 / (close_series + epsilon)
    feature_frame["minus_dm_rel_14"] = minus_dm_ema_14 / (close_series + epsilon)
    feature_frame["plus_di_scaled_14"] = (plus_di - 50.0) / 50.0
    feature_frame["minus_di_scaled_14"] = (minus_di - 50.0) / 50.0
    feature_frame["dx_scaled_14"] = dx / 100.0
    feature_frame["adx_scaled_14"] = adx / 100.0
    feature_frame["adxr_scaled_14"] = ((adx + adx.shift(14)) / 2.0) / 100.0
    feature_frame["dmi_spread_scaled_14"] = (plus_di - minus_di) / 100.0
    feature_frame["dmi_ratio_log_14"] = np.log((plus_di + epsilon) / (minus_di + epsilon))

    for window_size in (20, 60):
        rolling_high = _rolling_maximum(high_series, window_size)
        rolling_low = _rolling_minimum(low_series, window_size)
        feature_frame[f"range_position_scaled_{window_size}"] = (2.0 * (close_series - rolling_low) / (rolling_high - rolling_low + epsilon)) - 1.0
        feature_frame[f"dist_to_high_{window_size}"] = close_series / (rolling_high + epsilon) - 1.0
        feature_frame[f"dist_to_low_{window_size}"] = close_series / (rolling_low + epsilon) - 1.0
    previous_high_20 = high_series.shift(1).rolling(window=20, min_periods=1).max()
    previous_low_20 = low_series.shift(1).rolling(window=20, min_periods=1).min()
    feature_frame["breakout_high_20"] = (close_series > previous_high_20).astype("float64")
    feature_frame["breakdown_low_20"] = (close_series < previous_low_20).astype("float64")
    feature_frame["breakout_strength_20"] = (close_series - previous_high_20) / (close_series + epsilon)
    feature_frame["breakdown_strength_20"] = (close_series - previous_low_20) / (close_series + epsilon)
    pivot = (high_series.shift(1) + low_series.shift(1) + close_series.shift(1)) / 3.0
    resistance_one = (2.0 * pivot) - low_series.shift(1)
    support_one = (2.0 * pivot) - high_series.shift(1)
    feature_frame["pivot_gap"] = pivot / close_series - 1.0
    feature_frame["r1_gap"] = resistance_one / close_series - 1.0
    feature_frame["s1_gap"] = support_one / close_series - 1.0
    feature_frame["confirmed_fractal_high"] = (
        (high_series.shift(2) > high_series.shift(4))
        & (high_series.shift(2) > high_series.shift(3))
        & (high_series.shift(2) > high_series.shift(1))
        & (high_series.shift(2) > high_series)
    ).astype("float64")
    feature_frame["confirmed_fractal_low"] = (
        (low_series.shift(2) < low_series.shift(4))
        & (low_series.shift(2) < low_series.shift(3))
        & (low_series.shift(2) < low_series.shift(1))
        & (low_series.shift(2) < low_series)
    ).astype("float64")

    body_ratio = (close_series - open_series).abs() / (high_low_range + epsilon)
    upper_shadow_ratio = (high_series - pd.concat([open_series, close_series], axis=1).max(axis=1)) / (high_low_range + epsilon)
    lower_shadow_ratio = (pd.concat([open_series, close_series], axis=1).min(axis=1) - low_series) / (high_low_range + epsilon)
    feature_frame["body_ratio"] = body_ratio
    feature_frame["upper_shadow_ratio"] = upper_shadow_ratio
    feature_frame["lower_shadow_ratio"] = lower_shadow_ratio
    feature_frame["open_pos"] = (2.0 * (open_series - low_series) / (high_low_range + epsilon)) - 1.0
    feature_frame["close_pos"] = (2.0 * (close_series - low_series) / (high_low_range + epsilon)) - 1.0
    feature_frame["doji_flag"] = (body_ratio < 0.1).astype("float64")
    feature_frame["hammer_flag"] = ((lower_shadow_ratio > 0.6) & (body_ratio < 0.3) & (upper_shadow_ratio < 0.1)).astype("float64")
    feature_frame["shooting_star_flag"] = ((upper_shadow_ratio > 0.6) & (body_ratio < 0.3) & (lower_shadow_ratio < 0.1)).astype("float64")
    feature_frame["bullish_engulfing_flag"] = (
        (close_series.shift(1) < open_series.shift(1))
        & (close_series > open_series)
        & (open_series < close_series.shift(1))
        & (close_series > open_series.shift(1))
    ).astype("float64")
    feature_frame["bearish_engulfing_flag"] = (
        (close_series.shift(1) > open_series.shift(1))
        & (close_series < open_series)
        & (open_series > close_series.shift(1))
        & (close_series < open_series.shift(1))
    ).astype("float64")

    for window_size in (5, 20, 60):
        feature_frame[f"mean_ret_{window_size}"] = _simple_moving_average(ret_1, window_size)
    feature_frame["ret_zscore_5"] = (ret_1 - feature_frame["mean_ret_5"]) / (_rolling_standard_deviation(ret_1, 5) + epsilon)
    feature_frame["ret_zscore_20"] = (ret_1 - feature_frame["mean_ret_20"]) / (_rolling_standard_deviation(ret_1, 20) + epsilon)
    feature_frame["skew_ret_20"] = np.tanh(_compute_rolling_skew(ret_1, 20, epsilon))
    feature_frame["kurt_ret_20"] = np.tanh(_compute_rolling_kurtosis(ret_1, 20, epsilon) / 10.0)
    feature_frame["mdd_20"] = _compute_rolling_maximum_drawdown(close_series, 20, epsilon)
    feature_frame["mdd_60"] = _compute_rolling_maximum_drawdown(close_series, 60, epsilon)
    is_up_day = (ret_1 > 0.0).astype("float64")
    is_down_day = (ret_1 < 0.0).astype("float64")
    feature_frame["up_ratio_20"] = _simple_moving_average(is_up_day, 20)
    feature_frame["avg_up_20"] = _rolling_sum(ret_1.where(ret_1 > 0.0, 0.0), 20) / (_rolling_sum(is_up_day, 20) + epsilon)
    feature_frame["avg_down_20"] = _rolling_sum(ret_1.abs().where(ret_1 < 0.0, 0.0), 20) / (_rolling_sum(is_down_day, 20) + epsilon)
    feature_frame["autocorr_20_lag1"] = _compute_rolling_lag_one_autocorrelation(ret_1, 20, epsilon)
    feature_frame["trend_consistency_20"] = _rolling_sum(ret_1, 20).abs() / (_rolling_sum(ret_1.abs(), 20) + epsilon)

    feature_frame["dominant_cycle_scaled_20"] = _compute_dominant_cycle_scaled(ret_1, 20, epsilon)
    feature_frame["dominant_cycle_scaled_60"] = _compute_dominant_cycle_scaled(ret_1, 60, epsilon)
    hilbert_phase_sin, hilbert_phase_cos = _compute_hilbert_phase(close_series)
    feature_frame["hilbert_phase_sin"] = hilbert_phase_sin
    feature_frame["hilbert_phase_cos"] = hilbert_phase_cos

    return _sanitize_feature_frame(feature_frame)
