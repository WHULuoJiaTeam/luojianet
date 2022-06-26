# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
Validators for Audio processing operations.
"""

from functools import wraps

import numpy as np

from mindspore.dataset.core.validator_helpers import check_float32, check_float32_not_zero, check_int32, \
    check_int32_not_zero, check_list_same_size, check_non_negative_float32, check_non_negative_int32, \
    check_pos_float32, check_pos_int32, check_value, INT32_MAX, parse_user_args, type_check
from .utils import BorderType, DensityFunction, FadeShape, GainType, Interpolation, MelType, Modulation, NormType, \
    ScaleType, WindowType


def check_amplitude_to_db(method):
    """Wrapper method to check the parameters of AmplitudeToDB."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [stype, ref_value, amin, top_db], _ = parse_user_args(method, *args, **kwargs)

        type_check(stype, (ScaleType,), "stype")

        type_check(ref_value, (int, float), "ref_value")
        if ref_value is not None:
            check_pos_float32(ref_value, "ref_value")

        type_check(amin, (int, float), "amin")
        if amin is not None:
            check_pos_float32(amin, "amin")

        type_check(top_db, (int, float), "top_db")
        if top_db is not None:
            check_pos_float32(top_db, "top_db")

        return method(self, *args, **kwargs)

    return new_method


def check_biquad_sample_rate(sample_rate):
    """Wrapper method to check the parameters of sample_rate."""
    type_check(sample_rate, (int,), "sample_rate")
    check_int32_not_zero(sample_rate, "sample_rate")


def check_biquad_central_freq(central_freq):
    """Wrapper method to check the parameters of central_freq."""
    type_check(central_freq, (float, int), "central_freq")
    check_float32(central_freq, "central_freq")


def check_biquad_q(q):
    """Wrapper method to check the parameters of Q."""
    type_check(q, (float, int), "Q")
    check_value(q, [0, 1], "Q", True)


def check_biquad_noise(noise):
    """Wrapper method to check the parameters of noise."""
    type_check(noise, (bool,), "noise")


def check_biquad_const_skirt_gain(const_skirt_gain):
    """Wrapper method to check the parameters of const_skirt_gain."""
    type_check(const_skirt_gain, (bool,), "const_skirt_gain")


def check_biquad_gain(gain):
    """Wrapper method to check the parameters of gain."""
    type_check(gain, (float, int), "gain")
    check_float32(gain, "gain")


def check_band_biquad(method):
    """Wrapper method to check the parameters of BandBiquad."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [sample_rate, central_freq, q, noise], _ = parse_user_args(
            method, *args, **kwargs)
        check_biquad_sample_rate(sample_rate)
        check_biquad_central_freq(central_freq)
        check_biquad_q(q)
        check_biquad_noise(noise)
        return method(self, *args, **kwargs)

    return new_method


def check_biquad_cutoff_freq(cutoff_freq):
    """Wrapper method to check the parameters of cutoff_freq."""
    type_check(cutoff_freq, (float, int), "cutoff_freq")
    check_float32(cutoff_freq, "cutoff_freq")


def check_highpass_biquad(method):
    """Wrapper method to check the parameters of HighpassBiquad."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [sample_rate, cutoff_freq, q], _ = parse_user_args(method, *args, **kwargs)
        check_biquad_sample_rate(sample_rate)
        check_biquad_cutoff_freq(cutoff_freq)
        check_biquad_q(q)
        return method(self, *args, **kwargs)

    return new_method


def check_allpass_biquad(method):
    """Wrapper method to check the parameters of AllpassBiquad."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [sample_rate, central_freq, q], _ = parse_user_args(
            method, *args, **kwargs)
        check_biquad_sample_rate(sample_rate)
        check_biquad_central_freq(central_freq)
        check_biquad_q(q)
        return method(self, *args, **kwargs)

    return new_method


def check_bandpass_biquad(method):
    """Wrapper method to check the parameters of BandpassBiquad."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [sample_rate, central_freq, q, const_skirt_gain], _ = parse_user_args(
            method, *args, **kwargs)
        check_biquad_sample_rate(sample_rate)
        check_biquad_central_freq(central_freq)
        check_biquad_q(q)
        check_biquad_const_skirt_gain(const_skirt_gain)
        return method(self, *args, **kwargs)

    return new_method


def check_bandreject_biquad(method):
    """Wrapper method to check the parameters of BandrejectBiquad."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [sample_rate, central_freq, q], _ = parse_user_args(
            method, *args, **kwargs)
        check_biquad_sample_rate(sample_rate)
        check_biquad_central_freq(central_freq)
        check_biquad_q(q)
        return method(self, *args, **kwargs)

    return new_method


def check_bass_biquad(method):
    """Wrapper method to check the parameters of BassBiquad."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [sample_rate, gain, central_freq, q], _ = parse_user_args(
            method, *args, **kwargs)
        check_biquad_sample_rate(sample_rate)
        check_biquad_gain(gain)
        check_biquad_central_freq(central_freq)
        check_biquad_q(q)
        return method(self, *args, **kwargs)

    return new_method


def check_contrast(method):
    """Wrapper method to check the parameters of Contrast."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [enhancement_amount], _ = parse_user_args(method, *args, **kwargs)
        type_check(enhancement_amount, (float, int), "enhancement_amount")
        check_value(enhancement_amount, [0, 100], "enhancement_amount")
        return method(self, *args, **kwargs)

    return new_method


def check_db_to_amplitude(method):
    """Wrapper method to check the parameters of db_to_amplitude."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [ref, power], _ = parse_user_args(method, *args, **kwargs)
        type_check(ref, (float, int), "ref")
        check_float32(ref, "ref")
        type_check(power, (float, int), "power")
        check_float32(power, "power")
        return method(self, *args, **kwargs)

    return new_method


def check_dc_shift(method):
    """Wrapper method to check the parameters of DCShift."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [shift, limiter_gain], _ = parse_user_args(method, *args, **kwargs)
        type_check(shift, (float, int), "shift")
        check_value(shift, [-2.0, 2.0], "shift")
        if limiter_gain is not None:
            type_check(limiter_gain, (float, int), "limiter_gain")
        return method(self, *args, **kwargs)

    return new_method


def check_deemph_biquad(method):
    """Wrapper method to check the parameters of CutMixBatch."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [sample_rate], _ = parse_user_args(method, *args, **kwargs)
        type_check(sample_rate, (int,), "sample_rate")
        if sample_rate not in (44100, 48000):
            raise ValueError("Argument sample_rate should be 44100 or 48000, but got {0}.".format(sample_rate))
        return method(self, *args, **kwargs)

    return new_method


def check_dither(method):
    """Wrapper method to check the parameters of Dither."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [density_function, noise_shaping], _ = parse_user_args(
            method, *args, **kwargs)

        type_check(density_function, (DensityFunction), "density_function")
        type_check(noise_shaping, (bool,), "noise_shaping")

        return method(self, *args, **kwargs)

    return new_method


def check_equalizer_biquad(method):
    """Wrapper method to check the parameters of EqualizerBiquad."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [sample_rate, center_freq, gain, q], _ = parse_user_args(method, *args, **kwargs)
        check_biquad_sample_rate(sample_rate)
        check_biquad_central_freq(center_freq)
        check_biquad_gain(gain)
        check_biquad_q(q)
        return method(self, *args, **kwargs)

    return new_method


def check_gain(method):
    """Wrapper method to check the parameters of Gain."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [gain_db], _ = parse_user_args(method, *args, **kwargs)
        type_check(gain_db, (float, int), "gain_db")
        check_float32(gain_db, "gain_db")
        return method(self, *args, **kwargs)

    return new_method


def check_lfilter(method):
    """Wrapper method to check the parameters of LFilter."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [a_coeffs, b_coeffs, clamp], _ = parse_user_args(method, *args, **kwargs)
        type_check(a_coeffs, (list, tuple), "a_coeffs")
        type_check(b_coeffs, (list, tuple), "b_coeffs")
        for i, value in enumerate(a_coeffs):
            type_check(value, (float, int), "a_coeffs[{0}]".format(i))
            check_float32(value, "a_coeffs[{0}]".format(i))
        for i, value in enumerate(b_coeffs):
            type_check(value, (float, int), "b_coeffs[{0}]".format(i))
            check_float32(value, "b_coeffs[{0}]".format(i))
        check_list_same_size(a_coeffs, b_coeffs, "a_coeffs", "b_coeffs")
        type_check(clamp, (bool,), "clamp")
        return method(self, *args, **kwargs)

    return new_method


def check_lowpass_biquad(method):
    """Wrapper method to check the parameters of LowpassBiquad."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [sample_rate, cutoff_freq, q], _ = parse_user_args(method, *args, **kwargs)
        check_biquad_sample_rate(sample_rate)
        check_biquad_cutoff_freq(cutoff_freq)
        check_biquad_q(q)
        return method(self, *args, **kwargs)

    return new_method


def check_mask_along_axis(method):
    """Wrapper method to check the parameters of MaskAlongAxis."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [mask_start, mask_width, mask_value, axis], _ = parse_user_args(method, *args, **kwargs)
        type_check(mask_start, (int,), "mask_start")
        type_check(mask_width, (int,), "mask_width")
        type_check(mask_value, (int, float), "mask_value")
        type_check(axis, (int,), "axis")
        check_non_negative_int32(mask_start, "mask_start")
        check_pos_int32(mask_width, "mask_width")
        check_float32(mask_value, "mask_value")
        check_value(axis, [1, 2], "axis")
        return method(self, *args, **kwargs)

    return new_method


def check_mask_along_axis_iid(method):
    """Wrapper method to check the parameters of MaskAlongAxisIID."""
    @wraps(method)
    def new_method(self, *args, **kwargs):
        [mask_param, mask_value, axis], _ = parse_user_args(method, *args, **kwargs)
        type_check(mask_param, (int,), "mask_param")
        check_non_negative_int32(mask_param, "mask_param")
        type_check(mask_value, (int, float,), "mask_value")
        check_float32(mask_value, "mask_value")
        type_check(axis, (int,), "axis")
        check_value(axis, [1, 2], "axis")
        return method(self, *args, **kwargs)

    return new_method


def check_mu_law_coding(method):
    """Wrapper method to check the parameters of MuLawDecoding and MuLawEncoding"""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [quantization_channels], _ = parse_user_args(method, *args, **kwargs)
        check_pos_int32(quantization_channels, "quantization_channels")
        return method(self, *args, **kwargs)

    return new_method


def check_overdrive(method):
    """Wrapper method to check the parameters of Overdrive."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [gain, color], _ = parse_user_args(method, *args, **kwargs)
        type_check(gain, (float, int), "gain")
        check_value(gain, [0, 100], "gain")
        type_check(color, (float, int), "color")
        check_value(color, [0, 100], "color")
        return method(self, *args, **kwargs)

    return new_method


def check_phaser(method):
    """Wrapper method to check the parameters of Phaser."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [sample_rate, gain_in, gain_out, delay_ms, decay,
         mod_speed, sinusoidal], _ = parse_user_args(method, *args, **kwargs)
        type_check(sample_rate, (int,), "sample_rate")
        check_int32(sample_rate, "sample_rate")
        type_check(gain_in, (float, int), "gain_in")
        check_value(gain_in, [0, 1], "gain_in")
        type_check(gain_out, (float, int), "gain_out")
        check_value(gain_out, [0, 1e9], "gain_out")
        type_check(delay_ms, (float, int), "delay_ms")
        check_value(delay_ms, [0, 5.0], "delay_ms")
        type_check(decay, (float, int), "decay")
        check_value(decay, [0, 0.99], "decay")
        type_check(mod_speed, (float, int), "mod_speed")
        check_value(mod_speed, [0.1, 2], "mod_speed")
        type_check(sinusoidal, (bool,), "sinusoidal")
        return method(self, *args, **kwargs)

    return new_method


def check_riaa_biquad(method):
    """Wrapper method to check the parameters of RiaaBiquad."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [sample_rate], _ = parse_user_args(method, *args, **kwargs)
        type_check(sample_rate, (int,), "sample_rate")
        if sample_rate not in (44100, 48000, 88200, 96000):
            raise ValueError("sample_rate should be one of [44100, 48000, 88200, 96000], but got {0}.".format(
                sample_rate))
        return method(self, *args, **kwargs)

    return new_method


def check_spectrogram(method):
    """Wrapper method to check the parameters of Spectrogram."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [n_fft, win_length, hop_length, pad, window, power,
         normalized, center, pad_mode, onesided], _ = parse_user_args(method, *args, **kwargs)
        type_check(n_fft, (int,), "n_fft")
        check_pos_int32(n_fft, "n_fft")
        if win_length is not None:
            type_check(win_length, (int,), "win_length")
            check_pos_int32(win_length, "win_length")
            if win_length > n_fft:
                raise ValueError(
                    "Input win_length should be no more than n_fft, but got win_length: {0} and n_fft: {1}.".format(
                        win_length, n_fft))
        if hop_length is not None:
            type_check(hop_length, (int,), "hop_length")
            check_pos_int32(hop_length, "hop_length")

        type_check(pad, (int,), "pad")
        check_non_negative_int32(pad, "pad")
        type_check(window, (WindowType,), "window")
        type_check(power, (int, float), "power")
        check_non_negative_float32(power, "power")
        type_check(normalized, (bool,), "normalized")
        type_check(center, (bool,), "center")
        type_check(onesided, (bool,), "onesided")
        type_check(pad_mode, (BorderType,), "pad_mode")
        return method(self, *args, **kwargs)

    return new_method


def check_time_stretch(method):
    """Wrapper method to check the parameters of TimeStretch."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [hop_length, n_freq, fixed_rate], _ = parse_user_args(method, *args, **kwargs)

        if hop_length is not None:
            type_check(hop_length, (int,), "hop_length")
            check_pos_int32(hop_length, "hop_length")

        type_check(n_freq, (int,), "n_freq")
        check_pos_int32(n_freq, "n_freq")

        if fixed_rate is not None:
            type_check(fixed_rate, (int, float), "fixed_rate")
            check_pos_float32(fixed_rate, "fixed_rate")
        return method(self, *args, **kwargs)

    return new_method


def check_treble_biquad(method):
    """Wrapper method to check the parameters of TrebleBiquad."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [sample_rate, gain, central_freq, q], _ = parse_user_args(
            method, *args, **kwargs)
        check_biquad_sample_rate(sample_rate)
        check_biquad_gain(gain)
        check_biquad_central_freq(central_freq)
        check_biquad_q(q)
        return method(self, *args, **kwargs)

    return new_method


def check_masking(method):
    """Wrapper method to check the parameters of TimeMasking and FrequencyMasking"""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [iid_masks, mask_param, mask_start, mask_value], _ = parse_user_args(
            method, *args, **kwargs)
        type_check(iid_masks, (bool,), "iid_masks")
        type_check(mask_param, (int,), "mask_param")
        check_non_negative_float32(mask_param, "mask_param")
        type_check(mask_start, (int,), "mask_start")
        check_non_negative_float32(mask_start, "mask_start")
        type_check(mask_value, (int, float), "mask_value")
        check_non_negative_float32(mask_value, "mask_value")
        return method(self, *args, **kwargs)
    return new_method


def check_mel_scale(method):
    """Wrapper method to check the parameters of MelScale."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [n_mels, sample_rate, f_min, f_max, n_stft, norm, mel_type], _ = parse_user_args(method, *args, **kwargs)

        type_check(n_mels, (int,), "n_mels")
        check_pos_int32(n_mels, "n_mels")

        type_check(sample_rate, (int,), "sample_rate")
        check_pos_int32(sample_rate, "sample_rate")

        type_check(f_min, (int, float), "f_min")
        check_float32(f_min, "f_min")

        if f_max is not None:
            type_check(f_max, (int, float), "f_max")
            check_pos_float32(f_max, "f_max")
            if f_min >= f_max:
                raise ValueError("MelScale: f_max should be greater than f_min.")
        else:
            if f_min >= sample_rate // 2:
                raise ValueError("MelScale: sample_rate // 2 should be greater than f_min when f_max is set to None.")

        type_check(n_stft, (int,), "n_stft")
        check_pos_int32(n_stft, "n_stft")

        type_check(norm, (NormType,), "norm")

        type_check(mel_type, (MelType,), "mel_type")

        return method(self, *args, **kwargs)

    return new_method


def check_power(power):
    """Wrapper method to check the parameters of power."""
    type_check(power, (int, float), "power")
    check_non_negative_float32(power, "power")


def check_complex_norm(method):
    """Wrapper method to check the parameters of ComplexNorm."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [power], _ = parse_user_args(method, *args, **kwargs)
        check_power(power)
        return method(self, *args, **kwargs)

    return new_method


def check_magphase(method):
    """Wrapper method to check the parameters of Magphase."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [power], _ = parse_user_args(method, *args, **kwargs)
        check_power(power)
        return method(self, *args, **kwargs)

    return new_method


def check_biquad_coeff(coeff, arg_name):
    """Wrapper method to check the parameters of coeff."""
    type_check(coeff, (float, int), arg_name)
    check_float32(coeff, arg_name)


def check_biquad(method):
    """Wrapper method to check the parameters of Biquad."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [b0, b1, b2, a0, a1, a2], _ = parse_user_args(
            method, *args, **kwargs)
        check_biquad_coeff(b0, "b0")
        check_biquad_coeff(b1, "b1")
        check_biquad_coeff(b2, "b2")
        type_check(a0, (float, int), "a0")
        check_float32_not_zero(a0, "a0")
        check_biquad_coeff(a1, "a1")
        check_biquad_coeff(a2, "a2")
        return method(self, *args, **kwargs)

    return new_method


def check_fade(method):
    """Wrapper method to check the parameters of Fade."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [fade_in_len, fade_out_len, fade_shape], _ = parse_user_args(method, *args, **kwargs)
        type_check(fade_in_len, (int,), "fade_in_len")
        check_non_negative_int32(fade_in_len, "fade_in_len")
        type_check(fade_out_len, (int,), "fade_out_len")
        check_non_negative_int32(fade_out_len, "fade_out_len")
        type_check(fade_shape, (FadeShape,), "fade_shape")
        return method(self, *args, **kwargs)

    return new_method


def check_vol(method):
    """Wrapper method to check the parameters of Vol."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [gain, gain_type], _ = parse_user_args(method, *args, **kwargs)
        type_check(gain, (int, float), "gain")
        type_check(gain_type, (GainType,), "gain_type")
        if gain_type == GainType.AMPLITUDE:
            check_non_negative_float32(gain, "gain")
        elif gain_type == GainType.POWER:
            check_pos_float32(gain, "gain")
        else:
            check_float32(gain, "gain")
        return method(self, *args, **kwargs)

    return new_method


def check_detect_pitch_frequency(method):
    """Wrapper method to check the parameters of DetectPitchFrequency."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [sample_rate, frame_time, win_length, freq_low, freq_high], _ = parse_user_args(
            method, *args, **kwargs)
        type_check(sample_rate, (int,), "sample_rate")
        check_int32_not_zero(sample_rate, "sample_rate")
        type_check(frame_time, (float, int), "frame_time")
        check_pos_float32(frame_time, "frame_time")
        type_check(win_length, (int,), "win_length")
        check_pos_int32(win_length, "win_length")
        type_check(freq_low, (int, float), "freq_low")
        check_pos_float32(freq_low, "freq_low")
        type_check(freq_high, (int, float), "freq_high")
        check_pos_float32(freq_high, "freq_high")
        return method(self, *args, **kwargs)

    return new_method


def check_flanger(method):
    """Wrapper method to check the parameters of Flanger."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [sample_rate, delay, depth, regen, width, speed, phase, modulation, interpolation], _ = parse_user_args(
            method, *args, **kwargs)
        type_check(sample_rate, (int,), "sample_rate")
        check_int32_not_zero(sample_rate, "sample_rate")

        type_check(delay, (float, int), "delay")
        check_value(delay, [0, 30], "delay")

        type_check(depth, (float, int), "depth")
        check_value(depth, [0, 10], "depth")

        type_check(regen, (float, int), "regen")
        check_value(regen, [-95, 95], "regen")

        type_check(width, (float, int), "width")
        check_value(width, [0, 100], "width")

        type_check(speed, (float, int), "speed")
        check_value(speed, [0.1, 10], "speed")

        type_check(phase, (float, int), "phase")
        check_value(phase, [0, 100], "phase")

        type_check(modulation, (Modulation,), "modulation")
        type_check(interpolation, (Interpolation,), "interpolation")
        return method(self, *args, **kwargs)

    return new_method


def check_sliding_window_cmn(method):
    """Wrapper method to check the parameters of SlidingWidowCmn."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [cmn_window, min_cmn_window, center, norm_vars], _ = parse_user_args(method, *args, **kwargs)

        type_check(cmn_window, (int,), "cmn_window")
        check_non_negative_int32(cmn_window, "cmn_window")

        type_check(min_cmn_window, (int,), "min_cmn_window")
        check_non_negative_int32(min_cmn_window, "min_cmn_window")

        type_check(center, (bool,), "center")
        type_check(norm_vars, (bool,), "norm_vars")
        return method(self, *args, **kwargs)

    return new_method


def check_compute_deltas(method):
    """Wrapper method to check the parameter of ComputeDeltas."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [win_length, pad_mode], _ = parse_user_args(method, *args, **kwargs)
        type_check(pad_mode, (BorderType,), "pad_mode")
        type_check(win_length, (int,), "win_length")
        check_value(win_length, (3, INT32_MAX), "win_length")
        return method(self, *args, **kwargs)

    return new_method


def check_spectral_centroid(method):
    """Wrapper method to check the parameters of SpectralCentroid."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [sample_rate, n_fft, win_length, hop_length, pad, window], _ = parse_user_args(method, *args, **kwargs)
        type_check(sample_rate, (int,), "sample_rate")
        check_non_negative_int32(sample_rate, "sample_rate")
        type_check(pad, (int,), "pad")
        check_non_negative_int32(pad, "pad")
        type_check(window, (WindowType,), "window")
        type_check(n_fft, (int,), "n_fft")
        check_pos_int32(n_fft, "n_fft")
        if win_length is not None:
            type_check(win_length, (int,), "win_length")
            check_pos_int32(win_length, "win_length")
            if win_length > n_fft:
                raise ValueError(
                    "Input win_length should be no more than n_fft, but got win_length: {0} and n_fft: {1}.".format(
                        win_length, n_fft))
        if hop_length is not None:
            type_check(hop_length, (int,), "hop_length")
            check_pos_int32(hop_length, "hop_length")

        return method(self, *args, **kwargs)

    return new_method


def check_phase_vocoder(method):
    """Wrapper method to check the parameters of PhaseVocoder."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [rate, phase_advance], _ = parse_user_args(method, *args, **kwargs)
        type_check(rate, (int, float), "rate")
        check_pos_float32(rate, "rate")
        type_check(phase_advance, (np.ndarray,), "phase_advance")
        return method(self, *args, **kwargs)

    return new_method
