from os import error
import pickle
import numpy as np
from time import time
from scipy.signal import correlate, resample

MHz = 0x10000000 / 8
Vpp = ((1<<14) - 1) / 4
# conversion of bits to V
ANALOG_OUT_V = 1.8 / ((2**15) - 1)

LOW_PASS_FILTER = 0
HIGH_PASS_FILTER = 1

FAST_OUT1 = 0
FAST_OUT2 = 1
ANALOG_OUT0 = 2

DECIMATION = 8
assert DECIMATION % 2 == 0 or DECIMATION == 1
N_POINTS = int(16384 / DECIMATION)


class SpectrumUncorrelatedException(Exception):
    pass


def downsample_history(times, values, max_time_diff, max_N=N_POINTS):
    """The history should not grow too much. When recording for long intervals,
    we want to throw away some datapoints that were recorded with a sampling rate
    that is too high. This function takes care of this."""
    last_time = None

    to_remove = []

    for idx in range(len(times)):
        current_time = times[idx]
        remove = False

        if last_time is not None:
            if np.abs(last_time - current_time) < max_time_diff / max_N:
                remove = True

        if remove:
            to_remove.append(idx)
        else:
            last_time = current_time

    for idx in reversed(to_remove):
        times.pop(idx)
        values.pop(idx)


def truncate(times, values, max_time_diff):
    while len(values) > 0:
        if time() - times[0] > max_time_diff:
            times.pop(0)
            values.pop(0)
            continue
        break


def update_control_signal_history(history, to_plot, is_locked, max_time_diff):
    if not to_plot:
        return history

    now = time()

    if is_locked:
        history['values'].append(np.mean(to_plot['control_signal']))
        history['times'].append(time())

        if 'slow' in to_plot:
            history['slow_values'].append(to_plot['slow'])
            history['slow_times'].append(time())
    else:
        history['values'] = []
        history['times'] = []
        history['slow_values'] = []
        history['slow_times'] = []

    # truncate
    truncate(history['times'], history['values'], max_time_diff)
    truncate(history['slow_times'], history['slow_values'], max_time_diff)

    # downsample
    downsample_history(history['times'], history['values'], max_time_diff)
    downsample_history(history['slow_times'], history['slow_values'], max_time_diff)

    return history


def determine_shift_by_correlation(zoom_factor, reference_signal, error_signal):
    """Compares two spectra and determines the shift by correlation.

    `zoom_factor` is the zoom factor of `error_signal` with respect to
    `reference_signal`, i.e. it states how much reference signal has to be
    magnified in order to show the same region as the new error signal."""

    # values that should not be considered are np.nan
    # but the correlation has problems with np.nans
    # --> we set it to 0
    reference_signal[np.isnan(reference_signal)] = 0
    error_signal[np.isnan(error_signal)] = 0

    length = len(error_signal)
    center_idx = int(length / 2)

    # crop the reference signal such that it shows the same region as the new
    # error signal
    idx_shift = int(length * (1 / zoom_factor / 2))
    zoomed_ref = reference_signal[center_idx - idx_shift:center_idx + idx_shift]

    # correlation is slow on red pitaya --> use at maximum 4096 points
    skip_factor = int(len(zoomed_ref) / 4096)
    if skip_factor < 1:
        skip_factor = 1
    zoomed_ref = zoomed_ref[::skip_factor]

    # now sample the error signal down to the same length as the zoomed
    # reference signal
    downsampled_error_signal = resample(error_signal, len(zoomed_ref))

    correlation = correlate(zoomed_ref, downsampled_error_signal)

    if np.max(correlation) < 100 * len(zoomed_ref):
        raise SpectrumUncorrelatedException()

    shift = np.argmax(correlation)
    shift = (shift - len(zoomed_ref)) / len(zoomed_ref) * 2 / zoom_factor

    return shift, zoomed_ref, downsampled_error_signal


def get_lock_point(error_signal, x0, x1, final_zoom_factor=2):
    """Calculates parameters for the autolock based on the initial error signal.

    Takes the `error_signal` and two points (`x0` and `x1`) as arguments. The
    points are the points selected by the user, and we know that we want to
    lock between them.

    Use `final_zoom_factor` to specify how wide the line should be in the end:
    - 1: in the end, only the line should be visible
    - 5: an area of 5 times the linewidth should be visible
    """
    length = len(error_signal)

    # the data that is between the user-selected bounds
    cropped_data = np.array(error_signal[x0:x1])

    min_idx = np.argmin(cropped_data)
    max_idx = np.argmax(cropped_data)

    # the y value that is between minimum and maximum
    mean_signal = np.mean([cropped_data[min_idx], cropped_data[max_idx]])
    idxs = sorted([min_idx, max_idx])
    slope_data = np.array(cropped_data[idxs[0]:idxs[1]]) - mean_signal

    zero_idx = x0 + np.min(idxs) + np.argmin(np.abs(slope_data))

    # roll the error signal such that the target lock point is exactly in the
    # center
    roll = -int(zero_idx - (length/2))

    # set all the rolled points to nan such that they don't contribute
    # in the correlation
    filler = np.empty(abs(roll))
    filler[:] = np.nan

    if roll < 0:
        rolled_error_signal = np.hstack(
            (error_signal[-roll:], filler)
        )
    else:
        rolled_error_signal = np.hstack(
            (filler, error_signal[:-roll])
        )

    target_slope_rising = max_idx > min_idx
    target_zoom = N_POINTS / (idxs[1] - idxs[0]) / final_zoom_factor

    return mean_signal, target_slope_rising, target_zoom, rolled_error_signal


def convert_channel_mixing_value(value):
    if value <= 0:
        a_value = 128
        b_value = 128 + value
    else:
        a_value = 127 - value
        b_value = 128

    return a_value, b_value


def combine_error_signal(error_signals, dual_channel, channel_mixing,
        combined_offset, chain_factor_width=8):

    if not dual_channel:
        signal = error_signals[0]
    else:
        a_factor, b_factor = convert_channel_mixing_value(channel_mixing)

        signal = [
            (a_factor * a + b_factor * b) >> chain_factor_width
            for a, b in zip(*error_signals)
        ]

    return [v + combined_offset for v in signal]


def check_plot_data(is_locked, plot_data):
    if is_locked:
        if 'error_signal' not in plot_data or 'control_signal' not in plot_data:
            return False
    else:
        if 'error_signal_1' not in plot_data:
            return False
    return True


def pack(value):
    try:
        return pickle.dumps(value)
    except:
        # this happens when un-pickleable objects (e.g. functions) are assigned
        # to a parameter. In this case, we don't pickle it but transfer a netref
        # instead
        return value


def unpack(value):
    try:
        return pickle.loads(value)
    except:
        return value