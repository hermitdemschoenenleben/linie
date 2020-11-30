import pickle
import traceback
import numpy as np
from linien.common import get_lock_point, combine_error_signal, \
    check_plot_data, ANALOG_OUT0, SpectrumUncorrelatedException
from linien.server.approach_line import Approacher


class Autolock:
    """Spectroscopy autolock based on correlation."""
    def __init__(self, control, parameters, wait_time_between_current_corrections=None):
        self.control = control
        self.parameters = parameters

        self.first_error_signal = None
        self.parameters.autolock_running.value = False
        self.parameters.autolock_retrying.value = False

        self.should_watch_lock = False
        self.approacher = None
        self._data_listener_added = False

        self.reset_properties()

        self.wait_time_between_current_corrections = wait_time_between_current_corrections

    def reset_properties(self):
        # we check each parameter before setting it because otherwise
        # this may crash the client if called very often (e.g.if the
        # autolock continuously fails)
        if self.parameters.autolock_failed.value:
            self.parameters.autolock_failed.value = False
        if self.parameters.autolock_locked.value:
            self.parameters.autolock_locked.value = False
        if self.parameters.autolock_watching.value:
            self.parameters.autolock_watching.value = False

        if self.approacher:
            self.approacher.reset_properties()

    def run(self, x0, x1, spectrum, should_watch_lock=False, auto_offset=True):
        """Starts the autolock.

        If `should_watch_lock` is specified, the autolock continuously monitors
        the control and error signals after the lock was successful and tries to
        relock automatically using the spectrum that was recorded in the first
        run of the lock.
        """
        self.parameters.autolock_running.value = True
        self.x0, self.x1 = int(x0), int(x1)
        self.should_watch_lock = should_watch_lock
        self.auto_offset = auto_offset

        self.parameters.autolock_approaching.value = True
        self.record_first_error_signal(spectrum)

        self.initial_ramp_speed = self.parameters.ramp_speed.value
        self.parameters.autolock_initial_ramp_amplitude.value = self.parameters.ramp_amplitude.value
        self.initial_ramp_center = self.parameters.center.value

        self.add_data_listener()

    def add_data_listener(self):
        if not self._data_listener_added:
            self._data_listener_added = True
            self.parameters.to_plot.change(self.react_to_new_spectrum)

    def remove_data_listener(self):
        self._data_listener_added = False
        self.parameters.to_plot.remove_listener(self.react_to_new_spectrum)

    def react_to_new_spectrum(self, plot_data):
        """React to new spectrum data.

        If this is executed for the first time, a reference spectrum is
        recorded.

        If the autolock is approaching the desired line, a correlation
        function of the spectrum with the reference spectrum is calculated
        and the laser current is adapted such that the targeted line is centered.

        After this procedure is done, the real lock is turned on and after some
        time the lock is verified.

        If automatic relocking is desired, the control and error signals are
        continuously monitored after locking.
        """
        if self.parameters.pause_acquisition.value:
            return

        if plot_data is None or not self.parameters.autolock_running.value:
            return

        plot_data = pickle.loads(plot_data)
        if plot_data is None:
            return

        is_locked = self.parameters.lock.value

        # check that `plot_data` contains the information we need
        # otherwise skip this round
        if not check_plot_data(is_locked, plot_data):
            return

        if is_locked:
            error_signal = plot_data['error_signal']
            control_signal = plot_data['control_signal']
        else:
            combined_error_signal = combine_error_signal(
                (plot_data['error_signal_1'], plot_data['error_signal_2']),
                self.parameters.dual_channel.value,
                self.parameters.channel_mixing.value,
                self.parameters.combined_offset.value
            )

        try:
            if self.parameters.autolock_approaching.value:
                # we have already recorded a spectrum and are now approaching
                # the line by decreasing the scan range and adapting the
                # center current multiple times.
                if self.approacher is None:
                    self.approacher = Approacher(
                        self.control, self.parameters, self.first_error_signal,
                        self.target_zoom, self.central_y,
                        allow_ramp_speed_change=True,
                        wait_time_between_current_corrections=self.wait_time_between_current_corrections
                    )
                final_shift = self.approacher.approach_line(combined_error_signal)
                approaching_finished = final_shift is not None
                if approaching_finished:
                    # FIXME: use the final shift here
                    self._lock()

            elif self.parameters.autolock_watching.value:
                # the laser was locked successfully before. Now we check
                # periodically whether the laser is still in lock
                return self.watch_lock(error_signal, control_signal)

            else:
                # we are done with approaching and have started the lock.
                # skip some data and check whether we really are in lock
                # afterwards.
                return self.after_lock(error_signal, control_signal, plot_data.get('slow'))

        except SpectrumUncorrelatedException:
            print('spectrum uncorrelated')
            if self.parameters.watch_lock.value:
                print('retry')
                self.relock()
            else:
                self.exposed_stop()
                self.parameters.autolock_failed.value = True

        except Exception:
            traceback.print_exc()
            self.exposed_stop()

    def record_first_error_signal(self, error_signal):
        mean_signal, target_slope_rising, target_zoom, rolled_error_signal = \
            get_lock_point(error_signal, self.x0, self.x1)

        self.central_y = mean_signal

        self.parameters.target_slope_rising.value = target_slope_rising
        self.control.exposed_write_data()

        self.target_zoom = target_zoom
        self.first_error_signal = rolled_error_signal

    def after_lock(self, error_signal, control_signal, slow_out):
        """After locking, this method checks whether the laser really is locked.

        If desired, it automatically tries to relock if locking failed, or
        starts a watcher that does so over and over again.
        """
        def check_whether_in_lock(control_signal):
            """
            The laser is considered in lock if the mean value of the control
            signal is within the boundaries of the smallest current ramp we had
            before turning on the lock.
            """
            center = self.parameters.center.value
            initial_ampl = self.parameters.autolock_initial_ramp_amplitude.value
            ampl = initial_ampl / self.target_zoom

            slow_ramp = self.parameters.sweep_channel.value == ANALOG_OUT0
            slow_pid = self.parameters.pid_on_slow_enabled.value

            if not slow_ramp and not slow_pid:
                mean = np.mean(control_signal) / 8192
                return (center - ampl) <= mean <= (center + ampl)
            else:
                if slow_pid and not slow_ramp:
                    # we cannot handle this case. Just assume the laser is locked.
                    return True

                return (center - ampl) <= slow_out / 8192 <= (center + ampl)

        self.parameters.autolock_locked.value = check_whether_in_lock(control_signal) \
            if self.parameters.check_lock.value \
            else True

        if self.parameters.autolock_locked.value and self.should_watch_lock:
            # we start watching the lock status from now on.
            # this is done in `react_to_new_spectrum()` which is called regularly.
            self.watcher_last_value = np.mean(control_signal) / 8192
            self.parameters.autolock_watching.value = True
        else:
            self.remove_data_listener()

            if not self.parameters.autolock_locked.value:
                if self.should_watch_lock:
                    return self.relock()

                self.exposed_stop()
                self.parameters.autolock_failed.value = True

            self.parameters.autolock_running.value = False

    def watch_lock(self, error_signal, control_signal):
        """Check whether the laser is still in lock and init a relock if not."""
        mean = np.mean(control_signal) / 8192

        diff = np.abs(mean - self.watcher_last_value)
        lock_lost = diff > self.parameters.watch_lock_threshold.value

        too_close_to_edge = np.abs(mean) > 0.95

        if too_close_to_edge or lock_lost:
            self.relock()

        self.watcher_last_value = mean

    def relock(self):
        """
        Relock the laser using the reference spectrum recorded in the first
        locking approach.
        """
        # we check each parameter before setting it because otherwise
        # this may crash the client if called very often (e.g.if the
        # autolock continuously fails)
        if not self.parameters.autolock_running.value:
            self.parameters.autolock_running.value = True
        if not self.parameters.autolock_approaching.value:
            self.parameters.autolock_approaching.value = True
        if not self.parameters.autolock_retrying.value:
            self.parameters.autolock_retrying.value = True

        self.reset_properties()
        self._reset_scan()

        # add a listener that listens for new spectrum data and consequently
        # tries to relock.
        self.add_data_listener()

    def exposed_stop(self):
        """Abort any operation."""
        self.parameters.autolock_running.value = False
        self.parameters.autolock_locked.value = False
        self.parameters.autolock_approaching.value = False
        self.parameters.autolock_watching.value = False
        self.remove_data_listener()

        self._reset_scan()
        self.parameters.task.value = None

    def _lock(self):
        self.control.pause_acquisition()

        # acquisition in locked state doesn't care about ramp speed
        # therefore, we can reset it here
        self.parameters.ramp_speed.value = self.initial_ramp_speed
        self.parameters.autolock_approaching.value = False

        if self.auto_offset:
            # note: we only set the offset directly before turning on the lock
            # and not when approaching because this would cause problems in approacher
            self.parameters.combined_offset.value = -1 * self.central_y

        self.control.exposed_start_lock()

        self.control.continue_acquisition()

    def _reset_scan(self):
        self.control.pause_acquisition()

        self.parameters.center.value = self.initial_ramp_center
        self.parameters.ramp_amplitude.value = self.parameters.autolock_initial_ramp_amplitude.value
        self.parameters.ramp_speed.value = self.initial_ramp_speed
        self.control.exposed_start_ramp()

        self.control.continue_acquisition()