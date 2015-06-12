class PID(object):
    ''' 
    PID loop implementation. A short description of the semantic meanings of
    PID gains:

    Proportional gain KP is direct measure of error - the higher this value the
    faster the loop will try to bring the system to the set point.

    Integral gain KI is an error over time - some error that KP is not removing
    consistently through multiple loop updates. TODO: Provide example

    '''
    def __init__(self, KP, KI, KD, min_cor, max_cor):
        '''
        Initializes instance of PID loop.

        @param KP - proportional gain
        @param KI - integral gain
        @param KD - derivative gain
        @param min_cor - minimum value this PID loop can produce
        @param max_cor - maximum value this PID loop can produce
        '''
        self._kp = KP
        self._ki = KI
        self._kd = KD
        self._min_cor = min_cor
        self._max_cor = max_cor
        self._target = None
        self._error_sum = 0 # integral term
        self._last_error = 0 # used for derivative term

    def make_correction(self, new_value, passed_time):
        '''
        Calculates the correction to reach the target.

        @param new_value - current value of the system
        '''
        if passed_time <= 0:
            raise ValueError("Passed time must be a positive number.")

        error = self._target - new_value
        prop_correction = self._kp * error

        self._error_sum += error
        integral_correction = self._ki * self._error_sum

        slope = error - self._last_error
        derivative_correction = slope * self._kd / passed_time

        self._last_error = error

        correction = sum((prop_correction, integral_correction, derivative_correction))
        correction = min(correction, self._max_cor)
        correction = max(correction, self._min_cor)
        return correction

    @property
    def target(self):
        '''
        Current target that PID loop is trying to reach.
        '''
        return self._target

    @target.setter
    def target(self, new_target):
        self._target = new_target
        self._error_sum = 0
        self._last_error = 0
