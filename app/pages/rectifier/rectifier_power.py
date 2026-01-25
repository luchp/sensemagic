import numpy as np
import scipy.optimize as sopt

try:
    from .rectifier_base import RectifierModelBase
except ImportError:
    from rectifier_base import RectifierModelBase

class RectifierModelPower(RectifierModelBase):
    def __init__(self, T, nphase, U0, U2_min = 1.0):
        """
        Rectifier model with constant power discharge (e.g., switching power supplies)

        Parameters:
        T: Period time of sine [s]
        nphase: Number of phases if it is a half wave rectifier.
                Number of phases times two if it is a full wave rectifier.
        U0: Rectified sine amplitude [V]
        """
        assert (U2_min >= 1)
        assert (U0 > U2_min)
        super().__init__(T, nphase, U0)
        self.U2_min = max(self.U2_min, U2_min)


    def ripple_current(self, tau1, tau2, C, load_param):
        """
        Calculate RMS ripple current for constant current discharge

        The capacitor current has two distinct phases:
        1. Charging (0 to tau1, tau2 to Tn): I_C = C * dU/dt (following sine) - INDEPENDENT of Iload
        2. Discharging (tau1 to tau2): I_C = -Iload (constant) - DEPENDENT on Iload

        Parameters:
        tau1: Start of discharge [s]
        tau2: End of discharge [s]
        C: Capacitance [F]
        load_param: Load current [A]

        Returns:
        Iripple: RMS ripple current [A]
        """
        T, w, Tn, U0 = self.T, 2*np.pi/self.T, self.T/self.nphase, self.U0

        def cosint(tau):
            # Integral of (C * U0 * w * sin(wt))² from 0 to tau
            return (w * C * U0) ** 2 * (tau / 2 - np.sin(2 * w * tau) / (4 * w))

        # Charging phase 1: 0 to tau1
        charging1 = cosint(tau1)

        # Discharge contribution (DEPENDENT on Pload):
        # For constant power: I_C = -P/U
        # During quadratic discharge: U² = U1² - 2*P*(t-tau1)/C
        # So: I² = P²/U² = P²/(U1² - 2*P*(t-tau1)/C)
        #
        # Integral:
        # ∫[tau1 to tau2] P²/U² dt
        #   =  ∫[tau1 to tau2] P²/(U1² - 2*P*(t-tau1)/C)
        #
        # With U² = U1² - 2*P*(t-tau1)/C, this gives:
        #         = C * P² * ln(Uxa²/U1b²) / (2*P)
        #         = C * P * ln(Uxa/U1b)

        U1 = U0 * np.cos(w * tau1)
        U22 = max(0, U1**2 - 2*load_param*(tau2-tau1)/C)
        U2 = max(np.sqrt(U22), self.U2_min)
        discharging = C * load_param * np.log(U1 / U2)

        # Charging phase 2: tau2 to Tn
        # In case of 1 phase rectifier, the capacitor can be empty before start of next sine
        tau2_eff = max(tau2, Tn - 0.25 * T)
        charging2 = cosint(Tn - tau2_eff)

        # RMS current over one period
        Iripple = np.sqrt((charging1 + discharging + charging2) / Tn)
        return Iripple

    def solve_U1(self, C: float, load_param: float) -> tuple[float, float, float, float]:
        """
        Calculate discharge curve parameters for constant power load

        Args:
            C: Capacitance [F]
            load_param: Load power [W]

        Returns:
            tau1: Start of discharge [s]
            U1: Voltage at tau1 [V]
            tau2: End of discharge [s]
            U2: Voltage at tau2 [V]
        """
        T, w, Tn, U0 = self.T, 2*np.pi/self.T, self.T/self.nphase, self.U0

        if C < 0:
            raise ValueError("Capacitance must be positive")

        # find point where value is equal to self.U2_min
        # this will be our boundary value for U2
        tau1_max = np.arccos(self.U2_min / U0)/w
        tau2_min = Tn - tau1_max

        # For constant power: at tau1, charging stops when:
        # w*C*U0*sin(w*tau1) = P/U(tau1) = P/(U0*cos(w*tau1))
        # This gives: w*C*U0^2*sin(w*tau1)*cos(w*tau1) = P
        # Or: w*C*U0^2*sin(2*w*tau1a)/2 = P
        # So: sin(2*w*tau1) = 2*P/(w*C*U0^2)

        sin_arg = 2 * load_param / (w * C * U0**2)
        if sin_arg >= 1.0:
            # Power too high for this capacitance - capacitor fully discharges
            return tau1_max, self.U2_min, tau2_min, self.U2_min

        # Solve for tau1 (start of discharge)
        tau1 = np.arcsin(sin_arg) / (2 * w)
        if tau1 > tau1_max:
            # Power too high for this capacitance - capacitor fully discharges
            return tau1_max, self.U2_min, tau2_min, self.U2_min
        U1 = U0 * np.cos(w * tau1)
        # Check for intersection between discharge curve and self.U2_min
        tau2_u2_min = tau1 + C*(U1**2 - self.U2_min**2)/(2*load_param)
        if tau2_u2_min < tau1_max:
            # We intersect U2_min before the end of first sine
            # check for self intersection with first sine
            # Discharge: U² = U1² - 2*P*(t-tau1)/C
            # First sine: U = U0*cos(w*t)
            def eq1(t):
                ucos = max(U0*np.cos(w*t), 0)
                return U1**2 - 2*load_param*(t - tau1)/C - ucos**2
            # calculate a start value, we cannot start at tau1, because that is a zero of the equation
            tmid = 0.5*(0.5*Tn + tau1)
            while eq1(tmid) < 0:
                tmid = 0.5*(tmid + tau1)
            tau2 = sopt.brentq(eq1, tmid, 0.5*Tn)
            U2 = U0 * np.cos(w * tau2)
            return tau1, U1, tau2, U2
        elif tau2_u2_min < tau2_min:
            # we intersect between the first and second sine
            return tau1, U1, tau2_u2_min, self.U2_min
        else:
            # we intersect the second sine
            # Discharge: U² = U1² - 2*P*(t-tau1)/C
            # Second sine: U = U0*cos(w*t)
            def eq2(t):
                ucos = max(0, U0*np.cos(w*(t - Tn)))
                return U1**2 - 2*load_param*(t - tau1)/C - ucos**2
            tau2 = sopt.brentq(eq2, Tn/2, Tn)
            tau2 = max(tau2, tau2_min)
            U2 = U0*np.cos(w*(tau2 - Tn))
            return tau1, U1, tau2, U2

    def build_discharge_waveform(self, C: float, load_param: float, npoints: int):
        """
        Args:
            C: capacitance
            load_param: current
            npoints: number of points in waveform

        Returns:
            tt: timestamps
            sinewave: voltage waveform of sine at times tt
            capwave: voltage waveform of capacitor at times tt
            tau1: Start of discharge [s]
            U1: Voltage at tau1 [V]
            tau2: End of discharge [s]
            U2: Voltage at tau2 [V]
        """

        T, w, Tn, U0 = self.T, 2 * np.pi / self.T, self.T / self.nphase, self.U0

        dt = Tn / npoints
        tt = np.arange(npoints) * dt

        tau1, U1, tau2, U2 = self.solve_U1(C, load_param)
        # plot the two sines but not the parts that are < 0
        sine1, sine2 = U0 * np.cos(w * tt), U0 * np.cos(w * (tt - Tn))
        sinewave = np.clip(np.maximum(sine1, sine2), 0, None)

        # plot the discharge lines
        capwave = np.clip(sinewave, self.U2_min, None)
        i1, i2 = round(tau1 / dt), round(tau2 / dt)
        capwave[i1:i2] = U1**2 - 2*load_param*(tt[i1:i2] - tau1) / C
        capwave[i1:i2] = np.sqrt(np.clip(capwave[i1:i2], self.U2_min**2, None))
        return tt, sinewave, capwave, tau1, U1, tau2, U2

    @classmethod
    def get_load_param_name(cls) -> str:
        return "Load Power"

    @classmethod
    def get_load_param_unit(cls) -> str:
        return "W"

    @classmethod
    def get_load_param_description(cls) -> str:
        return "Constant power discharge (Watts)"

    @classmethod
    def get_model_name(cls) -> str:
        return "Constant Power"

    @classmethod
    def get_model_description(cls) -> str:
        return "For switching power supplies, buck/boost converters, phone chargers, modern regulated electronics"

    @classmethod
    def get_discharge_type(cls) -> str:
        return "Non-linear (square root)"

def main():
    RectifierModelPower. plot_one(50, 1, 50, 30e-6, 100)
    RectifierModelPower.plot_one(50, 1, 50, 280e-6, 100)
    RectifierModelPower.plot_one(50, 1, 50, 300e-6, 100)
    RectifierModelPower.plot_one(50, 1, 50, 400e-6, 100)
    RectifierModelPower.plot_one(50, 1, 50, 600e-6, 100)
    RectifierModelPower.plot_one(50, 1, 50, 1200e-6, 100)
    RectifierModelPower.plot_one(50, 1, 50, 2000e-6, 100)
    RectifierModelPower.plot_ripple(50, 1, 50, 3000e-6, 100)

    RectifierModelPower.plot_one(50, 2, 50, 30e-6, 100)
    RectifierModelPower.plot_one(50, 2, 50, 300e-6, 100)
    RectifierModelPower.plot_one(50, 2, 50, 600e-6, 100)
    RectifierModelPower.plot_ripple(50, 2, 50, 3000e-6, 100)
    #
    RectifierModelPower.plot_one(50, 6, 50, 30e-6, 100)
    RectifierModelPower.plot_one(50, 6, 50, 300e-6, 100)
    RectifierModelPower.plot_one(50, 6, 50, 600e-6, 100)
    RectifierModelPower.plot_ripple(50, 6, 50, 3000e-6, 100)


    import matplotlib.pyplot as plt
    plt.show()

if __name__=="__main__":
    main()

