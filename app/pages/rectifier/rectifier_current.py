import numpy as np
import scipy.optimize as sopt

from .rectifier_base import RectifierModelBase

class RectifierModelCurrent(RectifierModelBase):
    def __init__(self, T, nphase, U0):
        """
        Parameters:
        T: Period time of sine [s]
        nphase: Number of phases if it is a half wave rectifier.
                Number of phases times two if it is a full wave rectifier.
        U0: Rectified sine amplitude [V]
        """
        super().__init__(T, nphase, U0)

    def ripple_current(self, tau1: float, tau2: float, C: float, load_param: float) -> float:
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

        # Discharging phase: tau1 to tau2 (constant current = Iload)
        # Contribution: Iload² * (tau2 - tau1)
        discharging = load_param**2 * (tau2 - tau1)

        # Charging phase 2: tau2 to Tn
        # In case of 1 phase rectifier, the capacitor can be empty before start of next sine
        tau2_eff = max(tau2, Tn - 0.25 * T)
        charging2 = cosint(Tn - tau2_eff)

        # RMS current over one period
        Iripple = np.sqrt((charging1 + discharging + charging2) / Tn)

        return Iripple

    def solve_U1(self, C: float, load_param: float) -> tuple[float, float, float, float]:
        """
        Find tau1 and tau2 for constant current discharge

        tau1: Time when capacitor stops charging (discharge starts)
        tau2: Time when capacitor starts charging again (discharge ends)

        Between tau1 and tau2: Linear discharge at constant current Iload

        Args:
            C: Capacitance [F]
            load_param: Load current Iload [A]

        Returns:
            tau1: Start of discharge [s]
            U1: Voltage at tau1 [V]
            tau2: End of discharge [s]
            U2: Voltage at tau2 [V]
        """
        Iload = load_param
        T, w, Tn, U0 = self.T, 2*np.pi/self.T, self.T/self.nphase, self.U0
        I_min = C * w * U0 * np.sin(np.pi/self.nphase) if self.nphase > 1 else C * w * U0
        if C < 0:
            raise ValueError("Parameters out of range")
        elif Iload > I_min:
            # capacitor does not contribute to output current
            tau1, U1, tau2, U2 = 0.5 * Tn, self.U2_min, 0.5 * Tn, self.U2_min
        else:
            # Solve for U2
            tau1 = np.arcsin(Iload / (U0 * w * C)) / w
            U1 = U0 * np.cos(w * tau1)
            # special case for nphase = 1, then the capacitor can be empty before start of next sine
            tau_empty = U1*C/Iload + tau1
            if tau_empty <= Tn - T/4:
                tau2, U2 = tau_empty, 0.
            else:
                def eq(U2):
                    tau2 = Tn - np.arccos(U2 / U0) / w
                    return U2 - U1 + (tau2 - tau1) * Iload / C
                U2 = sopt.brentq(eq, 0, U1)
                tau2 = Tn - np.arccos(U2 / U0) / w

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
        capwave = sinewave.copy()
        i1, i2 = round(tau1 / dt), round(tau2 / dt)
        capwave[i1:i2] = U1 + (U2 - U1) * (tt[i1:i2] - tau1) / (tau2 - tau1)
        return tt, sinewave, capwave, tau1, U1, tau2, U2


    @classmethod
    def get_load_param_name(cls) -> str:
        return "Load Current"

    @classmethod
    def get_load_param_unit(cls) -> str:
        return "A"

    @classmethod
    def get_load_param_description(cls) -> str:
        return "Constant current discharge (Amperes)"

    @classmethod
    def get_model_name(cls) -> str:
        return "Constant Current"

    @classmethod
    def get_model_description(cls) -> str:
        return "For resistive loads, linear regulators (LM7805, etc.), battery chargers, LEDs with resistors"

    @classmethod
    def get_discharge_type(cls) -> str:
        return "Linear"


def main():
    RectifierModelCurrent. plot_one(50, 1, 50, 30e-6, 1)
    RectifierModelCurrent.plot_one(50, 1, 50, 150e-6, 1)
    RectifierModelCurrent.plot_one(50, 1, 50, 300e-6, 1)
    RectifierModelCurrent.plot_ripple(50, 1, 50, 500e-6, 1)
    #
    RectifierModelCurrent.plot_one(50, 2, 50, 30e-6, 1)
    RectifierModelCurrent.plot_one(50, 2, 50, 300e-6, 1)
    RectifierModelCurrent.plot_ripple(50, 2, 50, 500e-6, 1)
    #
    RectifierModelCurrent.plot_one(50, 6, 50, 30e-6, 1)
    RectifierModelCurrent.plot_one(50, 6, 50, 300e-6, 1)
    RectifierModelCurrent.plot_ripple(50, 6, 50, 500e-6, 1)


    import matplotlib.pyplot as plt
    plt.show()
        
if __name__=="__main__":
    main()
