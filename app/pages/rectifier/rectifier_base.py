"""
Base class for rectifier discharge models.
Provides common interface for constant current and constant power models.
"""
from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np
import scipy


class RectifierModelBase(ABC):
    """Base class for rectifier discharge models"""

    def __init__(self, T: float, nphase: int, U0: float):
        """
        Initialize rectifier model

        Parameters:
        T: Period time of sine [s]
        nphase: Number of phases if it is a half wave rectifier.
                Number of phases times two if it is a full wave rectifier.
        U0: Rectified sine amplitude [V]
        """
        if U0 <= 0. or T <= 0. or nphase <= 0:
            raise ValueError("Parameters out of range")

        self.T = T
        self.nphase = nphase
        self.U0 = U0
        w, pin = 2*np.pi/T, np.pi/nphase
        self.U2_min = U0 * np.cos(pin) if self.nphase > 2 else 0.0

    @abstractmethod
    def solve_U1(self, C: float, load_param: float) -> tuple[float, float, float, float]:
        """
        Calculate discharge curve parameters for given capacitance

        Parameters:
        C: Capacitor value [F]
        load_param: Load parameter (current [A] or power [W] depending on model)

        Returns:
        tuple: (tau1, U1, tau2, U2)
        """
        pass

    @abstractmethod
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
        load_param: Load current [A] or Load power [W]

        Returns:
        Iripple: RMS ripple current [A]
        """
        pass

    @classmethod
    @abstractmethod
    def get_load_param_name(cls) -> str:
        """Return the name of the load parameter"""
        pass

    @classmethod
    @abstractmethod
    def get_load_param_unit(cls) -> str:
        """Return the unit of the load parameter"""
        pass

    @classmethod
    @abstractmethod
    def get_load_param_description(cls) -> str:
        """Return a description of the load parameter"""
        pass

    @classmethod
    @abstractmethod
    def get_model_name(cls) -> str:
        """Return the human-readable name of the model"""
        pass

    @classmethod
    @abstractmethod
    def get_model_description(cls) -> str:
        """Return a description of when to use this model"""
        pass

    @classmethod
    @abstractmethod
    def get_discharge_type(cls) -> str:
        """Return the type of discharge (linear, non-linear, etc.)"""
        pass

    @abstractmethod
    def build_discharge_waveform(self, C: float, load_param: float, npoints: int) -> np.ndarray:
        """
        Build the complete voltage waveform for one period

        Parameters:
        C: Capacitance [F]
        load_param: Load parameter (current or power)
        npoints: number of points in the waveform

        Returns:
            tt: timestamps
            sinewave: voltage waveform of sine at times tt
            capwave: voltage waveform of capacitor at times tt
            tau1: Start of discharge [s]
            U1: Voltage at tau1 [V]
            tau2: End of discharge [s]
            U2: Voltage at tau2 [V]

        """
        pass

    def build_U1_ripple(self, Cmax: float, load_param: float, npoints: int):
        cc = np.linspace(0, Cmax, npoints)[1:]
        uu1 = np.zeros_like(cc, dtype=float)
        irr = np.zeros_like(cc, dtype=float)
        for k, c in enumerate(cc):
            tau1, U1, tau2, U2 = self.solve_U1(c, load_param)
            uu1[k] = U1
            irr[k] = self.ripple_current(tau1, tau2, c, load_param)
        return cc, uu1, irr

    def plot_discharge_lines(self, C: float, load_param: float, npoints: int = 5000):
        """
        Plot discharge curves for constant current load

        Uses member variables tau1, tau2 which must be set by calling
        solve_U1() or solve_C() first.

        Parameters:
        C: Capacitance [F]
        Iload: Load current [A]

        Raises:
        RuntimeError: If intersection times not set
        """
        import matplotlib.pyplot as plt
        from numlib import numlegend

        # Build waveform using only C, Iload, and tt (member variables already set)
        U0 = self.U0
        tt, sinewave, capwave, tau1, U1, tau2, U2 = self.build_discharge_waveform(C, load_param, npoints)

        # Plot the two sines but not the parts that are < 0
        fig, ax = plt.subplots()
        ax.plot(tt, sinewave, color="black", lw=3, label="Sine")

        # Plot the discharge waveform
        ax.plot(tt, capwave, color="red", lw=2, label="Output voltage")
        ax.set_xlabel("time [s]")
        ax.set_ylabel("voltage [V]")
        numlegend.gridsetup(ax)

        Iripple = self.ripple_current(tau1, tau2, C, load_param)
        fig.suptitle(f"U0={self.U0:.0f} [V] {U2=:.0f} [V] nphase={self.nphase} C={1e6 * C:.0f} [μF] Iripple={Iripple=:.1f} [A]")
        return fig, ax, tt, sinewave, capwave, tau1, U1, tau2, U2


    def plot_U1_ripple(self, Cmax, load_param, npoints: int = 1000):
        import matplotlib.pyplot as plt
        from numlib import numlegend

        cc, uu1, irr = self.build_U1_ripple(Cmax, load_param, npoints)

        fig, ax = plt.subplots()
        ax.plot(1e6*cc, uu1, color="red", label="Minimum output voltage")
        ax2 = ax.twinx()
        ax2.plot(1e6*cc, irr, color="blue", label="Ripple current")
        ax.set_xlabel("Capacitance [μF]")
        ax.set_ylabel("Voltage [V]")
        ax2.set_ylabel("Current [A]")
        numlegend.legendsetup((ax, ax2))
        numlegend.gridsetup(ax)
        fig.suptitle(f"U0={self.U0:.0f} [V] nphase={self.nphase}")

    def compute_fft_spectrum(self, C: float, load_param: float, n_samples: int = 4096) -> dict:
        """
        Compute FFT spectrum of the discharge waveform

        Parameters:
        C: Capacitance [F]
        load_param: Load parameter (current or power)
        n_samples: Number of samples for FFT (power of 2 recommended)

        Returns:
        dict: {
            'frequencies': frequency array [Hz],
            'magnitudes': magnitude array [V],
            'phases': phase array [rad],
            'dc_component': DC voltage [V],
            'thd': Total Harmonic Distortion [%],
            'harmonics': list of tuples (harmonic_number, frequency, magnitude, magnitude_percent)
        }
        """

        # Use derived class method to build the waveform
        # (uses member variables set by solve_U1)
        tt, sinewave, capwave, tau1, U1, tau2, U2 = self.build_discharge_waveform(C, load_param, n_samples)
        dt = tt[1] - tt[0]
        # Perform FFT
        fft_result = scipy.fft.rfft(capwave)
        frequencies = scipy.fft.rfftfreq(n_samples, dt)
        magnitudes = np.abs(fft_result) * 2.0 / n_samples  # Normalize
        phases = np.angle(fft_result)

        # DC component
        dc_component = magnitudes[0] / 2.0  # DC is not doubled

        # Find fundamental frequency and harmonics
        fundamental_freq = self.nphase / self.T  # Ripple frequency
        freq_resolution = frequencies[1] - frequencies[0]

        # Identify harmonics (multiples of fundamental)
        harmonics = []
        harmonic_magnitudes = []

        for k in range(1, 21):  # First 20 harmonics
            harmonic_freq = k * fundamental_freq
            # Find closest frequency bin
            idx = int(round(harmonic_freq / freq_resolution))
            if idx < len(frequencies):
                mag = magnitudes[idx]
                harmonics.append((k, frequencies[idx], mag, 100.0 * mag / dc_component if dc_component > 0 else 0))
                if k > 1:  # Exclude fundamental for THD calculation
                    harmonic_magnitudes.append(mag)

        # Calculate THD (Total Harmonic Distortion)
        if len(harmonics) > 0:
            fundamental_mag = harmonics[0][2]
            if fundamental_mag > 0:
                thd = 100.0 * np.sqrt(np.sum(np.array(harmonic_magnitudes)**2)) / fundamental_mag
            else:
                thd = 0.0
        else:
            thd = 0.0

        return {
            'frequencies': frequencies,
            'magnitudes': magnitudes,
            'phases': phases,
            'dc_component': dc_component,
            'thd': thd,
            'harmonics': harmonics,
            'waveform': capwave,
            'time': tt
        }

    def get_harmonic_summary(self, C: float, load_param: float) -> str:
        """
        Get a human-readable summary of the harmonic content

        Parameters:
        C: Capacitance [F]
        load_param: Load parameter

        Returns:
        str: Formatted summary of harmonics
        """
        fft_data = self.compute_fft_spectrum(C, load_param)

        summary = f"FFT Analysis Summary\n"
        summary += f"{'='*60}\n"
        summary += f"DC Component: {fft_data['dc_component']:.2f} V\n"
        summary += f"Total Harmonic Distortion (THD): {fft_data['thd']:.2f}%\n"
        summary += f"\nHarmonics (first 10):\n"
        summary += f"{'Harmonic':<10} {'Frequency':<12} {'Magnitude':<12} {'% of DC':<10}\n"
        summary += f"{'-'*60}\n"

        for i, (k, freq, mag, pct) in enumerate(fft_data['harmonics'][:10]):
            summary += f"{k:<10d} {freq:<12.2f} {mag:<12.4f} {pct:<10.2f}\n"

        return summary

    @classmethod
    def plot_one(cls, f0, nphase, U0, C1, load_param):
        rm = cls(1/f0, nphase, U0)
        rm.plot_discharge_lines(C1, load_param)
        print("\n  FFT Analysis:")
        try:
            fft_data2 = rm.compute_fft_spectrum(C1, load_param)
            print(f"    DC Component: {fft_data2['dc_component']:.2f} V")
            print(f"    THD: {fft_data2['thd']:.2f} %")
            print(f"    Ripple frequency: {fft_data2['harmonics'][0][1]:.1f} Hz")
            print(f"    First 5 harmonics:")
            for k, freq, mag, pct in fft_data2['harmonics'][:5]:
                print(f"      H{k}: {freq:.1f} Hz, {mag:.4f} V ({pct:.2f}% of DC)")
        except Exception as e:
            print(f"    FFT failed: {e}")

    @classmethod
    def plot_ripple(cls, f0, nphase, U0, Cmax, load_param):
        rm = cls(1/f0, nphase, U0)
        rm.plot_U1_ripple(Cmax, load_param)
