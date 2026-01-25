from pathlib import Path
import numpy as np

from numlib import numplot, numlegend
import matplotlib.pyplot as plt

# Use a font that supports emoji glyphs
plt.rcParams['font.family'] = ['Segoe UI Emoji', 'DejaVu Sans']
plt.rcParams['font.size'] = 14


def open_loop(ss, Kf, Kp, Kd, fderiv, fplant, fs):
    delay = np.exp(-ss/fs)
    sd = ss/(fderiv*2*np.pi)
    sp = ss / (fplant * 2 * np.pi)
    pid = Kf/ss + Kp + Kd*sd/(1+sd)
    plant = 1/(1+sp)
    # add a moving average filter to the controller

    return pid * plant, delay

def plot_simple():
    ff = np.logspace(-1, np.log10(500), 5000)
    ss = 2j*np.pi*ff
    gg, delay = open_loop(ss, 10, 7, 40 , 10,2,1000)
    fig, ax = numplot.plotspek(gg, ff, c='g', label='open loop')
    da = np.angle(delay, deg=True)
    ga = np.angle(gg, deg=True)
    tota = da+ga
    idx = np.argmax(tota<=-120)

    ax[1].semilogx(ff, da, c='b', label='delay')
    ax[1].semilogx(ff, da+ga, c='r', label='total')
    ax[1].axvline(ff[idx], ls='--', c='k')
    ax[1].axhline(-120, ls='--', c='k')
    ax[0].axvline(ff[idx], ls='--', c='k')
    ax[0].axhline(1, ls='--', c='k')
    numlegend.legendsetup(ax[1])

    # Draw phase margin on ax[1]
    # Find crossover frequency (where gain = 1)
    gain_mag = np.abs(gg)
    idx_crossover = np.argmin(np.abs(gain_mag - 1))
    f_crossover = ff[idx_crossover]
    phase_at_crossover = tota[idx_crossover]

    # Draw vertical line at crossover and bracket showing phase margin
    ax[1].axhline(-180, ls=':', c='gray', alpha=0.7)
    ax[1].annotate('', xy=(f_crossover * 0.95 , -180),
                   xytext=(f_crossover* 0.95, phase_at_crossover),
                   arrowprops=dict(arrowstyle='<->', color='red', lw=2))
    ax[1].text(f_crossover * 0.85, (phase_at_crossover - 180) / 2 - 10,
               f'Phase margin 60°',
               ha='right', color='darkred', fontweight='bold')

    # Add mountain annotations above the plot with arrows pointing to the curve
    # Get y-values at specific x positions for arrow targets
    def get_gain_at_freq(freq):
        idx = np.argmin(np.abs(ff - freq))
        return freq, np.abs(gg[idx])

    ypos = 35
    # Peak of Integration - low frequency
    ax[0].annotate('🏔️ Peak of\nIntegration',
                   xy=get_gain_at_freq(0.2), xytext=(0.15, ypos),
                   ha='center', color='green',
                   arrowprops=dict(arrowstyle='->', color='green', lw=1.5))

    # Valley of Proportionality - middle flat region
    ax[0].annotate('🏕️ Valley of\nProportionality',
                   xy=get_gain_at_freq(0.8), xytext=(1.2, ypos),
                   ha='center', color='green',
                   arrowprops=dict(arrowstyle='->', color='green', lw=1.5))

    # Hill of Differentiation - the bump
    ax[0].annotate('⛰️ Hill of\nDifferentiation',
                   xy=get_gain_at_freq(4), xytext=(8, ypos),
                   ha='center', color='green',
                   arrowprops=dict(arrowstyle='->', color='green', lw=1.5))

    # Slope of Stability - the descent
    ax[0].annotate('🎿 Slope of\nStability',
                   xy=get_gain_at_freq(50), xytext=(60, ypos),
                   ha='center', color='green',
                   arrowprops=dict(arrowstyle='->', color='green', lw=1.5))
    return fig, ax
def main():
    fig, ax = plot_simple()
    fp = Path(__file__).parent.joinpath("pid_mountain.svg")
    my_dpi = 96
    fig.set_size_inches(1200/my_dpi, 800/my_dpi)
    #fig.savefig(fp, format="png", transparent=True, dpi=my_dpi)
    fig.savefig(fp, format="svg", transparent=True)
    plt.show()

if __name__ == "__main__":
    main()
