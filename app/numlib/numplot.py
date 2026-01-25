# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 13:24:49 2016

@author: Luc
"""
import numpy as np

from . import specutil, numutil, numlegend
import matplotlib.offsetbox as offsetbox
import matplotlib.colors as mcolors
import matplotlib.transforms as mtransforms


def luminance(colors):
    """convert a nx3 matrix with (r,g,b) color tuple to a luminance"""
    c = np.atleast_2d(colors)
    x = ((c + 0.055) / 1.055) ** 2.4
    x[c <= 0.03928] = c[c <= 0.03928] / 12
    return 0.2126 * x[:, 0] + 0.7152 * x[:, 1] + 0.0722 * x[:, 2]


def contrast(l1, l2):
    """convert two luminance matrices to a contrast"""
    con = (l1 + 0.05) / (l2 + 0.05)
    con[con < 1] = 1 / con[con < 1]
    return con


def contrast_color(color):
    """convert color"""
    corners = [(0, 0, 0), (1, 1, 1)]
    c1 = mcolors.to_rgb(color)
    con = contrast(luminance(corners), luminance(c1))
    return corners[np.argmax(con)]


class ZPacker(offsetbox.PackerBase):
    """
    The ZPacker simply stacks its children.
    It automatically adjusts the relative positions of children at draw time.
    """

    def __init__(self, pad=None, width=None, height=None, align="baseline", children=None):
        """
        Parameters
        ----------
        pad : float, optional
            Boundary pad.

        width : float, optional

        height : float, optional
           Width and height of the container box, calculated if
           `None`.

        align : str
           Alignment of boxes.

        Notes
        -----
        *pad* need to given in points and will be scale with
        the renderer dpi, while *width* and *height* need to be in
        pixels.
        """
        super().__init__(pad, 0, width, height, align, "fixed", children)

    def _get_bbox_and_child_offsets(self, renderer):
        """
        update offset of children and return the extents of the box
        """
        dpicor = renderer.points_to_pixels(1.0)
        pad = self.pad * dpicor

        whd_list = [c.get_bbox(renderer) for c in self.get_visible_children()]
        if not whd_list:
            return 2 * pad, 2 * pad, pad, pad, []

        whd_list = [c.bounds for c in whd_list]
        hd_list = [(w, xd) for w, h, xd, yd in whd_list]
        (x0, x1), xoffsets = offsetbox._get_aligned_offsets(hd_list, self.width, self.align)
        hd_list = [(h, yd) for w, h, xd, yd in whd_list]
        (y0, y1), yoffsets = offsetbox._get_aligned_offsets(hd_list, self.height, self.align)
        return (mtransforms.Bbox.from_bounds(x0, y0, x1 - x0, y1 - y0).padded(pad), list(zip(xoffsets, yoffsets)))


def align_yaxis(ax1, ax2):
    """Align zeros of the two axes, zooming them out by same ratio"""
    axes = np.array([ax1, ax2])
    extrema = np.array([ax.get_ylim() for ax in axes])
    tops = extrema[:, 1] / (extrema[:, 1] - extrema[:, 0])
    # Ensure that plots (intervals) are ordered bottom to top:
    if tops[0] > tops[1]:
        axes, extrema, tops = [a[::-1] for a in (axes, extrema, tops)]

    # How much would the plot overflow if we kept current zoom levels?
    tot_span = tops[1] + 1 - tops[0]

    extrema[0, 1] = extrema[0, 0] + tot_span * (extrema[0, 1] - extrema[0, 0])
    extrema[1, 0] = extrema[1, 1] + tot_span * (extrema[1, 0] - extrema[1, 1])
    [axes[i].set_ylim(*extrema[i]) for i in range(2)]


def set_3d_axes_equal(ax):
    """Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    """

    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])

    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))

    ax.set_xlim3d([origin[0] - radius, origin[0] + radius])
    ax.set_ylim3d([origin[1] - radius, origin[1] + radius])
    ax.set_zlim3d([origin[2] - radius, origin[2] + radius])


# 1 Hummingbird head is not higher than 1.4,
# 2 Hummingbird head is not at a lower frequency than 3 Hz
# 3 Hummingbird crossover frequency (where power reflection coefficient  r=1) is not lower than 1.1 Hz,
# 4 Hummingbird minimum r, r_min is less than 0.1 in the frequency band from 0.03 to 0.25 Hz.
# 5 Hummingbird hi cutoff frequency (the 2nd frequency where r is halfway 1 and r_min) in not lower than 0.7 Hz,
# 6 Hummingbird low frequency cutoff (the 1st frequency where r is halfway 1 and r_min)  is not higher than 0.01 Hz.
def draw_hum_crit(ax, xlim=None, ylim=None):
    from matplotlib.path import Path
    from matplotlib.patches import PathPatch

    x0, x1 = xlim if xlim else ax.get_xlim()
    y0, y1 = ylim if ylim else ax.get_ylim()
    x0, x1 = min(x0, 0.01), max(x1, 3.0)
    y0, y1 = min(y0, 0.00), max(y1, 1.5)
    x0, x1 = ax.set_xlim(x0, x1)
    y0, y1 = ax.set_ylim(y0, y1)
    verts = [
        (x0, y1),
        (x0, 1.0),
        (0.01, 1.0),
        (0.01, 0.5),
        (0.03, 0.1),
        (0.25, 0.1),
        (0.7, 0.5),
        (1.1, 1.0),
        (1.1, 1.4),
        (x1, 1.4),
        (x1, y1),
        (0, 0),
    ]
    codes = [
        Path.MOVETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.CLOSEPOLY,
    ]
    # Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY,]
    path = Path(verts, codes)
    patch = PathPatch(path, facecolor=(0.9, 0.9, 0.9), edgecolor=(0.6, 0.6, 0.6), lw=1, zorder=0)
    ax.add_patch(patch)
    ax.plot((3, 3), (0, 1.4), color=(0.6, 0.6, 0.6), zorder=1)


def plotnichols(hh, ff, fannon=False, pbase=0.5, unitcircle=True, mindist=False, fax=None, label=None):
    """Work in progress"""
    if fax is None:
        import matplotlib.pyplot as plt

        fax = plt.subplots()
    # data
    hhm, hha = np.abs(hh), np.angle(hh, True)
    hha = specutil.unwrap(hha, 360)
    fax[1].semilogy(hha, hhm, label=label)
    if unitcircle:
        xmin, xmax = fax[1].get_xlim()
        ymin, ymax = fax[1].get_ylim()
        fax[0].semilogy([xmin, xmax], [1, 1], "k-")
        fax[0].plot([-180, -180], [ymin, ymax], "k-")
        if xmax > 90:
            fax[0].plot([180, 180], [ymin, ymax], "k-")

    # anotate powers of 10
    if fannon is True:
        mb = 10**pbase
        f0 = ff[0] if ff[0] > 0 else ff[1]
        f1 = ff[-1]
        fannon = np.arange(int(np.ceil(np.log(f0) / np.log(mb))), int(np.floor(np.log(f1) / np.log(mb))))
    try:
        for p in fannon:
            i = np.argmin(np.abs(ff - mb**p))
            fax[1].annotate(p * pbase, (hha[i], hhm[i]))
    except:
        pass
    # annotate minimum distance
    if mindist:
        sens = np.abs(1.0 / (hh + 1.0))
        si = np.argmin(sens)
        fax[1].scatter(hha[si], hhm[si], marker="*", color="red", s=60)
        fax[1].annotate(f"{sens[si]:.3g}@{ff[si]:.3g}", (hhr[si], hhi[si]), color="red")
    numlegend.gridsetup(fax[1])
    # fax[1].grid(True)
    fax[0].tight_layout()
    return fax


def plotnyquist(
    hh,
    ff,
    color=None,
    fannon=False,
    crossing=None,
    pbase=0.5,
    unitcircle=True,
    mindist=False,
    fax=None,
    label=None,
    boxlimit=10,
    linewidth=1,
):
    # if ff is complex, we assume it is a contour or something like it.
    # and we only annotate the positive frequencies
    contour = np.iscomplexobj(ff)
    if contour:
        # contour is complex frequencies in rad/s
        ff_start = np.argmax(ff.imag > 0)
        ff = ff.imag / (2 * np.pi)
    else:
        # ff is in Hz
        ff_start = np.argmax(ff > 0)
    # create axis
    if fax is None:
        import matplotlib.pyplot as plt

        fax = plt.subplots()
        fax[1].set_aspect("equal", adjustable="datalim")
        # draw unit circle
    if unitcircle:
        phi = np.linspace(-np.pi, np.pi, 1000)
        x, y = np.cos(phi), np.sin(phi)
        (l,) = fax[1].plot(x, y, color="black", linewidth=linewidth, label="Unit cicle")
        sc = fax[1].scatter(-1, 0, marker="*", color="black", s=60)
        sc.legendhandler_dependency = l
    # use crossing data to label stability
    if crossing is not None:
        slabel = f" crossings({crossing[0].size - crossing[1].size})"
        if label is None:
            label = slabel
        else:
            label += slabel
    # draw data
    hhr, hhi = np.real(hh), np.imag(hh)
    if color is None:
        (line,) = fax[1].plot(hhr[ff_start:], hhi[ff_start:], label="+" + label, linewidth=linewidth)
        color = line.get_color()
    else:
        (line,) = fax[1].plot(hhr[ff_start:], hhi[ff_start:], color=color, label="+" + label, linewidth=linewidth)
    fax[1].plot(hhr[:ff_start], hhi[:ff_start], ls="dotted", label="-" + label, color=color, linewidth=linewidth)
    # draw crossing markers
    if crossing is not None:
        fax[1].scatter(crossing[0], np.zeros_like(crossing[0]), marker="^", color=color, s=60)
        fax[1].scatter(crossing[1], np.zeros_like(crossing[1]), marker="v", color=color, s=60)
    # anotate powers of 10
    mb = 10**pbase
    if fannon is True:
        f0 = ff[ff_start]
        f1 = ff[-1]
        fannon = np.arange(int(np.ceil(np.log(f0) / np.log(mb))), 1 + int(np.floor(np.log(f1) / np.log(mb))))
    try:  # assuming the user passed in annotation frequencies in fannon
        for p in fannon:
            i = np.argmin(np.abs(ff - mb**p))
            an = fax[1].annotate(p * pbase, (hhr[i], hhi[i]), color=color)
            an.legendhandler_dependency = line
    except:
        pass
    # annotate minimum distance
    if mindist:
        i = np.argmin(np.abs(hh + 1))
        sc = fax[1].scatter(hhr[i], hhi[i], marker="0", color=color, s=60)
        sc.legendhandler_dependency = line
        an = fax[1].annotate(f"{ff[i]:.3g}", (hhr[i], hhi[i]), color=color)
        an.legendhandler_dependency = line
    # limit size
    if not contour:
        xl, yl = fax[1].get_xlim(), fax[1].get_ylim()
        fax[1].set_xbound(max(xl[0], -boxlimit), min(xl[1], boxlimit))
        fax[1].set_ybound(max(yl[0], -boxlimit), min(yl[1], boxlimit))
    numlegend.gridsetup(fax[1])
    fax[0].tight_layout()
    return fax


def plotzpk(zpk, unitcircle=True, fax=None):
    if fax is None:
        import matplotlib.pyplot as plt

        fax = plt.subplots()
        fax[1].set_aspect("equal", adjustable="datalim")

    fax[1].plot(np.real(zpk[0]), np.imag(zpk[0]), "ob")
    fax[1].plot(np.real(zpk[1]), np.imag(zpk[1]), "xr")
    fax[1].legend(["Zeros", "Poles"], loc=2)
    if unitcircle:
        phi = np.linspace(-np.pi, np.pi, 1000)
        x, y = np.cos(phi), np.sin(phi)
        fax[1].plot(x, y)
    numlegend.gridsetup(fax[1])
    fax[0].tight_layout()
    return fax


def plotspek(spek, df, **kwargs):
    fax = kwargs.pop("fax", None)
    phasedelay = kwargs.pop("phasedelay", False)
    groupdelay = kwargs.pop("groupdelay", False)
    logx = kwargs.pop("logx", True)
    logy = kwargs.pop("logy", True)
    unwrap = kwargs.pop("unwrap", False)
    scaley = kwargs.pop("scaley", True)
    label = kwargs.pop("label", None)
    maxpoints = kwargs.pop("maxpoints", 4000)
    if fax is None:
        import matplotlib.pyplot as plt

        nrplots = 2
        if phasedelay:
            nrplots += 1
        if groupdelay:
            nrplots += 1
        fax = plt.subplots(nrplots, sharex=True)
    ax = fax[1]
    nf = spek.shape[0]
    df = np.atleast_1d(df)
    if len(df) == 1:
        ff = np.arange(nf) * df[0]
    else:
        ff = df
    stride = max(1, int(ff.size / maxpoints))
    ang = np.angle(spek)
    if unwrap:
        ang = np.unwrap(np.angle(spek))
    y = np.abs(spek)
    ax[0].plot(ff[::stride], y[::stride], scaley=scaley, label=label, **kwargs)
    if logx:
        ax[0].set_xscale("log")
    if logy:
        ax[0].set_yscale("log")
    numlegend.gridsetup(ax[0])
    ax[1].set_ylabel("Phase [degrees]")
    ax[1].plot(ff[::stride], (180 / np.pi) * ang[::stride], label=label, **kwargs)
    if logx:
        ax[1].set_xscale("log")
    numlegend.gridsetup(ax[1])
    if phasedelay:
        fd = -np.unwrap(np.angle(spek)) / (2 * np.pi * ff)
        ax[2].plot(ff[::stride], fd[::stride], label=label, **kwargs)
        ax[2].set_ylabel("Phase delay [s]")
        numlegend.gridsetup(ax[2])
    if groupdelay:
        gdi = 3 if phasedelay else 2
        gd = numutil.gdelay(df, spek)
        ax[gdi].plot(ff[::stride], gd[::stride], label=label, **kwargs)
        ax[gdi].set_ylabel("Group delay [s]")
        numlegend.gridsetup(ax[gdi])
    if logx:
        ax[-1].set_xscale("log")
    ax[-1].set_xlabel("Frequency [Hz]")
    fax[0].subplots_adjust(hspace=0.05)
    # fax[0].tight_layout()
    return fax


def plot_table(ax, corner_label, row_labels, col_labels, val_format, vals, fontsize="xx-small"):
    """Display a matrix of values with labels.
    The available space is distributed across the rows and columns"""
    nrows, ncols = len(row_labels) + 1, len(col_labels) + 1
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(-nrows, 0)
    col_pos = np.linspace(0.0, 1.0, ncols) - 0.05
    ax.text(col_pos[0], -0.5, corner_label, ha="center", fontsize=fontsize, fontweight="bold", color="darkred")
    for col, name in enumerate(col_labels):
        ax.text(col_pos[col + 1], -0.5, name, ha="center", fontsize=fontsize, fontweight="bold")
    for row, name in enumerate(row_labels):
        ax.text(col_pos[0], -row - 1.5, name, ha="left", fontsize=fontsize, fontweight="bold")
    for row in range(vals.shape[0]):
        for col in range(vals.shape[1]):
            ax.text(
                col_pos[col + 1], -row - 1.5, val_format[col].format(vals[row, col]), ha="center", fontsize=fontsize
            )
