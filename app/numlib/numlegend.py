# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 13:24:49 2016

@author: Luc
"""
import time
import matplotlib.legend as mlegend


def adjust_xscale(ax, extra10, include0):
    """Adjust xscale to include the extra 10% or the origin"""
    xmin, xmax = ax.get_xlim()
    if "log" in ax.get_xscale():
        if extra10:
            s = (xmax / xmin) ** 0.05
            xmin /= s
            xmax *= s
    else:
        if include0:
            xmin = min(0.0, xmin)
            xmax = max(0.0, xmax)
        if extra10:
            extra = abs(xmax - xmin) * 0.05
            xmin -= extra
            xmax += extra
    ax.set_xlim(xmin, xmax)


def adjust_yscale(ax, extra10, include0):
    """Adjust yscale to include the extra 10% or the origin"""
    ymin, ymax = ax.get_ylim()
    if "log" in ax.get_yscale():
        if extra10:
            s = (ymax / ymin) ** 0.05
            ymin /= s
            ymax *= s
    else:
        if include0:
            ymin = min(0.0, ymin)
            ymax = max(0.0, ymax)
        if extra10:
            extra = abs(ymax - ymin) * 0.05
            ymin -= extra
            ymax += extra
    ax.set_ylim(ymin, ymax)


def legendsetup(ax, labels=None, loc=0, fontsize="small"):
    """Collect labels from axis and display in a dragable legend"""
    # make it into a list of tuples
    if hasattr(ax, "tolist"):
        ax = ax.tolist()  # matplotlib return a numpy array from subplots
    if isinstance(ax, (list, tuple)):
        if labels is None:
            labels = [None] * len(ax)
        axlb = list(zip(ax, labels))
    else:
        axlb = [(ax, labels)]
    # collect all handles and labels
    all_handles, all_labels = [], []
    for x, lbl in axlb:
        if lbl is None:
            handles, lbl = x.get_legend_handles_labels()
        else:
            handles = [handle for handle, label in zip(mlegend._get_legend_handles([x]), labels)]
        all_handles.extend(handles)
        all_labels.extend(lbl)
    # plot it
    legend = axlb[-1][0].legend(all_handles, all_labels, loc=loc, fontsize=fontsize)
    if legend:
        legend.set_draggable(True)
    return legend, all_handles


class LegendHandler(object):
    """Collect labels from axis and display in a dragable legend with visibility toggles"""

    def __init__(self, canvas):
        self.lines = {}
        self.picker_time = time.time()
        self.canvas = canvas
        self.canvas.mpl_connect("pick_event", self.onpick)

    @staticmethod
    def _find_dependent_artists(dependencies, artist, orig_line):
        if isinstance(artist, (list, tuple)):
            for a in artist:
                LegendHandler._find_dependent_artists(dependencies, a, orig_line)
        if hasattr(artist, "get_children"):
            for a in artist.get_children():
                LegendHandler._find_dependent_artists(dependencies, a, orig_line)
        if getattr(artist, "legendhandler_dependency", None) is orig_line:
            dependencies.add(artist)

    def setup(self, ax, labels=None, loc=0, fontsize="small"):
        legend, all_handles = legendsetup(ax, labels, loc, fontsize)
        if legend:
            for legend_line, orig_line in zip(legend.get_lines(), all_handles):
                legend_line.set_picker(True)  # 5 pts tolerance
                legend_line.set_pickradius(5)
                dependencies = set()
                self._find_dependent_artists(dependencies, ax, orig_line)
                self.lines[legend_line] = (orig_line, dependencies)

    def onpick(self, event):
        # on the pick event, find the orig line corresponding to the
        # legend proxy line, and toggle the visibility
        legend_line = event.artist
        if legend_line in self.lines and time.time() > self.picker_time:
            orig_line, dependencies = self.lines[legend_line]

            vis = not orig_line.get_visible()
            orig_line.set_visible(vis)
            for d in dependencies:
                d.set_visible(vis)
            # Change the alpha on the line in the legend so we can see what lines
            # have been toggled
            legend_line.set_alpha(1.0 if vis else 0.2)
            self.picker_time = time.time() + 0.1
            self.canvas.draw()

    def update_legend_visibility(self):
        for legend_line, (orig_line, dependencies) in self.lines.items():
            vis = orig_line.get_visible()
            legend_line.set_alpha(1.0 if vis else 0.2)
            for d in dependencies:
                d.set_visible(vis)


def gridsetup(ax):
    """Display a mayor and minor grid with subdued colors"""
    if hasattr(ax, "tolist"):
        ax = ax.tolist()  # matplotlib return a numpy array from subplots
    axis = ax if isinstance(ax, (list, tuple)) else [ax]
    for a in axis:
        a.grid(visible=True, which="major", color="0.6", linestyle="-")
        a.grid(visible=True, which="minor", color="0.7", linestyle="-")
        a.minorticks_on()


class RestorePlot:
    def __init__(self, fig, legendhandler, mpltoolbar):
        self.fig = fig
        self.legendhandler = legendhandler
        self.mpltoolbar = mpltoolbar
        self.line_info = {}  # visible
        self.axes_info = []  # (zoomed, xlabel, ylabel, xlim, ylim)

    def save_state(self):
        self.line_info = {}
        self.axes_info = []
        for n, ax in enumerate(self.fig.get_axes()):
            zoom_active = len(self.mpltoolbar._nav_stack) > 0
            self.axes_info.append((zoom_active, ax.get_xlabel(), ax.get_ylabel(), ax.get_xlim(), ax.get_ylim()))
            for m, ln in enumerate(ax.get_lines()):
                if hasattr(ln, "comp_addr"):
                    addr = ln.comp_addr
                    if addr in self.line_info:
                        raise ValueError(f"Duplicated ADDR {addr}:  line {m} of axis {n} label '{ln.get_label()}'")
                    self.line_info[addr] = ln.get_visible()
                elif not hasattr(ln, "legendhandler_dependency"):
                    if ln.get_label() in self.line_info:
                        print(f"WARNING Duplicated label: line {m} of axis {n} label '{ln.get_label()}'")
                    else:
                        print(f"WARNING No ADDR, using label: line {m} of axis {n} label '{ln.get_label()}'")
                        self.line_info[ln.get_label()] = ln.get_visible()

    def restore_state(self, restore_zoom=True):
        def almost_equal(a, b):
            return abs(a - b) < 1e-3 or abs(a - b) / (0.5 * (abs(a) + abs(b))) < 1e-3

        #
        def lim_almost_equal(lim1, lim2):
            return almost_equal(lim1[0], lim2[0]) and almost_equal(lim1[1], lim2[1])

        #
        must_update, update_info = False, []
        for k, ax in enumerate(self.fig.get_axes()):
            if k < len(self.axes_info):
                zoom_active, xlabel, ylabel, xlim, ylim = self.axes_info[k]
                update_x = zoom_active and ax.get_xlabel() == xlabel and not lim_almost_equal(ax.get_xlim(), xlim)
                update_y = zoom_active and ax.get_ylabel() == ylabel and not lim_almost_equal(ax.get_ylim(), ylim)
                update_info.append((update_x, xlim, update_y, ylim))
                if update_x or update_y:
                    must_update = True
                for ln in ax.get_lines():
                    addr = ln.comp_addr if hasattr(ln, "comp_addr") else ln.get_label()
                    if addr in self.line_info:
                        ln.set_visible(self.line_info[addr])
        #
        if must_update and restore_zoom:
            self.mpltoolbar.push_current()
            for ax, (update_x, limx, update_y, limy) in zip(self.fig.get_axes(), update_info):
                if update_x:
                    ax.set_xlim(limx)
                if update_y:
                    ax.set_ylim(limy)
            #
        self.legendhandler.update_legend_visibility()


class SignalInfo:
    def __init___(self, symbol, minval, maxval, category, units):
        self.symbol = symbol
        self.minval = minval
        self.maxval = maxval
        self.category = category
        self.units = units


class PlotScaler:
    def __init__(self):
        self.signals = {}
        self.category = {}
        self.unit = {}

    def add(self, symbol, minval, maxval, category, units):
        pass
        # self.signals[symbol] = SignalInfo(symbol, minval, maxval, category, units)
        # self.category.setdefault(category, []).append(symbol)
        # self.original_units.setdefault(units, []).append(symbol)

    def calc_scaling(self, scaletype):
        pass
        # step one, let each signal do its own prescaling.

    def unit_scale(self, symbol):
        return 1.0

    def plot_scale(self, symbol):
        return 1.0

    def plot_offset(self, symbol):
        return 1.0

    def auto_scale(self, symbol):
        return 1.0

    @property
    def plot_limits(self):
        return 1.0
