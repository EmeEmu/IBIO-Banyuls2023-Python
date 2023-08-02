import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


class OrthoAxes:
    def __init__(self, fig, coords, interactive=True, closest_threshold=10):

        # Creating figure at the right size and ratios
        size = coords.ptp(axis=1)
        W = size[0] + size[2]
        H = size[1] + size[2]
        if W<H:
            wr = W/H
            hr = 1
        else:
            wr = 1
            hr = H/W
        gs = GridSpec(ncols=2, nrows=2, figure=fig, width_ratios=[wr, 1-wr], height_ratios=[hr, 1-hr])
        subfig = fig.add_subfigure(gs[0,0])

        # creating sub-axes
        gs = GridSpec(
            2, 2, 
            figure=subfig,
            width_ratios=[size[0]/W, size[2]/W], 
            height_ratios=[size[1]/H, size[2]/H], 
            wspace=0, hspace=0
        )
        self.ax_xy = subfig.add_subplot(gs[0], xticks=[], yticks=[])
        self.ax_yz = subfig.add_subplot(gs[1], sharey=self.ax_xy, xticks=[], yticks=[])
        self.ax_xz = subfig.add_subplot(gs[2], sharex=self.ax_xy, xticks=[], yticks=[])
        self._axs = [self.ax_xy, self.ax_yz, self.ax_xz]
        pad = 0.05
        self.ax_xy.set_xlim(coords[0].min()-pad*size[0], coords[0].max()+pad*size[0])
        self.ax_xy.set_ylim(coords[1].min()-pad*size[1], coords[1].max()+pad*size[1])
        self.ax_yz.set_xlim(coords[2].min()-pad*size[2], coords[2].max()+pad*size[2])
        self.ax_xz.set_ylim(coords[2].min()-pad*size[2], coords[2].max()+pad*size[2])

        # setting up interactive cursor
        self._interactive = interactive
        if self._interactive:
            self.closest_threshold = closest_threshold
            self._ax_text = subfig.add_subplot(gs[3], xticks=[], yticks=[])
            _ = self._ax_text.text(
                0.5, 0.95, 
                "Clic to \nmove Cursor", 
                transform=self._ax_text.transAxes, 
                horizontalalignment="center", 
                verticalalignment="top",
                fontweight="bold",
                wrap=True,
            )
            self._text = self._ax_text.text(
                0.5, 0.65, 
                "", 
                transform=self._ax_text.transAxes, 
                horizontalalignment="center",
                verticalalignment="top",
                wrap=True,
            )
            self._cursor = np.array([0,0,0], dtype=np.float_)
            self._sets = []
            self._h_lines, self._v_lines = [], []
            for ax in self._axs:
                self._h_lines.append(ax.axhline(color='k', lw=0.8, ls='--'))
                self._v_lines.append(ax.axvline(color='k', lw=0.8, ls='--'))
            subfig.canvas.mpl_connect('button_press_event', self._on_mouse_clic)

    def _on_mouse_clic(self, event):
        if self.ax_xy.get_navigate_mode() is None:
            x, y = event.xdata, event.ydata
            a = self._find_axnb(event.inaxes)
            if a==0:
                self._cursor[0] = x
                self._cursor[1] = y
            elif a==1:
                self._cursor[1] = y
                self._cursor[2] = x
            elif a==2:
                self._cursor[0] = x
                self._cursor[2] = y
            self._draw_cursor()
            
    def _draw_cursor(self):
        x,y,z = self._cursor
        closest_str = self._find_closest_per_set()
        self._v_lines[0].set_xdata([x])
        self._h_lines[0].set_ydata([y])
        self._v_lines[1].set_xdata([z])
        self._h_lines[1].set_ydata([y])
        self._v_lines[2].set_xdata([x])
        self._h_lines[2].set_ydata([z])
        self._text.set_text(
            f"position\n{np.round(self._cursor, 3)}\n\n" + closest_str
        )
        self.ax_xy.figure.canvas.draw()
        self.ax_xy.figure.canvas.flush_events()
            
    def _find_axnb(self, ax):
        for i,a in enumerate(self._axs):
            if ax==a:
                return i
        return None

    def _find_closest(self, coord):
        dists = np.sqrt(((coord - self._cursor[:,np.newaxis])**2).sum(axis=0))
        i = np.argmin(dists)
        return i, dists[i]
        
    def _find_closest_per_set(self):
        STRING = ""
        for s,set in enumerate(self._sets):
            i,d = self._find_closest(set)
            if d < self.closest_threshold:
                STRING += f"set:{s} - neuron:{i}\n"
        return STRING
    
    def scatter(self, XYZ, save_set=True, edgecolors='none', **kwargs):
        scatt_xy = self.ax_xy.scatter(XYZ[0],XYZ[1], edgecolors=edgecolors, **kwargs)
        scatt_yz = self.ax_yz.scatter(XYZ[2],XYZ[1], edgecolors=edgecolors, **kwargs)
        scatt_xz = self.ax_xz.scatter(XYZ[0],XYZ[2], edgecolors=edgecolors, **kwargs)
        if save_set and self._interactive:
            self._sets.append(XYZ)
        return scatt_xy, scatt_yz, scatt_xz

    @classmethod
    def change_color(cls, scatt, color):
        assert isinstance(scatt, tuple)
        for s in scatt:
            s.set_color(color)

    def plot_contour(self, contours, **kwargs):
        if isinstance(contours[0], np.ndarray):
            contours =  [[[contours[0]],[contours[1]]]]
        elif (
            isinstance(contours, list) and 
            isinstance(contours[0], list) and 
            isinstance(contours[0][0], list) and
            isinstance(contours[0][0][0], np.ndarray)
        ):
            pass
        else:
            raise TypeError("")
        for r,region in enumerate(contours):
            ax = self.ax_xy
            for c, contour in enumerate(region[0]):
                ax.plot(contour[1], contour[0], **kwargs)
            ax = self.ax_yz
            for c, contour in enumerate(region[1]):
                ax.plot(contour[1], contour[0], **kwargs)

