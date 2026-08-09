"""Router for the sealed-box driver-selection page.

Single GET-driven form/results page (all state round-trips through query
parameters, like ``app_mimoshape``'s ``/generate``): the user sets a room /
target-SPL scenario plus independent size windows for a "sub" role (own band
``[f_low, f_split]``) and an "attack" role (``[f_split, f_high]``), each with
its own driver combobox. A combobox starts pre-populated alphabetically;
pressing that role's Search button re-ranks it (``audioshape.ranking``)
and auto-selects the top pick. Calculate evaluates whichever driver is
selected in each combobox and draws SPL / distortion figures.

Follows the sensemagic conventions: ``prefix`` from the filename, a
``router`` variable, ``router.description`` for WordPress sync. Plotly
figure-building lives here (not in ``pages/``), mirroring
``app_rectifier.py``'s convention -- the audioshape package itself stays
plot-free (matplotlib lives in its own ``plots`` module, unused by the web
app).

The Calculate results also offer a "download as VituixCAD project" link: a
zip (data URL, like ``app_mimoshape_file``'s downloads) containing the
``.vxp`` project plus a driver-database TSV snippet for the selected
driver(s). VituixCAD's project file has no field for Thiele/Small
parameters -- those live only in its separate driver database, matched by
"Manufacturer Model" name -- so the selected driver must already be in the
user's local VituixCAD driver database, or be added there from the bundled
TSV (see ``audioshape.vituixcad``).
"""

from __future__ import annotations

import base64
import io
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import plotly.graph_objs as go
import plotly.offline as pyo
from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from audioshape import physics
from audioshape.driver import Driver
from audioshape.ranking import Evaluation
from audioshape.scenario import Scenario
from audioshape.vituixcad import RoleSelection, driver_database_tsv, project_xml
from pages.driver_criteria.catalog import (
    alphabetical_options,
    band_for_role,
    evaluate_role,
    filter_by_size,
    find_driver,
    load_drivers,
    ranked_options,
)

prefix = Path(__file__).stem  # "app_driver_criteria"
router = APIRouter(prefix=f"/{prefix}", tags=["driver_criteria"])

router.description = (
    "Rank sealed-box bass drivers from first principles (excursion, "
    "Doppler and thermal limits at your room and target SPL) and plot "
    "predicted SPL ceilings and non-correctable distortion"
)

templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))


@dataclass(frozen=True)
class FormState:
    """Everything the page round-trips through GET query parameters."""

    v_room: float = 60.0
    l_max: float = 6.0
    r_listen: float = 3.0
    target_spl: float = 110.0
    distortion_budget_pct: float = 3.0
    qtc: float = 0.55
    f_low: float = 15.0
    f_split: float = 80.0
    f_high: float = 250.0
    burst_shape: float = 2.0
    burst_headroom: float = 4.0
    sub_units: int = 2
    attack_units: int = 1
    sub_size_min: float = 15.0
    sub_size_max: float = 99.0
    attack_size_min: float = 0.0
    attack_size_max: float = 10.0
    sub_driver: str = ""
    attack_driver: str = ""
    sub_mode: str = "alpha"     # "alpha" | "ranked"
    attack_mode: str = "alpha"  # "alpha" | "ranked"
    action: str = ""            # "" | "search_sub" | "search_attack" | "calculate"

    def scenario(self) -> Scenario:
        return Scenario(
            v_room=self.v_room, l_max=self.l_max, r_listen=self.r_listen,
            target_spl=self.target_spl,
            distortion_budget=self.distortion_budget_pct / 100.0,
            qtc=self.qtc, f_low=self.f_low, f_split=self.f_split,
            f_high=self.f_high, burst_shape=self.burst_shape,
            burst_headroom=self.burst_headroom)


def form_state(
    v_room: float = Query(60.0, gt=0),
    l_max: float = Query(6.0, gt=0),
    r_listen: float = Query(3.0, gt=0),
    target_spl: float = Query(110.0),
    distortion_budget_pct: float = Query(3.0, gt=0, lt=100),
    qtc: float = Query(0.55, gt=0),
    f_low: float = Query(15.0, gt=0),
    f_split: float = Query(80.0, gt=0),
    f_high: float = Query(250.0, gt=0),
    burst_shape: float = Query(2.0, gt=0),
    burst_headroom: float = Query(4.0, gt=0),
    sub_units: int = Query(2, ge=1, le=8),
    attack_units: int = Query(1, ge=1, le=8),
    sub_size_min: float = Query(15.0, ge=0),
    sub_size_max: float = Query(99.0, ge=0),
    attack_size_min: float = Query(0.0, ge=0),
    attack_size_max: float = Query(10.0, ge=0),
    sub_driver: str = Query(""),
    attack_driver: str = Query(""),
    sub_mode: str = Query("alpha"),
    attack_mode: str = Query("alpha"),
    action: str = Query(""),
) -> FormState:
    return FormState(
        v_room=v_room, l_max=l_max, r_listen=r_listen, target_spl=target_spl,
        distortion_budget_pct=distortion_budget_pct, qtc=qtc,
        f_low=f_low, f_split=f_split, f_high=f_high,
        burst_shape=burst_shape, burst_headroom=burst_headroom,
        sub_units=sub_units, attack_units=attack_units,
        sub_size_min=sub_size_min, sub_size_max=sub_size_max,
        attack_size_min=attack_size_min, attack_size_max=attack_size_max,
        sub_driver=sub_driver, attack_driver=attack_driver,
        sub_mode=sub_mode, attack_mode=attack_mode, action=action,
    )


# ----------------------------------------------------------------------
# Plotly figures (this module's own rendering layer; see module docstring)
# ----------------------------------------------------------------------

def _freq_axis(band_low: float, band_high: float, n: int = 400) -> np.ndarray:
    return np.geomspace(band_low * 0.7, band_high, n)


def _room_gain_factor(f: float, sc: Scenario) -> float:
    """Linear pressure gain of the room relative to free half-space radiation
    at the listening distance (>= 1 below f_pz, 1 above)."""
    w = 2.0 * np.pi * f
    v_radiation = (np.sqrt(2.0) * sc.target_pressure * 2.0 * np.pi
                   * sc.r_listen / (physics.RHO0 * w * w))
    return v_radiation / sc.demand_volume(f)


def spl_figure(ev: Evaluation, band_low: float, band_high: float) -> go.Figure:
    """Achievable SPL at the listening position vs frequency, over this
    role's own band: sine and burst (pulse) excursion ceilings, thermal
    ceiling (driver Pmax, with EQ tax below Fc), all including room
    pressure-zone gain; target line and f_pz / Fc / f_x markers."""
    sc, d, boxed = ev.scenario, ev.driver, ev.boxed
    f = _freq_axis(band_low, band_high)

    v_dem = np.array([sc.demand_volume(x) for x in f])
    spl_sine = sc.target_spl + 20.0 * np.log10(boxed.vd_total / v_dem)
    spl_burst = spl_sine - 20.0 * np.log10(sc.burst_shape)

    spl_pb = physics.spl_thermal_ceiling(d.eta0, d.p_max, sc.r_listen) + boxed.spl_gain_db()
    tax = np.array([physics.eq_tax_power(x, 1.0, boxed.wc, d.sigma_m) for x in f])
    room = np.array([20.0 * np.log10(_room_gain_factor(x, sc)) for x in f])
    spl_thermal = spl_pb - 10.0 * np.log10(tax) + room

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=f.tolist(), y=spl_sine.tolist(), mode="lines",
                             name="excursion ceiling, sine", line=dict(width=2)))
    fig.add_trace(go.Scatter(x=f.tolist(), y=spl_burst.tolist(), mode="lines",
                             name=f"excursion ceiling, burst (C={sc.burst_shape:g})",
                             line=dict(width=2, dash="dash")))
    fig.add_trace(go.Scatter(x=f.tolist(), y=spl_thermal.tolist(), mode="lines",
                             name="thermal ceiling (driver P_max, EQ tax)",
                             line=dict(width=2, dash="dashdot")))
    fig.add_hline(y=sc.target_spl, line=dict(color="black", width=1),
                 annotation_text=f"target {sc.target_spl:g} dB",
                 annotation_position="bottom right")

    for x, name in ((sc.f_pz, "f_pz"), (boxed.fc, "F_c"), (ev.f_x, "f_x")):
        if np.isfinite(x) and f[0] <= x <= f[-1]:
            fig.add_vline(x=x, line=dict(color="grey", width=1, dash="dot"),
                         annotation_text=name, annotation_position="top")

    fig.update_layout(
        xaxis_type="log", xaxis_title="frequency [Hz]",
        yaxis_title=f"SPL at {sc.r_listen:g} m [dB]",
        yaxis_range=[sc.target_spl - 25, None],
        title=f"{d.label()} \u2014 {boxed.n_units}x in {boxed.vb*1e3:.0f} L "
              f"(Q<sub>tc</sub>={sc.qtc:g}, F<sub>c</sub>={boxed.fc:.1f} Hz)",
        template="plotly_white", height=430,
        margin=dict(t=60, b=40),
        legend=dict(x=0.99, y=0.01, xanchor="right", yanchor="bottom",
                   bgcolor="rgba(255,255,255,0.7)"),
    )
    return fig


def distortion_figure(ev: Evaluation, band_low: float, band_high: float,
                      doppler_ref: float) -> go.Figure:
    """Predicted non-correctable distortion vs frequency at the target SPL,
    over this role's own band: motor/suspension HD, Doppler IM onto this
    role's own top-of-band reference, box air-spring HD2, and their sum,
    against the distortion budget D*."""
    sc, d, boxed = ev.scenario, ev.driver, ev.boxed
    f = _freq_axis(band_low, band_high)

    v_dem = np.array([sc.demand_volume(x) for x in f])
    xi = v_dem / boxed.vd_total
    hd = np.array([physics.harmonic_distortion(x) for x in xi])
    x1 = np.minimum(xi, 1.0) * d.xmax
    doppler = np.array([physics.doppler_im(doppler_ref, x) for x in x1])
    box = np.array([physics.box_hd2(min(v, boxed.n_units * d.vd) / boxed.n_units,
                                    boxed.vb, d.qts, sc.qtc) for v in v_dem])
    total = hd + doppler + box

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=f.tolist(), y=(100 * hd).tolist(), mode="lines",
                             name="motor/suspension HD", line=dict(width=2)))
    fig.add_trace(go.Scatter(x=f.tolist(), y=(100 * doppler).tolist(), mode="lines",
                             name=f"Doppler IM onto {doppler_ref:g} Hz",
                             line=dict(width=2, dash="dash")))
    fig.add_trace(go.Scatter(x=f.tolist(), y=(100 * box).tolist(), mode="lines",
                             name="box air-spring HD2", line=dict(width=2, dash="dashdot")))
    fig.add_trace(go.Scatter(x=f.tolist(), y=(100 * total).tolist(), mode="lines",
                             name="total", line=dict(width=3, color="black")))
    fig.add_hline(y=100 * sc.distortion_budget, line=dict(color="red", width=1),
                 annotation_text=f"budget D*={100*sc.distortion_budget:g}%",
                 annotation_position="bottom right")
    if band_low <= sc.f_pz <= band_high:
        fig.add_vline(x=sc.f_pz, line=dict(color="grey", width=1, dash="dot"),
                     annotation_text="f_pz", annotation_position="top")

    fig.update_layout(
        xaxis_type="log", yaxis_type="log",
        xaxis_title="frequency [Hz]",
        yaxis_title=f"distortion at {sc.target_spl:g} dB target [%]",
        title=f"{d.label()} \u2014 non-correctable distortion, {boxed.n_units} unit(s)",
        template="plotly_white", height=430,
        margin=dict(t=60, b=40),
        legend=dict(x=0.99, y=0.99, xanchor="right", yanchor="top",
                   bgcolor="rgba(255,255,255,0.7)"),
    )
    return fig


# ----------------------------------------------------------------------
# Route
# ----------------------------------------------------------------------

def _role_summary(ev: Evaluation, role: str, n_units: int) -> dict:
    return {
        "role": role,
        "driver": ev.driver.label(),
        "n_units": n_units,
        "vb_l": ev.boxed.vb * 1e3,
        "fc": ev.boxed.fc,
        "xi_x": ev.xi_x,
        "hd_pct": 100 * ev.hd,
        "doppler_pct": 100 * ev.doppler_im,
        "box_pct": 100 * ev.box_hd2,
        "total_pct": 100 * ev.total_distortion,
        "xi_p": ev.xi_p,
        "feasible": ev.feasible,
        "reasons": "; ".join(ev.reasons),
    }


def _vituixcad_zip_data_url(selections: list[RoleSelection]) -> str:
    """Zip the ``.vxp`` project + driver-database TSV, base64 as a data URL
    (like ``app_mimoshape_file``'s downloads -- no extra route/round-trip)."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("driver_selection.vxp",
                    project_xml(selections).encode("utf-8-sig"))
        zf.writestr("VituixCAD_Drivers_selection.txt",
                    driver_database_tsv([s.evaluation.driver for s in selections]))
    encoded = base64.b64encode(buf.getvalue()).decode()
    return f"data:application/zip;base64,{encoded}"


@router.get("/", response_class=HTMLResponse)
async def index(request: Request, standalone: bool = True,
                state: FormState = Depends(form_state)):
    sc = state.scenario()
    result = load_drivers()
    drivers = result.drivers

    sub_mode = "ranked" if state.action == "search_sub" else state.sub_mode
    attack_mode = "ranked" if state.action == "search_attack" else state.attack_mode
    sub_driver_sel = state.sub_driver
    attack_driver_sel = state.attack_driver

    if sub_mode == "ranked":
        sub_options, sub_evals = ranked_options(
            drivers, sc, "sub", state.sub_units, state.sub_size_min, state.sub_size_max)
        if state.action == "search_sub":
            sub_driver_sel = sub_evals[0].driver.label() if sub_evals else ""
    else:
        sub_pool = filter_by_size(drivers, state.sub_size_min, state.sub_size_max)
        sub_options = alphabetical_options(sub_pool)

    if attack_mode == "ranked":
        attack_options, attack_evals = ranked_options(
            drivers, sc, "attack", state.attack_units,
            state.attack_size_min, state.attack_size_max)
        if state.action == "search_attack":
            attack_driver_sel = attack_evals[0].driver.label() if attack_evals else ""
    else:
        attack_pool = filter_by_size(drivers, state.attack_size_min, state.attack_size_max)
        attack_options = alphabetical_options(attack_pool)

    errors: list[str] = []
    plots_html = ""
    summary_rows: list[dict] = []
    pair_summary: dict | None = None
    vituixcad_download_url: str | None = None

    if state.action == "calculate":
        sub_drv: Driver | None = find_driver(drivers, sub_driver_sel)
        atk_drv: Driver | None = find_driver(drivers, attack_driver_sel)
        if sub_drv is None:
            errors.append("Select a sub driver (or press its Search button) first.")
        if atk_drv is None:
            errors.append("Select an attack driver (or press its Search button) first.")

        ev_sub = ev_attack = None
        vituixcad_selections: list[RoleSelection] = []
        if sub_drv is not None:
            ev_sub = evaluate_role(sub_drv, sc, "sub", state.sub_units)
            band_low, band_high, doppler_ref = band_for_role(sc, "sub")
            fig_spl = spl_figure(ev_sub, band_low, band_high)
            fig_dist = distortion_figure(ev_sub, band_low, band_high, doppler_ref)
            plots_html += (f'<h3>Sub \u2014 {sub_drv.label()}</h3>'
                          + pyo.plot(fig_spl, include_plotlyjs=True, output_type="div")
                          + pyo.plot(fig_dist, include_plotlyjs=False, output_type="div"))
            summary_rows.append(_role_summary(ev_sub, "sub", state.sub_units))
            vituixcad_selections.append(RoleSelection("sub", ev_sub, band_low, band_high))

        if atk_drv is not None:
            ev_attack = evaluate_role(atk_drv, sc, "attack", state.attack_units)
            band_low, band_high, doppler_ref = band_for_role(sc, "attack")
            fig_spl = spl_figure(ev_attack, band_low, band_high)
            fig_dist = distortion_figure(ev_attack, band_low, band_high, doppler_ref)
            plots_html += (f'<h3>Attack \u2014 {atk_drv.label()}</h3>'
                          + pyo.plot(fig_spl, include_plotlyjs=(ev_sub is None), output_type="div")
                          + pyo.plot(fig_dist, include_plotlyjs=False, output_type="div"))
            summary_rows.append(_role_summary(ev_attack, "attack", state.attack_units))
            vituixcad_selections.append(RoleSelection("attack", ev_attack, band_low, band_high))

        if ev_sub is not None and ev_attack is not None:
            pair_summary = {
                "total_pct": 100 * (ev_sub.total_distortion + ev_attack.total_distortion),
                "feasible": ev_sub.feasible and ev_attack.feasible,
            }

        if vituixcad_selections:
            vituixcad_download_url = _vituixcad_zip_data_url(vituixcad_selections)

    context = asdict(state)
    context.update({
        "request": request,
        "standalone": standalone,
        "sub_mode": sub_mode,
        "attack_mode": attack_mode,
        "sub_driver": sub_driver_sel,
        "attack_driver": attack_driver_sel,
        "sub_options": sub_options,
        "attack_options": attack_options,
        "num_drivers": len(drivers),
        "num_skipped": len(result.skipped),
        "f_pz": sc.f_pz,
        "errors": errors,
        "plots_html": plots_html,
        "summary_rows": summary_rows,
        "pair_summary": pair_summary,
        "vituixcad_download_url": vituixcad_download_url,
        "show_results": state.action == "calculate",
    })
    return templates.TemplateResponse(request, "driver_criteria/index.html", context)
