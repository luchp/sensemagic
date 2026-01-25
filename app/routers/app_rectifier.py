from pathlib import Path
from fastapi import Form, Query
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi import Request
import plotly.graph_objs as go
import plotly.offline as pyo
from plotly.subplots import make_subplots
import numpy as np

from fastapi import APIRouter
from pathlib import Path

from pages.rectifier.rectifier_base import RectifierModelBase
from pages.rectifier.rectifier_current import RectifierModelCurrent
from pages.rectifier.rectifier_power import RectifierModelPower
from .markdown_renderer import MarkdownRenderer

prefix = Path(__file__).stem
router = APIRouter(prefix=f"/{prefix}", tags=[prefix])

# Application description for WordPress sync
router.description = "Design and analyze rectifier circuits with capacitor smoothing for both constant current and constant power loads"

# Set up Jinja2 templates
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

# Set up markdown renderer for documentation routes
md_renderer = MarkdownRenderer(
    base_path=Path(__file__).parent.parent / "pages" / "rectifier",
    route_prefix="/app_rectifier"
)


def create_discharge_plot(rm: RectifierModelBase, tau1, Ux, tau2, U1, C, Iripple, U_discharge=None):
    """Create a Plotly figure for discharge lines (works with both current and power models)"""
    T, w, Tn, U0 = rm.T, 2*np.pi/rm.T, rm.T/rm.nphase, rm.U0
    npoints = 10000

    dt = Tn / npoints
    tt = np.arange(npoints) * dt

    # Calculate the two sines
    sine1 = U0 * np.cos(w * tt)
    sine2 = U0 * np.cos(w * (tt - Tn))

    # Calculate discharge curve
    tau3 = max(tau2, Tn - 0.25*T)
    i1, i2, i3 = round(tau1 / dt), round(tau2 / dt), round(tau3 / dt)

    # Create figure
    fig = go.Figure()

    # Plot first sine (positive parts only)
    mask1 = sine1 > 0
    fig.add_trace(go.Scatter(
        x=tt[mask1] * 1000,  # Convert to ms
        y=sine1[mask1],
        mode='lines',
        name='First sine',
        line=dict(color='black', width=3)
    ))

    # Plot second sine (positive parts only)
    mask2 = sine2 > 0
    fig.add_trace(go.Scatter(
        x=tt[mask2] * 1000,
        y=sine2[mask2],
        mode='lines',
        name='Second sine',
        line=dict(color='darkblue', width=3)
    ))

    # Plot discharge curve (output voltage)
    output_t = []
    output_v = []

    # Charging phase
    output_t.extend(tt[:i1] * 1000)
    output_v.extend(sine1[:i1])

    # Discharge phase - linear for current, curved for power
    t_discharge = tt[i1:i2]
    if U_discharge is not None and len(U_discharge) > 1:
        # Non-linear discharge (power model)
        U_curve = np.interp(t_discharge, np.linspace(tau1, tau2, len(U_discharge)), U_discharge)
        output_t.extend(t_discharge * 1000)
        output_v.extend(U_curve)

        # Check if discharge intersected first sine (tau2 is early in the cycle)
        # If so, follow the first sine from tau2 until it goes negative or second sine takes over
        if tau2 < Tn * 0.6:  # Intersection with first sine likely occurred
            # Continue with first sine from tau2 onwards
            t_sine_continue = tt[tt > tau2]
            t_sine_continue = t_sine_continue[t_sine_continue < Tn]

            if len(t_sine_continue) > 0:
                sine1_continue = U0 * np.cos(w * t_sine_continue)
                # Only include positive part of sine
                positive_mask = sine1_continue > 0
                if np.any(positive_mask):
                    output_t.extend(t_sine_continue[positive_mask] * 1000)
                    output_v.extend(sine1_continue[positive_mask])
                    # Update i2 to skip the zero phase
                    last_t = t_sine_continue[positive_mask][-1]
                    i2 = np.searchsorted(tt, last_t)
    else:
        # Linear discharge (current model)
        output_t.extend(t_discharge * 1000)
        output_v.extend(Ux + (U1 - Ux) * (t_discharge - tau1) / (tau2 - tau1))

    # Zero phase (if any)
    if i3 > i2:
        output_t.extend(tt[i2:i3] * 1000)
        output_v.extend(np.zeros(i3-i2))

    # Next charging phase
    output_t.extend(tt[i3:] * 1000)
    output_v.extend(sine2[i3:])

    fig.add_trace(go.Scatter(
        x=output_t,
        y=output_v,
        mode='lines',
        name=f'Output voltage ({rm.get_discharge_type()})',
        line=dict(color='green', width=2)
    ))

    fig.update_layout(
        title=f"{rm.get_model_name()} Rectifier Discharge Curve<br><sub>U0={U0:.0f}V, U1={U1:.0f}V, nphase={rm.nphase}, C={C*1e6:.1f}μF, Iripple={Iripple:.2f}A</sub>",
        xaxis_title="Time [ms]",
        yaxis_title="Voltage [V]",
        hovermode='x unified',
        template='plotly_white',
        height=500
    )

    return fig


def create_u1c_plot(rm: RectifierModelBase, Cmax, load_param):
    """Create a Plotly figure for U1 vs C plot (works with both models)"""
    # Use the base class method to build U1 and ripple data
    cc, U1_values, Iripple_values = rm.build_U1_ripple(Cmax, load_param, npoints=500)

    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add U1 trace
    fig.add_trace(
        go.Scatter(
            x=(cc * 1e6).tolist(),  # Convert to μF and to list
            y=U1_values,
            name="Minimum voltage (U1)",
            line=dict(color='red', width=2)
        ),
        secondary_y=False
    )

    # Add Iripple trace
    fig.add_trace(
        go.Scatter(
            x=(cc * 1e6).tolist(),
            y=Iripple_values,
            name="Ripple current",
            line=dict(color='blue', width=2)
        ),
        secondary_y=True
    )

    fig.update_xaxes(title_text="Capacitance [μF]")
    fig.update_yaxes(title_text="Voltage [V]", secondary_y=False)
    fig.update_yaxes(title_text="Current [A]", secondary_y=True)

    fig.update_layout(
        title=f"{rm.get_model_name()} Rectifier Performance vs Capacitance<br><sub>U0={rm.U0:.0f}V, nphase={rm.nphase}, {rm.get_load_param_name()}={load_param:.3f}{rm.get_load_param_unit()}</sub>",
        hovermode='x unified',
        template='plotly_white',
        height=500
    )

    return fig


def create_fft_plot(rm: RectifierModelBase, C, load_param):
    """Create a Plotly figure for FFT spectrum analysis"""

    # Get FFT data
    fft_data = rm.compute_fft_spectrum(C, load_param)

    # Create subplots: waveform and spectrum
    fig = make_subplots(
        rows=1, cols=1,
        subplot_titles=('FFT Spectrum (Frequency Domain)', ),
    )

    # Plot 2: Frequency spectrum (first 20 harmonics only for clarity)
    harmonic_freqs = [h[1] for h in fft_data['harmonics'][:20]]
    harmonic_mags = [h[2] for h in fft_data['harmonics'][:20]]
    harmonic_labels = [f"H{h[0]}" for h in fft_data['harmonics'][:20]]

    fig.add_trace(
        go.Bar(
            x=harmonic_freqs,
            y=harmonic_mags,
            name='Harmonics',
            marker=dict(color='blue'),
            text=harmonic_labels,
            textposition='outside',
            hovertemplate='<b>%{text}</b><br>Frequency: %{x:.1f} Hz<br>Magnitude: %{y:.4f} V<extra></extra>'
        ),
        row=1, col=1
    )

    # Update axes
    fig.update_xaxes(title_text="Frequency [Hz]", row=1, col=1)
    fig.update_yaxes(title_text="Magnitude [V]", row=1, col=1)

    # Overall layout
    fundamental_freq = fft_data['harmonics'][0][1] if len(fft_data['harmonics']) > 0 else 0
    fig.update_layout(
        title=f"{rm.get_model_name()} FFT Analysis<br><sub>C={C*1e6:.1f}μF, {rm.get_load_param_name()}={load_param:.3f}{rm.get_load_param_unit()}, DC={fft_data['dc_component']:.2f}V, THD={fft_data['thd']:.2f}%, f_ripple={fundamental_freq:.1f}Hz</sub>",
        showlegend=False,
        template='plotly_white',
        height=600
    )

    return fig


@router.get("/", response_class=HTMLResponse)
def form(
    request: Request,
    model_type: str = "current",
    frequency: float = 50,
    voltage: float = 325,
    nphase: int = 2,
    load_param: float = 0.1,
    max_load: float = 1.0,
    capacitance: float = 100,
    cmax: float = 200,
    plot_type: str = "both",
    standalone: bool = Query(True, description="Standalone page with logo (true) or iframe mode (false)")
):
    """Display the parameter input form with optional pre-filled values"""
    return templates.TemplateResponse("rectifier/form.html", {
        "request": request,
        "model_type": model_type,
        "frequency": frequency,
        "voltage": voltage,
        "nphase": nphase,
        "load_param": load_param,
        "max_load": max_load,
        "capacitance": capacitance,
        "cmax": cmax,
        "plot_type": plot_type,
        "standalone": standalone
    })


@router.post("/plot", response_class=HTMLResponse)
def plot(
    request: Request,
    model_type: str = Form("current"),
    frequency: float = Form(...),
    voltage: float = Form(...),
    nphase: int = Form(...),
    load_param: float = Form(...),
    max_load: float = Form(1.0),
    capacitance: float = Form(...),
    cmax: float = Form(...),
    plot_type: str = Form(...),
    standalone: str = Form("false")
):
    """Generate plots based on user parameters"""

    # Convert standalone string to boolean
    standalone_bool = standalone.lower() == 'true'

    try:
        # Create rectifier model based on selected type
        T = 1.0 / frequency
        if model_type == "current":
            rm = RectifierModelCurrent(T, nphase, voltage)
        elif model_type == "power":
            rm = RectifierModelPower(T, nphase, voltage)
        else:
            # Fallback to current model
            rm = RectifierModelCurrent(T, nphase, voltage)

        # Convert capacitance from μF to F
        C = capacitance * 1e-6
        Cmax = cmax * 1e-6

        # Generate plots based on selection
        plots_html = ""

        if plot_type in ["discharge", "both"]:
            # Build the waveform using the model's method (internally calls solve_U1)
            npoints = 10000
            tt, sinewave, capwave, tau1, U1, tau2, U2 = rm.build_discharge_waveform(C, load_param, npoints)

            # Calculate ripple current from the discharge parameters
            Iripple = rm.ripple_current(tau1, tau2, C, load_param)

            # Extract other parameters for display
            U0 = rm.U0
            Ux = np.max(sinewave)

            # Create Plotly figure
            fig_discharge = go.Figure()

            # Plot the sine envelope
            fig_discharge.add_trace(go.Scatter(
                x=(tt * 1000).tolist(),
                y=sinewave.tolist(),
                mode='lines',
                name='Sine envelope',
                line=dict(color='black', width=3)
            ))

            # Plot output voltage
            fig_discharge.add_trace(go.Scatter(
                x=(tt * 1000).tolist(),
                y=capwave.tolist(),
                mode='lines',
                name=f'Output voltage ({rm.get_discharge_type()})',
                line=dict(color='green', width=2)
            ))

            fig_discharge.update_layout(
                title=f"{rm.get_model_name()}: U0={U0:.0f}V, U1={U1:.2f}V, U2={U2:.2f}V, C={C*1e6:.1f}μF, Iripple={Iripple:.2f}A",
                xaxis_title="Time [ms]",
                yaxis_title="Voltage [V]",
                hovermode='x unified',
                template='plotly_white',
                height=500
            )

            plots_html += pyo.plot(fig_discharge, include_plotlyjs=True, output_type="div")

            # Add results summary
            plots_html += f"""
            <div style="background-color: #f0f8ff; padding: 15px; margin: 20px 0; border-radius: 5px; border-left: 4px solid #2196F3;">
                <h3 style="margin-top: 0;">📊 Results Summary - {rm.get_model_name()} Model</h3>
                <ul style="line-height: 1.8;">
                    <li><strong>Peak Voltage (Ux):</strong> {Ux:.2f} V</li>
                    <li><strong>Minimum Voltage (U1):</strong> {U1:.2f} V</li>
                    <li><strong>End Voltage (U2):</strong> {U2:.2f} V</li>
                    <li><strong>Voltage Ripple:</strong> {Ux - U2:.2f} V</li>
                    <li><strong>Ripple Current:</strong> {Iripple:.3f} A</li>
                    <li><strong>Discharge Start (τ1):</strong> {tau1*1000:.3f} ms</li>
                    <li><strong>Discharge End (τ2):</strong> {tau2*1000:.3f} ms</li>
                    <li><strong>Discharge Duration:</strong> {(tau2-tau1)*1000:.3f} ms</li>
                    <li><strong>Discharge Type:</strong> {rm.get_discharge_type()}</li>
                </ul>
            </div>
            """

        if plot_type in ["u1c", "both"]:
            # Create U1 vs C plot
            fig_u1c = create_u1c_plot(rm, Cmax, load_param)
            plots_html += pyo.plot(fig_u1c, include_plotlyjs=False, output_type="div")

        if plot_type in ["fft", "both"]:
            # Create FFT plot - use actual capacitance C, not Cmax!
            fig_fft = create_fft_plot(rm, C, load_param)
            plots_html += pyo.plot(fig_fft, include_plotlyjs=False, output_type="div")

        # Return template with plots
        return templates.TemplateResponse("rectifier/results.html", {
            "request": request,
            "model_type": model_type,
            "model_name": rm.get_model_name(),
            "frequency": frequency,
            "voltage": voltage,
            "nphase": nphase,
            "load_param": load_param,
            "load_param_name": rm.get_load_param_name(),
            "load_param_unit": rm.get_load_param_unit(),
            "max_load": max_load,
            "capacitance": capacitance,
            "cmax": cmax,
            "plot_type": plot_type,
            "plots_html": plots_html,
            "standalone": standalone_bool
        })

    except Exception as e:
        return templates.TemplateResponse("rectifier/error.html", {
            "request": request,
            "error_message": str(e),
            "standalone": standalone_bool
        })


@router.post("/plot_data")
def plot_data(
    model_type: str = Form("current"),
    frequency: float = Form(...),
    voltage: float = Form(...),
    nphase: int = Form(...),
    load_param: float = Form(...),
    max_load: float = Form(1.0),
    capacitance: float = Form(...),
    cmax: float = Form(...),
    plot_type: str = Form(...)
):
    """Return plot data as JSON for in-place updates"""
    try:
        # Create rectifier model based on selected type
        T = 1.0 / frequency
        if model_type == "current":
            rm = RectifierModelCurrent(T, nphase, voltage)
        elif model_type == "power":
            rm = RectifierModelPower(T, nphase, voltage)
        else:
            rm = RectifierModelCurrent(T, nphase, voltage)

        # Convert capacitance from μF to F
        C = capacitance * 1e-6
        Cmax = cmax * 1e-6

        response_data = {}

        if plot_type in ["discharge", "both"]:
            # Build the waveform using the model's method (internally calls solve_U1)
            npoints = 10000
            tt, sinewave, capwave, tau1, U1, tau2, U2 = rm.build_discharge_waveform(C, load_param, npoints)

            # Calculate ripple current from the discharge parameters
            Iripple = rm.ripple_current(tau1, tau2, C, load_param)

            # Extract other parameters for display
            U0 = rm.U0
            Ux = np.max(sinewave)

            # Create Plotly figure
            fig_discharge = go.Figure()

            # Plot the sine envelope
            fig_discharge.add_trace(go.Scatter(
                x=(tt * 1000).tolist(),
                y=sinewave.tolist(),
                mode='lines',
                name='Sine envelope',
                line=dict(color='black', width=3)
            ))

            # Plot output voltage
            fig_discharge.add_trace(go.Scatter(
                x=(tt * 1000).tolist(),
                y=capwave.tolist(),
                mode='lines',
                name=f'Output voltage ({rm.get_discharge_type()})',
                line=dict(color='green', width=2)
            ))

            fig_discharge.update_layout(
                title=f"{rm.get_model_name()}: U0={U0:.0f}V, U1={U1:.2f}V, U2={U2:.2f}V, C={C*1e6:.1f}μF, Iripple={Iripple:.2f}A",
                xaxis_title="Time [ms]",
                yaxis_title="Voltage [V]",
                hovermode='x unified',
                template='plotly_white',
                height=500
            )

            # Store figure data and layout
            response_data['discharge'] = {
                'data': fig_discharge.to_dict()['data'],
                'layout': fig_discharge.to_dict()['layout']
            }

            # Results summary
            response_data['summary'] = {
                'model_name': rm.get_model_name(),
                'Ux': f"{Ux:.2f}",
                'U1': f"{U1:.2f}",
                'U2': f"{U2:.2f}",
                'ripple': f"{Ux - U2:.2f}",
                'Iripple': f"{Iripple:.3f}",
                'tau1': f"{tau1*1000:.3f}" if tau1 is not None else "N/A",
                'tau2': f"{tau2*1000:.3f}" if tau2 is not None else "N/A",
                'duration': f"{(tau2-tau1)*1000:.3f}" if (tau1 is not None and tau2 is not None) else "N/A",
                'discharge_type': rm.get_discharge_type()
            }

        if plot_type in ["u1c", "both"]:
            # Create U1 vs C plot
            fig_u1c = create_u1c_plot(rm, Cmax, load_param)

            # Store figure data and layout
            response_data['u1c'] = {
                'data': fig_u1c.to_dict()['data'],
                'layout': fig_u1c.to_dict()['layout']
            }

        if plot_type in ["fft", "both"]:
            # Create FFT plot - use actual capacitance C, not Cmax!
            fig_fft = create_fft_plot(rm, C, load_param)

            # Store figure data and layout
            response_data['fft'] = {
                'data': fig_fft.to_dict()['data'],
                'layout': fig_fft.to_dict()['layout']
            }

        return response_data

    except Exception as e:
        return {"error": str(e)}


# ============================================================================
# Documentation Routes - Markdown pages with LaTeX support
# ============================================================================

@router.get("/math", response_class=HTMLResponse)
def rectifier_mathematics(standalone: bool = Query(True, description="Standalone page with MathJax (true) or iframe mode (false)")):
    """Serve the Rectifier Mathematics documentation as HTML with LaTeX support"""
    return md_renderer.render_to_page(
        filename="RECTIFIER_MATH_CLEAN.md",
        title="Rectifier Circuit Mathematics",
        standalone=standalone,
        include_toc=True
    )
