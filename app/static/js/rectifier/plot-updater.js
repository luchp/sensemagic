/**
 * Rectifier Calculator - Plot Updater
 * Handles real-time plot updates via AJAX when sliders change
 */

function updatePlots() {
    // Show loading toast and change cursor
    document.getElementById('loading-toast').style.display = 'block';
    document.body.classList.add('updating');

    // Get current slider values
    const loadParam = document.getElementById('load_slider').value;
    const capacitance = document.getElementById('cap_slider').value;

    // Get parameters from data attributes (set by Jinja template)
    const plotsContainer = document.getElementById('plots-container');
    const modelType = plotsContainer.dataset.modelType;
    const frequency = plotsContainer.dataset.frequency;
    const voltage = plotsContainer.dataset.voltage;
    const nphase = plotsContainer.dataset.nphase;
    const maxLoad = plotsContainer.dataset.maxLoad;
    const cmax = plotsContainer.dataset.cmax;
    const plotType = plotsContainer.dataset.plotType;

    // Prepare form data
    const formData = new FormData();
    formData.append('model_type', modelType);
    formData.append('frequency', frequency);
    formData.append('voltage', voltage);
    formData.append('nphase', nphase);
    formData.append('load_param', loadParam);
    formData.append('max_load', maxLoad);
    formData.append('capacitance', capacitance);
    formData.append('cmax', cmax);
    formData.append('plot_type', plotType);

    // Send AJAX request
    fetch('/app_rectifier/plot_data', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            alert('Error: ' + data.error);
            document.getElementById('loading-toast').style.display = 'none';
            document.body.classList.remove('updating');
            return;
        }

        // Update discharge plot if present
        if (data.discharge) {
            const dischargeDiv = document.querySelector('#plots-container > div:first-child');
            if (dischargeDiv) {
                Plotly.react(dischargeDiv, data.discharge.data, data.discharge.layout);
            }

            // Update results summary
            if (data.summary) {
                const summaryDiv = document.querySelector('#plots-container > div:nth-child(2)');
                if (summaryDiv) {
                    summaryDiv.innerHTML = `
                        <h3 style="margin-top: 0;">📊 Results Summary - ${data.summary.model_name} Model</h3>
                        <ul style="line-height: 1.8;">
                            <li><strong>Peak Voltage (Ux):</strong> ${data.summary.Ux} V</li>
                            <li><strong>Minimum Voltage (U1):</strong> ${data.summary.U1} V</li>
                            <li><strong>Voltage Ripple:</strong> ${data.summary.ripple} V</li>
                            <li><strong>Ripple Current:</strong> ${data.summary.Iripple} A</li>
                            <li><strong>Discharge Start (τ1):</strong> ${data.summary.tau1} ms</li>
                            <li><strong>Discharge End (τ2):</strong> ${data.summary.tau2} ms</li>
                            <li><strong>Discharge Duration:</strong> ${data.summary.duration} ms</li>
                            <li><strong>Discharge Type:</strong> ${data.summary.discharge_type}</li>
                        </ul>
                    `;
                }
            }
        }

        // Update U1C plot if present
        if (data.u1c) {
            // Find U1C plot - it's after the summary div
            const plotDivs = document.querySelectorAll('#plots-container .plotly-graph-div');
            if (plotDivs.length > 1) {
                Plotly.react(plotDivs[1], data.u1c.data, data.u1c.layout);
            }
        }

        // Update FFT plot if present
        if (data.fft) {
            const plotDivs = document.querySelectorAll('#plots-container .plotly-graph-div');
            if (plotDivs.length > 2) {
                Plotly.react(plotDivs[2], data.fft.data, data.fft.layout);
            }
        }

        // Hide loading toast and restore cursor
        document.getElementById('loading-toast').style.display = 'none';
        document.body.classList.remove('updating');
    })
    .catch(error => {
        alert('Network error: ' + error);
        document.getElementById('loading-toast').style.display = 'none';
        document.body.classList.remove('updating');
    });
}

// Update slider display values
function updateSliderDisplay(sliderId, displayId, unit = '') {
    const slider = document.getElementById(sliderId);
    const display = document.getElementById(displayId);
    if (slider && display) {
        display.textContent = slider.value + unit;
    }
}

