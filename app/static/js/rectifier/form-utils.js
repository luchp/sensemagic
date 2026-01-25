/**
 * Rectifier Calculator - Form Utilities
 * Handles dynamic form updates for model type switching
 */

function updateLoadLabel() {
    const modelType = document.getElementById('model_type').value;
    const loadLabel = document.getElementById('load_label');
    const loadUnit = document.getElementById('load_unit');
    const loadParam = document.getElementById('load_param');
    const maxLoadLabel = document.getElementById('max_load_label');
    const maxLoadUnit = document.getElementById('max_load_unit');
    const maxLoad = document.getElementById('max_load');
    const modelDesc = document.getElementById('model_description');

    if (modelType === 'current') {
        loadLabel.textContent = 'Load Current:';
        loadUnit.textContent = 'A (Amperes)';
        maxLoadLabel.textContent = 'Max Load Current:';
        maxLoadUnit.textContent = 'A (for slider range)';
        loadParam.value = '0.1';
        loadParam.step = '0.001';
        maxLoad.value = '1.0';
        maxLoad.step = '0.01';
        modelDesc.textContent = 'For resistive loads, linear regulators, LEDs with resistors';
    } else {
        loadLabel.textContent = 'Load Power:';
        loadUnit.textContent = 'W (Watts)';
        maxLoadLabel.textContent = 'Max Load Power:';
        maxLoadUnit.textContent = 'W (for slider range)';
        loadParam.value = '10';
        loadParam.step = '0.1';
        maxLoad.value = '100';
        maxLoad.step = '1';
        modelDesc.textContent = 'For switching power supplies, phone chargers, regulated electronics';
    }
}

