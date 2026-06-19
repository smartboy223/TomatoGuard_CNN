/* ---------------------------------------------------------------
   app.js  -  client-side logic for the main page (index.html).
   What it does, in plain English:
     1. Lets the user pick a tomato image (or drag one in).
     2. Sends it to the local Python server (POST /predict).
     3. Shows the result: Fresh or Rotten + confidence percentage.
   That is the whole file - about 60 lines.
--------------------------------------------------------------- */

(function () {
    const fileInput      = document.getElementById('fileInput');
    const uploadArea     = document.getElementById('uploadArea');
    const previewImage   = document.getElementById('previewImage');
    const uploadContent  = document.getElementById('uploadContent');
    const analyzeBtn     = document.getElementById('analyzeBtn');
    const clearBtn       = document.getElementById('clearBtn');
    const errorBox       = document.getElementById('errorBox');
    const resultBox      = document.getElementById('resultBox');

    let currentFile = null;

    // --- Open the picker when the upload box is clicked ----- //
    uploadArea.addEventListener('click', () => fileInput.click());

    // --- Handle file picker / drag and drop ----------------- //
    fileInput.addEventListener('change', (e) => handleFile(e.target.files[0]));

    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.style.background = '#fee2e2';
    });
    uploadArea.addEventListener('dragleave', () => {
        uploadArea.style.background = '';
    });
    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.style.background = '';
        if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
    });

    // --- Buttons -------------------------------------------- //
    analyzeBtn.addEventListener('click', analyze);
    clearBtn.addEventListener('click', clearImage);

    if (new URLSearchParams(window.location.search).get('demo') === 'prediction') {
        loadDemoPrediction();
    }

    function handleFile(file) {
        if (!file) return;
        if (!file.type.startsWith('image/')) {
            showError('Please pick an image file (JPG or PNG).');
            return;
        }
        hideError();
        currentFile = file;
        const reader = new FileReader();
        reader.onload = (ev) => {
            previewImage.src = ev.target.result;
            previewImage.classList.remove('hidden');
            uploadContent.classList.add('hidden');
            uploadArea.classList.add('has-image');
            analyzeBtn.classList.remove('hidden');
            clearBtn.classList.remove('hidden');
            resultBox.innerHTML = '';
        };
        reader.readAsDataURL(file);
    }

    function clearImage() {
        currentFile = null;
        fileInput.value = '';
        previewImage.src = '';
        previewImage.classList.add('hidden');
        uploadContent.classList.remove('hidden');
        uploadArea.classList.remove('has-image');
        analyzeBtn.classList.add('hidden');
        clearBtn.classList.add('hidden');
        resultBox.innerHTML = '';
        hideError();
    }

    async function analyze() {
        if (!currentFile) return;
        analyzeBtn.disabled = true;
        analyzeBtn.innerHTML = 'Analyzing <span class="loading"></span>';
        hideError();
        try {
            const fd = new FormData();
            fd.append('image', currentFile);
            const res = await fetch('/predict', { method: 'POST', body: fd });
            const data = await res.json();
            if (!res.ok) throw new Error(data.error || 'Server error');
            renderResult(data);
        } catch (err) {
            showError(err.message);
        } finally {
            analyzeBtn.disabled = false;
            analyzeBtn.innerHTML = 'Analyze Image';
        }
    }

    async function loadDemoPrediction() {
        try {
            const res = await fetch('/dataset/not_trained/fresh/fresh_001.jpg');
            if (!res.ok) throw new Error('Demo image could not be loaded.');
            const blob = await res.blob();
            const file = new File([blob], 'fresh_001.jpg', { type: blob.type || 'image/jpeg' });
            handleFile(file);
            window.setTimeout(analyze, 250);
        } catch (err) {
            showError(err.message);
        }
    }

    function renderResult(data) {
        const labelUpper = data.label.toUpperCase();
        resultBox.innerHTML = `
            <div class="status-pill ${data.label}">
                ${labelUpper} - ${data.confidence}% confident
            </div>
            <div class="metric-grid">
                <div class="metric">
                    <div class="label">Fresh score</div>
                    <div class="value">${data.fresh_percent}%</div>
                </div>
                <div class="metric">
                    <div class="label">Rotten score</div>
                    <div class="value">${data.rotten_percent}%</div>
                </div>
            </div>
            <p style="margin-top: 14px; font-size: 13px; color: #6b7280;">
                The model outputs a number between 0 and 1.
                Below 0.5 -> Fresh.  Above 0.5 -> Rotten.
            </p>
        `;
    }

    function showError(msg) {
        errorBox.textContent = msg;
        errorBox.classList.remove('hidden');
    }
    function hideError() {
        errorBox.textContent = '';
        errorBox.classList.add('hidden');
    }
})();
