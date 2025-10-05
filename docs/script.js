document.addEventListener('DOMContentLoaded', () => {
    // --- DOM ELEMENT SELECTOR ---
    const getEl = (id) => document.getElementById(id);

    // Main Panels & States
    const analysisForm = getEl('analysis-form');
    const resultsPanel = document.querySelector('.results-panel');
    const idleState = getEl('idle-state');
    const loadingState = getEl('loading-state');
    const resultsState = getEl('results-state');

    // Input Elements
    const tabs = document.querySelectorAll('.tab-button');
    const tabContents = document.querySelectorAll('.tab-content');
    const newsTextInput = getEl('news-text');
    const newsUrlInput = getEl('news-url');
    const fileInput = getEl('news-file-input');
    const fileNameEl = getEl('file-name');

    // Buttons
    const analyzeBtn = getEl('analyze-btn');
    const clearBtn = getEl('clear-btn');

    // Results Display Elements
    const trustScoreEl = getEl('trust-score');
    const scoreCircle = document.querySelector('.score-circle');
    const summaryVerdictEl = getEl('summary-verdict');
    const summaryTextEl = getEl('summary-text');
    const metricsContainer = document.querySelector('.metrics-breakdown');
    const highlightedTextOutput = getEl('highlighted-text-output');

    let originalTextForHighlighting = '';

    // --- STATE MANAGEMENT ---
    const showState = (stateToShow) => {
        [idleState, loadingState, resultsState].forEach(state => {
            state.classList.remove('active');
        });
        stateToShow.classList.add('active');
    };

    // --- EVENT LISTENERS ---
    // Tab Switching
    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            tabs.forEach(t => t.classList.remove('active'));
            tabContents.forEach(c => c.classList.remove('active'));
            tab.classList.add('active');
            getEl(tab.dataset.tab).classList.add('active');
        });
    });

    // File Input
    // fileInput.addEventListener('change', () => {
    //     fileNameEl.textContent = fileInput.files.length > 0 ? fileInput.files[0].name : 'Click to upload a .txt file';
    // });

    // Form Submission
    analysisForm.addEventListener('submit', handleAnalysisRequest);

    // Clear Button
    clearBtn.addEventListener('click', resetUI);

    // --- CORE FUNCTIONS ---
    async function handleAnalysisRequest(e) {
        e.preventDefault();
        let inputData = '';
        const activeTab = document.querySelector('.tab-button.active').dataset.tab;

        if (activeTab === 'text-input') {
            inputData = newsTextInput.value.trim();
        } else if (activeTab === 'url-input') {
            inputData = newsUrlInput.value.trim();
        } else if (activeTab === 'file-input' && fileInput.files.length > 0) {
            inputData = await fileInput.files[0].text();
        }

        if (!inputData) {
            alert('Please provide input (text, URL, or file) before analyzing.');
            return;
        }

        originalTextForHighlighting = inputData; // Store original text

        // Transition to loading state
        showState(loadingState);
        analyzeBtn.disabled = true;
        analyzeBtn.querySelector('.btn-icon').innerHTML = '<i class="fas fa-spinner fa-spin"></i>';
        analyzeBtn.querySelector('.btn-text').textContent = 'Analyzing...';

        // --- ACTUAL API CALL TO FASTAPI BACKEND ---
        try {
            // API Base URL - adjust based on environment
            const API_BASE_URL = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
                ? 'http://localhost:3000'
                : 'https://fake-news-detection-system-893j.onrender.com';

            console.log('Making API call to:', `${API_BASE_URL}/predict`);

            const response = await fetch(`${API_BASE_URL}/predict`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    text: inputData,
                    title: "" // Empty title for now
                })
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const result = await response.json();
            console.log('API Response:', result);

            // Check if there's an error in the response
            if (result.error) {
                throw new Error(result.error);
            }

            // Convert your API response to frontend format
            const analysisData = convertApiResponseToFrontend(result);
            displayResults(analysisData);

        } catch (error) {
            console.error('API call failed:', error);

            // Show user-friendly error message
            let errorMessage = 'Analysis failed. ';
            if (error.message.includes('Failed to fetch')) {
                errorMessage += 'Cannot connect to server. Please check if the backend is running.';
            } else if (error.message.includes('Models not loaded')) {
                errorMessage += 'ML models are not loaded. Please wait a moment and try again.';
            } else {
                errorMessage += 'Please try again. Error: ' + error.message;
            }

            alert(errorMessage);
            showState(idleState);
        } finally {
            analyzeBtn.disabled = false;
            analyzeBtn.querySelector('.btn-icon').innerHTML = '<i class="fas fa-search-plus"></i>';
            analyzeBtn.querySelector('.btn-text').textContent = 'Analyze';
        }
    }

    function displayResults(data) {
        // 1. Set summary and overall score
        summaryVerdictEl.textContent = data.summary.verdict;
        summaryVerdictEl.style.color = data.summary.color;
        summaryTextEl.textContent = data.summary.recommendation;

        // Animate score count-up
        animateCountUp(trustScoreEl, data.trustScore, 1500);
        scoreCircle.style.setProperty('--progress-value', data.trustScore);
        scoreCircle.style.background = `conic-gradient(${data.summary.color} calc(var(--progress-value) * 1%), var(--panel-border) 0)`;

        // 2. Build and inject detailed metrics
        let metricsHTML = '<h3>Detailed Metrics</h3>';
        data.metrics.forEach(metric => {
            metricsHTML += `
                <div class="metric-item">
                    <span class="metric-label">${metric.name}</span>
                    <div class="metric-bar-wrapper">
                        <div class="metric-bar" style="width: 0%; background-color: ${metric.color};"></div>
                    </div>
                    <span class="metric-score" style="color: ${metric.color};">${metric.score}</span>
                </div>
            `;
        });
        metricsContainer.innerHTML = metricsHTML;

        // Animate metric bars
        setTimeout(() => {
            document.querySelectorAll('.metric-bar').forEach((bar, index) => {
                bar.style.width = `${data.metrics[index].score}%`;
            });
        }, 100);

        // 3. Display highlighted text
        highlightedTextOutput.innerHTML = data.highlightedText;

        // 4. Transition to results state
        showState(resultsState);
        analyzeBtn.disabled = false;
        analyzeBtn.querySelector('.btn-icon').innerHTML = '<i class="fas fa-search-plus"></i>';
        analyzeBtn.querySelector('.btn-text').textContent = 'Analyze';
    }

    function resetUI() {
        analysisForm.reset();
        fileNameEl.textContent = 'Click to upload a .txt file';
        showState(idleState);
    }

    // --- API RESPONSE CONVERTER ---
    function convertApiResponseToFrontend(apiResult) {
        // Your FastAPI returns: {"label": "Real News", "confidence": 0.85, "model_accuracy": 0.9966, "model_metrics": {...}}
        const confidence = apiResult.confidence || 0.5;
        const isReal = apiResult.label === "Real News";

        // Use actual model accuracy instead of fake trust score
        const modelAccuracy = (apiResult.model_accuracy || 0.9966) * 100; // Convert to percentage

        let summary;
        if (isReal && confidence > 0.80) {
            summary = {
                verdict: isReal ? 'Highly Credible' : 'High Risk of Misinformation',
                color: isReal ? 'var(--success)' : 'var(--danger)',
                recommendation: isReal ? 'Content appears factual and well-sourced. Safe to share.' : 'Strong indicators of false content. Advised not to share.'
            };
        } else if (confidence > 0.50) {
            summary = {
                verdict: 'Moderately Reliable',
                color: 'var(--warning)',
                recommendation: 'Some uncertainty detected. Verify from additional sources.'
            };
        } else {
            summary = {
                verdict: 'High Risk of Misinformation',
                color: 'var(--danger)',
                recommendation: 'Low confidence prediction. Strongly advised to verify from reliable sources.'
            };
        }

        // Use your actual model metrics instead of fake ones
        const modelMetrics = apiResult.model_metrics || {};

        return {
            trustScore: Math.round(modelAccuracy), // This will show actual model accuracy
            summary,
            metrics: [
                {
                    name: 'Model Accuracy',
                    score: Math.round(modelAccuracy * 100) / 100,
                    color: 'var(--success)'
                },
                {
                    name: 'Precision',
                    score: Math.round((modelMetrics.precision || 0.9962) * 10000) / 100,
                    color: 'var(--primary-accent)'
                },
                {
                    name: 'Recall',
                    score: Math.round((modelMetrics.recall || 0.9975) * 10000) / 100,
                    color: 'var(--secondary-accent)'
                },
                {
                    name: 'F1-Score',
                    score: Math.round((modelMetrics.f1_score || 0.9969) * 10000) / 100,
                    color: 'var(--info)'
                }
            ],
            highlightedText: originalTextForHighlighting
        };
    }    // --- UTILITY & SIMULATION ---
    function animateCountUp(element, target, duration) {
        let start = 0;
        const stepTime = 20; // ms
        const steps = duration / stepTime;
        const increment = target / steps;

        const timer = setInterval(() => {
            start += increment;
            if (start >= target) {
                clearInterval(timer);
                start = target;
            }
            element.textContent = Math.floor(start);
        }, stepTime);
    }

    function generateFakeAnalysis() {
        // This is a mock API response. A real backend would provide this.
        const score = Math.floor(Math.random() * 100);
        let summary, metrics, highlightedText;

        if (score > 75) {
            summary = { verdict: 'Highly Credible', color: 'var(--success)', recommendation: 'Content appears factual and well-sourced. Safe to share.' };
        } else if (score > 50) {
            summary = { verdict: 'Generally Reliable', color: 'var(--primary-accent)', recommendation: 'Minor inconsistencies or potential bias detected. Proceed with awareness.' };
        } else if (score > 25) {
            summary = { verdict: 'Questionable', color: 'var(--warning)', recommendation: 'Contains unverified claims or emotionally charged language. Cross-reference with trusted sources.' };
        } else {
            summary = { verdict: 'High Risk of Misinformation', color: 'var(--danger)', recommendation: 'Strong indicators of false or manipulative content. Advised not to share.' };
        }

        metrics = [
            { name: 'Source Credibility', score: Math.floor(Math.random() * 100), color: 'var(--primary-accent)' },
            { name: 'Sentiment Neutrality', score: Math.floor(Math.random() * 100), color: 'var(--secondary-accent)' },
            { name: 'Claim Verification', score: Math.floor(Math.random() * 100), color: 'var(--success)' },
            { name: 'Bias & Propaganda', score: Math.floor(Math.random() * 100), color: 'var(--warning)' }
        ];

        // Simulate highlighting
        const words = originalTextForHighlighting.split(' ');
        highlightedText = words.map(word => {
            const r = Math.random();
            if (r < 0.05) return `<span class="highlight highlight-claim" title="Unverified Claim">${word}</span>`;
            if (r < 0.1) return `<span class="highlight highlight-sensational" title="Sensationalist Language">${word}</span>`;
            return word;
        }).join(' ');

        return { trustScore: score, summary, metrics, highlightedText };
    }
});