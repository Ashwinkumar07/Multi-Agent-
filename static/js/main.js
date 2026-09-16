/* =====================================================================
   UNIVERSAL ML WEBAPP - INTERACTIVE FRONTEND ENGINE (JS + CHART.JS)
   ===================================================================== */

document.addEventListener("DOMContentLoaded", function () {
    // -----------------------------------------------------------------
    // GLOBAL UI HANDLERS
    // -----------------------------------------------------------------

    // Switch between Code Tabs
    const tabButtons = document.querySelectorAll(".code-tab");
    tabButtons.forEach(button => {
        button.addEventListener("click", () => {
            const tabsContainer = button.parentElement;
            const viewerContainer = tabsContainer.parentElement;
            const targetId = button.getAttribute("data-target");

            // Deactivate all sibling tabs
            tabsContainer.querySelectorAll(".code-tab").forEach(t => t.classList.remove("active"));
            button.classList.add("active");

            // Hide all sibling content panes
            viewerContainer.querySelectorAll(".code-content").forEach(pane => {
                pane.classList.remove("active");
            });

            // Show active pane
            viewerContainer.querySelector(`#${targetId}`).classList.add("active");
        });
    });

    // -----------------------------------------------------------------
    // DIABETES PREDICTION DASHBOARD
    // -----------------------------------------------------------------
    const diabetesForm = document.getElementById("diabetes-form");
    if (diabetesForm) {
        const fill = document.getElementById("gauge-fill-id");
        
        // Force the stroke-dasharray programmatically so it behaves consistently
        if (fill) {
            fill.style.strokeDasharray = "110 110";
        }

        // SVG Risk Gauge fill function
        function updateRiskGauge(percentage) {
            const label = document.getElementById("gauge-val-id");
            const desc = document.getElementById("gauge-desc-id");
            
            if (!fill || !label || !desc) return;
            
            // Map 0-100% to offset 110 (empty) -> 0 (fully filled)
            const offset = 110 - (percentage * 1.1);
            fill.style.strokeDashoffset = offset;
            label.textContent = `${Math.round(percentage)}%`;

            // Change colors based on risk levels
            const stop2 = document.getElementById("gauge-stop2");
            
            if (percentage < 30) {
                if (stop2) stop2.style.stopColor = "var(--emerald)";
                desc.textContent = "Low Diabetes Risk";
                desc.style.color = "var(--emerald)";
            } else if (percentage < 65) {
                if (stop2) stop2.style.stopColor = "var(--amber)";
                desc.textContent = "Moderate Risk (Prediabetic Alert)";
                desc.style.color = "var(--amber)";
            } else {
                if (stop2) stop2.style.stopColor = "var(--rose)";
                desc.textContent = "High Diabetes Risk - Consult Doctor";
                desc.style.color = "var(--rose)";
            }
        }

        // Main calculation trigger
        function runPrediction() {
            const formData = new FormData(diabetesForm);
            
            fetch("/api/predict_diabetes", {
                method: "POST",
                body: formData
            })
            .then(res => res.json())
            .then(data => {
                if (data.success) {
                    const probability = data.probability_yes * 100;
                    updateRiskGauge(probability);

                    // Show detailed card prediction
                    const outcomeVal = document.getElementById("outcome-value");
                    const outcomeCard = document.getElementById("outcome-card");
                    
                    if (outcomeVal) {
                        if (data.prediction === 1) {
                            outcomeVal.innerHTML = "<span class='badge badge-rose'>Positive (Diabetes)</span>";
                            if (outcomeCard) outcomeCard.style.borderColor = "var(--rose)";
                        } else {
                            outcomeVal.innerHTML = "<span class='badge badge-emerald'>Negative (Healthy)</span>";
                            if (outcomeCard) outcomeCard.style.borderColor = "var(--emerald)";
                        }
                    }

                    // Update metrics / probabilities text
                    const yesVal = document.getElementById("prob-yes-val");
                    const noVal = document.getElementById("prob-no-val");
                    if (yesVal) yesVal.textContent = `${(data.probability_yes * 100).toFixed(1)}%`;
                    if (noVal) noVal.textContent = `${(data.probability_no * 100).toFixed(1)}%`;
                }
            })
            .catch(err => console.error("Error predicting diabetes:", err));
        }

        // Trigger on submit (fallback/button click)
        diabetesForm.addEventListener("submit", function (e) {
            e.preventDefault();
            runPrediction();
        });

        // Setup real-time slider value labels and trigger live predictions on input
        const sliders = diabetesForm.querySelectorAll("input[type='range']");
        sliders.forEach(slider => {
            const valSpan = document.getElementById(`${slider.id}-val`);
            
            slider.addEventListener("input", () => {
                if (valSpan) valSpan.textContent = slider.value;
                // Live recalculate
                runPrediction();
            });
        });

        // Trigger on dropdown selection change
        const modelSelect = document.getElementById("model_name");
        if (modelSelect) {
            modelSelect.addEventListener("change", runPrediction);
        }

        // Run initial prediction immediately so the dashboard displays loaded state
        runPrediction();
    }

    // -----------------------------------------------------------------
    // SIMPLE LINEAR REGRESSION (SALES & ADVERTISING)
    // -----------------------------------------------------------------
    const regChartCanvas = document.getElementById("sales-reg-chart");
    if (regChartCanvas) {
        const budgetSlider = document.getElementById("adv-budget");
        const budgetValText = document.getElementById("adv-budget-val");
        const predSalesText = document.getElementById("predicted-sales");
        
        let salesChart = null;

        // Fetch initial chart data and model parameters
        fetch("/api/sales_regression_data")
            .then(res => res.json())
            .then(data => {
                const points = data.points.map(p => ({ x: p.Advertising, y: p.Sales }));
                const line = data.line_data;
                const slope = data.slope;
                const intercept = data.intercept;

                // Setup math indicators
                document.getElementById("formula-repr").textContent = `Sales = ${intercept.toFixed(4)} + ${slope.toFixed(4)} * Advertising`;

                // Initial prediction point
                let currentBudget = parseFloat(budgetSlider.value);
                let currentPred = intercept + slope * currentBudget;

                // Draw Chart.js scatter plot
                salesChart = new Chart(regChartCanvas, {
                    type: "scatter",
                    data: {
                        datasets: [
                            {
                                label: "Actual Sales & Ad Spend",
                                data: points,
                                backgroundColor: "rgba(99, 102, 241, 0.5)",
                                borderColor: "hsl(245, 82%, 58%)",
                                borderWidth: 1,
                                pointRadius: 5,
                                pointHoverRadius: 7
                            },
                            {
                                label: "Regression Fit Line",
                                data: line,
                                type: "line",
                                borderColor: "rgba(16, 185, 129, 0.8)",
                                borderDash: [],
                                borderWidth: 3,
                                fill: false,
                                pointRadius: 0
                            },
                            {
                                label: "Live Prediction Point",
                                data: [{ x: currentBudget, y: currentPred }],
                                backgroundColor: "hsl(355, 78%, 56%)",
                                borderColor: "#fff",
                                borderWidth: 2,
                                pointRadius: 8,
                                pointHoverRadius: 10,
                                z: 10
                            }
                        ]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        scales: {
                            x: {
                                title: { display: true, text: "Advertising Budget (in Thousands $)", font: { weight: "bold" } },
                                grid: { color: "var(--border-light)" }
                            },
                            y: {
                                title: { display: true, text: "Sales (in Thousands $)", font: { weight: "bold" } },
                                grid: { color: "var(--border-light)" }
                            }
                        },
                        plugins: {
                            legend: {
                                labels: { font: { family: "Inter", weight: "500" } }
                            }
                        }
                    }
                });

                // Update on slider change
                function updateRegressionPrediction() {
                    const spend = parseFloat(budgetSlider.value);
                    budgetValText.textContent = spend.toFixed(1);

                    // Predict using frontend formula
                    const pred = intercept + slope * spend;
                    predSalesText.textContent = `$${pred.toFixed(2)}k`;

                    // Update red dot in dataset
                    if (salesChart) {
                        salesChart.data.datasets[2].data = [{ x: spend, y: pred }];
                        salesChart.update("none"); // Update instantly without animation flicker
                    }
                }

                budgetSlider.addEventListener("input", updateRegressionPrediction);
                updateRegressionPrediction();
            });
    }

    // -----------------------------------------------------------------
    // TITANIC DATA PREPROCESSING PIPELINE
    // -----------------------------------------------------------------
    const preprocessingTimeline = document.getElementById("preprocess-timeline");
    if (preprocessingTimeline) {
        const steps = preprocessingTimeline.querySelectorAll(".pipeline-step");
        
        steps.forEach(step => {
            step.addEventListener("click", () => {
                // Remove active classes
                steps.forEach(s => s.classList.remove("active"));
                step.classList.add("active");

                const targetStage = step.getAttribute("data-stage");
                
                // Hide all preview panels
                document.querySelectorAll(".preprocessing-stage-panel").forEach(panel => {
                    panel.style.display = "none";
                });

                // Show target preview panel
                const activePanel = document.getElementById(`panel-${targetStage}`);
                if (activePanel) {
                    activePanel.style.display = "block";
                }
            });
        });
    }

    // -----------------------------------------------------------------
    // TITANIC SUPERVISED COMPARISON
    // -----------------------------------------------------------------
    const supervisedChartCanvas = document.getElementById("titanic-supervised-chart");
    if (supervisedChartCanvas) {
        // Fetch accuracy scores embedded in data attributes or via API
        const dtAcc = parseFloat(supervisedChartCanvas.getAttribute("data-dt-acc"));
        const lrAcc = parseFloat(supervisedChartCanvas.getAttribute("data-lr-acc"));
        const dtF1 = parseFloat(supervisedChartCanvas.getAttribute("data-dt-f1"));
        const lrF1 = parseFloat(supervisedChartCanvas.getAttribute("data-lr-f1"));

        new Chart(supervisedChartCanvas, {
            type: "bar",
            data: {
                labels: ["Accuracy", "F1 Score"],
                datasets: [
                    {
                        label: "Decision Tree",
                        data: [dtAcc, dtF1],
                        backgroundColor: "rgba(99, 102, 241, 0.8)",
                        borderColor: "hsl(245, 82%, 58%)",
                        borderWidth: 1,
                        borderRadius: 6
                    },
                    {
                        label: "Linear Probability Model",
                        data: [lrAcc, lrF1],
                        backgroundColor: "rgba(20, 184, 166, 0.8)",
                        borderColor: "hsl(175, 75%, 42%)",
                        borderWidth: 1,
                        borderRadius: 6
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 1.0,
                        ticks: { callback: value => `${(value * 100).toFixed(0)}%` },
                        grid: { color: "var(--border-light)" }
                    },
                    x: {
                        grid: { display: false }
                    }
                },
                plugins: {
                    legend: { position: "top" }
                }
            }
        });
    }

    // -----------------------------------------------------------------
    // TITANIC UNSUPERVISED PCA + K-MEANS
    // -----------------------------------------------------------------
    const pcaChartCanvas = document.getElementById("titanic-pca-chart");
    if (pcaChartCanvas) {
        const clusterSlider = document.getElementById("unsupervised-clusters");
        const clusterValText = document.getElementById("cluster-count-label");
        const colorSelect = document.getElementById("color-scheme-select");
        
        let pcaChart = null;
        let cachedScatterData = null;

        const clusterColors = [
            "rgba(99, 102, 241, 0.85)", // Indigo
            "rgba(20, 184, 166, 0.85)", // Teal
            "rgba(245, 158, 11, 0.85)", // Amber
            "rgba(239, 68, 68, 0.85)",  // Red
            "rgba(139, 92, 246, 0.85)", // Purple
            "rgba(236, 72, 153, 0.85)"  // Pink
        ];

        const survivalColors = [
            "rgba(239, 68, 68, 0.8)", // Perished (Red)
            "rgba(16, 185, 129, 0.8)" // Survived (Green)
        ];

        function fetchAndDrawUnsupervised(k) {
            clusterValText.textContent = k;
            
            fetch(`/api/titanic_unsupervised_data?k=${k}`)
                .then(res => res.json())
                .then(data => {
                    cachedScatterData = data.scatter_data;
                    updateUnsupervisedChart();
                    renderClusterDemographics(data.profiles);
                });
        }

        function updateUnsupervisedChart() {
            if (!cachedScatterData) return;

            const colorMode = colorSelect.value; // 'cluster' or 'survival'
            let datasets = [];

            if (colorMode === "cluster") {
                const k = parseInt(clusterSlider.value);
                // Create a dataset for each cluster
                for (let c = 0; c < k; c++) {
                    const clusterPoints = cachedScatterData
                        .filter(p => p.cluster === c)
                        .map(p => ({ x: p.x, y: p.y, label: p }));

                    datasets.push({
                        label: `Cluster ${c + 1}`,
                        data: clusterPoints,
                        backgroundColor: clusterColors[c % clusterColors.length],
                        pointRadius: 4.5,
                        pointHoverRadius: 6.5
                    });
                }
            } else {
                // Create a dataset for Survived vs Perished
                const perishedPoints = cachedScatterData
                    .filter(p => p.survived === 0)
                    .map(p => ({ x: p.x, y: p.y, label: p }));

                const survivedPoints = cachedScatterData
                    .filter(p => p.survived === 1)
                    .map(p => ({ x: p.x, y: p.y, label: p }));

                datasets.push(
                    {
                        label: "Perished (0)",
                        data: perishedPoints,
                        backgroundColor: survivalColors[0],
                        pointRadius: 4.5,
                        pointHoverRadius: 6.5
                    },
                    {
                        label: "Survived (1)",
                        data: survivedPoints,
                        backgroundColor: survivalColors[1],
                        pointRadius: 4.5,
                        pointHoverRadius: 6.5
                    }
                );
            }

            if (pcaChart) {
                pcaChart.data.datasets = datasets;
                pcaChart.update();
            } else {
                pcaChart = new Chart(pcaChartCanvas, {
                    type: "scatter",
                    data: { datasets: datasets },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        scales: {
                            x: {
                                title: { display: true, text: "Principal Component 1", font: { weight: "bold" } },
                                grid: { color: "var(--border-light)" }
                            },
                            y: {
                                title: { display: true, text: "Principal Component 2", font: { weight: "bold" } },
                                grid: { color: "var(--border-light)" }
                            }
                        },
                        plugins: {
                            tooltip: {
                                callbacks: {
                                    label: function (context) {
                                        const raw = context.raw.label;
                                        return [
                                            `PC1: ${raw.x.toFixed(2)}, PC2: ${raw.y.toFixed(2)}`,
                                            `Age: ${raw.age.toFixed(0)} yrs`,
                                            `Gender: ${raw.sex}`,
                                            `Pclass: Class ${raw.pclass}`,
                                            `Fare: $${raw.fare.toFixed(2)}`,
                                            `Survived: ${raw.survived === 1 ? 'Yes' : 'No'}`
                                        ];
                                    }
                                }
                            }
                        }
                    }
                });
            }
        }

        function renderClusterDemographics(profiles) {
            const tbody = document.getElementById("cluster-profile-tbody");
            if (!tbody) return;

            tbody.innerHTML = "";
            profiles.forEach(p => {
                const tr = document.createElement("tr");
                tr.innerHTML = `
                    <td><strong>Cluster ${p.cluster_id + 1}</strong></td>
                    <td>${p.count}</td>
                    <td>${p.avg_age.toFixed(1)} yrs</td>
                    <td>$${p.avg_fare.toFixed(2)}</td>
                    <td><span class="badge ${p.survival_rate > 50 ? 'badge-success' : 'badge-danger'}">${p.survival_rate.toFixed(1)}%</span></td>
                    <td>${p.female_ratio.toFixed(1)}%</td>
                    <td>${p.first_class_ratio.toFixed(1)}%</td>
                `;
                tbody.appendChild(tr);
            });
        }

        clusterSlider.addEventListener("input", () => {
            fetchAndDrawUnsupervised(parseInt(clusterSlider.value));
        });

        colorSelect.addEventListener("change", () => {
            updateUnsupervisedChart();
        });

        // Initialize first load
        fetchAndDrawUnsupervised(3);
    }

    // -----------------------------------------------------------------
    // NETWORK ANOMALY DETECTION (DDoS UPLOAD)
    // -----------------------------------------------------------------
    const uploadZone = document.getElementById("anomaly-upload-zone");
    const fileInput = document.getElementById("file");
    const ddosForm = document.getElementById("ddos-form");

    if (uploadZone && fileInput) {
        // Dynamic file input label
        fileInput.addEventListener("change", () => {
            if (fileInput.files.length > 0) {
                const fileName = fileInput.files[0].name;
                document.getElementById("file-selected-text").textContent = `Selected: ${fileName}`;
                uploadZone.style.borderColor = "var(--primary)";
                uploadZone.style.backgroundColor = "var(--primary-bg)";
            }
        });

        // Drag and drop event listeners
        ["dragenter", "dragover"].forEach(eventName => {
            uploadZone.addEventListener(eventName, (e) => {
                e.preventDefault();
                e.stopPropagation();
                uploadZone.classList.add("dragover");
            }, false);
        });

        ["dragleave", "drop"].forEach(eventName => {
            uploadZone.addEventListener(eventName, (e) => {
                e.preventDefault();
                e.stopPropagation();
                uploadZone.classList.remove("dragover");
            }, false);
        });

        uploadZone.addEventListener("drop", (e) => {
            const dt = e.dataTransfer;
            const files = dt.files;
            if (files.length > 0) {
                fileInput.files = files;
                const fileName = files[0].name;
                document.getElementById("file-selected-text").textContent = `Selected: ${fileName}`;
                uploadZone.style.borderColor = "var(--primary)";
                uploadZone.style.backgroundColor = "var(--primary-bg)";
            }
        }, false);

        uploadZone.addEventListener("click", () => {
            fileInput.click();
        });

        if (ddosForm) {
            ddosForm.addEventListener("submit", () => {
                // Show loading spinner/progress bar
                const loader = document.getElementById("prediction-loading-spinner");
                if (loader) {
                    loader.style.display = "flex";
                }
            });
        }
    }
});
