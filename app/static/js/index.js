document.getElementById('analyze-btn').addEventListener('click', () => {
  // Clear previous results and visualizations
  document.getElementById('prediction-result').innerHTML = "";
  document.getElementById('lime-visual').innerHTML = "";
  // Hide results placeholder until analysis is complete
  document.getElementById('results-placeholder').style.display = 'none';

  const spinner = document.getElementById('loading-spinner');
  const timerCounter = document.getElementById('timer-counter');
  spinner.style.display = 'block';
  let seconds = 0;
  const timerInterval = setInterval(() => {
    seconds++;
    timerCounter.textContent = seconds;
  }, 1000);

  const subjectField = document.getElementById('subject');
  const bodyField = document.getElementById('body');

  // Remove previous error highlighting
  subjectField.classList.remove('error');
  bodyField.classList.remove('error');

  const subject = subjectField.value.trim();
  const body = bodyField.value.trim();

  // Check: At least one input (subject or body) is required
  if (subject === "" && body === "") {
    // Add error class to highlight missing input
    subjectField.classList.add('error');
    bodyField.classList.add('error');
    spinner.style.display = 'none';
    clearInterval(timerInterval);
    alert("Please provide at least an Email Subject or Email Body.");
    return;
  }

  // Add a visual pointer to indicate analysis is in progress
  subjectField.classList.add('analyzing');
  bodyField.classList.add('analyzing');

  fetch('/analyze', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ subject, body })
  })
    .then(response => response.json())
    .then(data => {
      clearInterval(timerInterval);
      timerCounter.textContent = `Analyzed in ${seconds} seconds`;
      spinner.style.display = 'none';

      document.getElementById('prediction-result').innerHTML = `
        <p><strong>Prediction:</strong> ${data.predicted_label}</p>
        <p><strong>Confidence:</strong> ${data.confidence.toFixed(2)}</p>
      `;

      // Create and display the LIME visualization
      const iframe = document.createElement('iframe');
      iframe.style.width = "100%";
      iframe.style.height = "600px";
      iframe.style.border = "none";
      iframe.srcdoc = data.lime_html;
      const limeVisual = document.getElementById('lime-visual');
      limeVisual.innerHTML = "";
      limeVisual.appendChild(iframe);

      // Show the results container only after analysis is complete
      document.getElementById('results-placeholder').style.display = 'block';
    });
});

document.getElementById('subject').addEventListener('focus', () => {
  document.getElementById('subject').classList.remove('analyzing');
  document.getElementById('subject').classList.remove('error');
});

document.getElementById('body').addEventListener('focus', () => {
  document.getElementById('body').classList.remove('analyzing');
  document.getElementById('body').classList.remove('error');
});
