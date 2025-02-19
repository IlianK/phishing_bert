document.getElementById('analyze-btn').addEventListener('click', () => {
    document.getElementById('prediction-result').innerHTML = "";
    document.getElementById('lime-visual').innerHTML = "";
    const spinner = document.getElementById('loading-spinner');
    const timerCounter = document.getElementById('timer-counter');
    spinner.style.display = 'block';
    let seconds = 0;
    const timerInterval = setInterval(() => {
      seconds++;
      timerCounter.textContent = seconds;
    }, 1000);
  
    document.getElementById('subject').classList.add('analyzing');
    document.getElementById('body').classList.add('analyzing');
  
    const subject = document.getElementById('subject').value;
    const body = document.getElementById('body').value;
  
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
      const iframe = document.createElement('iframe');
      iframe.style.width = "100%";
      iframe.style.height = "600px";
      iframe.style.border = "none";
      iframe.srcdoc = data.lime_html;
      const limeVisual = document.getElementById('lime-visual');
      limeVisual.innerHTML = "";
      limeVisual.appendChild(iframe);
    });
  });
  
  document.getElementById('subject').addEventListener('focus', () => {
    document.getElementById('subject').classList.remove('analyzing');
  });
  
  document.getElementById('body').addEventListener('focus', () => {
    document.getElementById('body').classList.remove('analyzing');
  });
  