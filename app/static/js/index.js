document.getElementById('analyze-btn').addEventListener('click', () => {
    const subject = document.getElementById('subject').value;
    const body = document.getElementById('body').value;

    fetch('/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ subject, body })
    })
        .then(response => response.json())
        .then(data => {
            const resultsPlaceholder = document.getElementById('results-placeholder');
            resultsPlaceholder.innerHTML = `<h3>${data.xai_results}</h3>`;
        });
});
