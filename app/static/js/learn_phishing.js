// learn_phishing.js

let emailIndex = 0;   // Tracks the current email index
let totalEmails = 15; // Or fetch dynamically if needed

// Fetch email and tokenize its content
function fetchEmail(index) {
    // Clear any old LIME visual
    document.getElementById('lime-visual').innerHTML = "";

    // Hide results from the previous email
    document.getElementById('results-placeholder').style.display = 'none';
    document.getElementById('highlighted-words-list').textContent = 'No words highlighted yet.';
    
    // Clear any previous result message
    const resultMessage = document.getElementById('result-message');
    resultMessage.style.display = 'none';
    resultMessage.textContent = '';

    // Make sure the "Check" button is disabled until user makes a new choice
    document.getElementById('check-email').disabled = true;

    fetch('/get_email', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email_index: index })
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            console.error(data.error);
            return;
        }

        const subjectContainer = document
          .getElementById('email-subject')
          .querySelector('span');
        const bodyContainer = document.getElementById('email-body');

        // Tokenize the subject and body
        subjectContainer.innerHTML = tokenize(data.subject);
        bodyContainer.innerHTML = tokenize(data.body);

        // Reset the checkbox
        document.getElementById('mark-legit').checked = false;

        // Update nav button states and check-button state
        updateNavigationButtons();
        updateCheckButtonState();
    })
    .catch(error => console.error('Error fetching email:', error));
}


// Tokenize text into clickable words
function tokenize(text) {
    return text
        .split(' ')
        .map(word => `<span class="token">${word}</span>`)
        .join(' ');
}

// Event delegation for clickable tokens
function initializeEmailBoxClick() {
    const emailBox = document.querySelector('.email-box');
    emailBox.addEventListener('click', (event) => {
        if (event.target.classList.contains('token')) {
            const checkbox = document.getElementById('mark-legit');
            // If user had checked legit, uncheck it if they highlight a token
            if (checkbox.checked) {
                checkbox.checked = false;
            }
            // Toggle highlight
            event.target.classList.toggle('highlighted');
            // Update "Check" button
            updateCheckButtonState();
        }
    });
}

// Enable/disable "Check" button
function updateCheckButtonState() {
    const highlightedTokens = document.querySelectorAll('.token.highlighted');
    const checkButton = document.getElementById('check-email');
    const checkbox = document.getElementById('mark-legit');
    checkButton.disabled = !(highlightedTokens.length > 0 || checkbox.checked);
}

// On checkbox change
document.getElementById('mark-legit').addEventListener('change', () => {
    const checkbox = document.getElementById('mark-legit');
    if (checkbox.checked) {
        // Unhighlight all tokens if the checkbox is set to legit
        document.querySelectorAll('.token.highlighted').forEach(token => {
            token.classList.remove('highlighted');
        });
    }
    updateCheckButtonState();
});

// Next / Previous
document.getElementById('prev-email').addEventListener('click', () => {
    if (emailIndex > 0) {
        emailIndex--;
        fetchEmail(emailIndex);
    }
});

document.getElementById('next-email').addEventListener('click', () => {
    if (emailIndex < totalEmails - 1) {
        emailIndex++;
        fetchEmail(emailIndex);
    }
});

function updateNavigationButtons() {
    document.getElementById('prev-email').disabled = emailIndex === 0;
    document.getElementById('next-email').disabled = emailIndex >= totalEmails - 1;
}

// "Check" button
document.getElementById('check-email').addEventListener('click', () => {
    const highlightedTokens = Array.from(document.querySelectorAll('.token.highlighted'))
                                  .map(token => token.textContent);
    const resultMessage = document.getElementById('result-message');
    
    // 1) Show if it's truly phishing or legit
    fetch('/get_email_label', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email_index: emailIndex })
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            console.error(data.error);
            return;
        }
        const isPhishing = data.label === 1; // 1 = phishing
        resultMessage.style.display = 'block';
        if (isPhishing) {
            resultMessage.textContent = 'This email is a phishing attempt.';
            resultMessage.style.color = 'red';
        } else {
            resultMessage.textContent = 'This email is legit.';
            resultMessage.style.color = 'green';
        }
    })
    .catch(error => console.error('Error fetching email label:', error));

    // 2) Display highlighted tokens
    const resultsPlaceholder = document.getElementById('results-placeholder');
    const highlightedWordsList = document.getElementById('highlighted-words-list');
    if (highlightedTokens.length > 0) {
        highlightedWordsList.textContent = highlightedTokens.join(', ');
    } else {
        highlightedWordsList.textContent = 'No words highlighted yet.';
    }
    resultsPlaceholder.style.display = 'block';

    // 3) Fetch the precomputed LIME HTML and embed it
    fetch('/get_lime_html', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email_index: emailIndex })
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            console.error(data.error);
            return;
        }
        const limeVisual = document.getElementById('lime-visual');
        limeVisual.innerHTML = ''; // Clear old
        // Insert LIME HTML into an iframe
        const iframe = document.createElement('iframe');
        iframe.style.width = '100%';
        iframe.style.height = '600px';
        iframe.style.border = 'none';
        iframe.srcdoc = data.lime_html;
        limeVisual.appendChild(iframe);
    })
    .catch(error => console.error('Error fetching LIME HTML:', error));
});

// "Continue" button
document.getElementById('continue-btn').addEventListener('click', () => {
    document.getElementById('info-box').style.display = 'none';
    document.getElementById('email-content').style.display = 'block';
    document.getElementById('nav-buttons').style.display = 'flex';
    document.getElementById('info-icon').style.display = 'inline-block';
});

// Info icon
document.getElementById('info-icon').addEventListener('click', () => {
    const infoBox = document.getElementById('info-box');
    const emailContent = document.getElementById('email-content');
    const navButtons = document.getElementById('nav-buttons');
    const resultsPlaceholder = document.getElementById('results-placeholder');

    if (infoBox.style.display === 'none') {
        infoBox.style.display = 'block';
        emailContent.style.display = 'none';
        navButtons.style.display = 'none';
        resultsPlaceholder.style.display = 'none';
    } else {
        infoBox.style.display = 'none';
        emailContent.style.display = 'block';
        navButtons.style.display = 'flex';
    }
});

// On page load, fetch first email
fetchEmail(emailIndex);
// Initialize event delegation for tokens
initializeEmailBoxClick();
