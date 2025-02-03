let emailIndex = 0;   // Tracks the current email index
let totalEmails = 15; // Total number of emails (set dynamically if needed)

// Fetch email and tokenize its content
const fetchEmail = (index) => {
    fetch('/get_email', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email_index: index })
    })
        .then(response => response.json())
        .then(data => {
            const subjectContainer = document.getElementById('email-subject').querySelector('span');
            const bodyContainer = document.getElementById('email-body');

            // Tokenize the subject and body
            subjectContainer.innerHTML = tokenize(data.subject);
            bodyContainer.innerHTML = tokenize(data.body);

            // Reset the checkbox to unchecked
            document.getElementById('mark-legit').checked = false;

            // Update navigation button states
            updateNavigationButtons();

            // Re-check the "Check" button status (in case no tokens are selected initially)
            updateCheckButtonState();

            document.getElementById('check-email').disabled = true; // Default disabled
        })
        .catch(error => console.error('Error fetching email:', error));
};

// Tokenize text into clickable words
const tokenize = (text) => {
    return text
        .split(' ')
        .map(word => `<span class="token">${word}</span>`)
        .join(' ');
};

// Add a single event listener using event delegation
const initializeEmailBoxClick = () => {
    const emailBox = document.querySelector('.email-box');

    // Ensure only token clicks are processed
    emailBox.addEventListener('click', (event) => {
        if (event.target.classList.contains('token')) {
            const checkbox = document.getElementById('mark-legit');
            const checkButton = document.getElementById('check-email');

            // If the checkbox is checked and a token is selected, uncheck the checkbox
            if (checkbox.checked) {
                checkbox.checked = false;
            }

            // Toggle the "highlighted" class for tokens
            event.target.classList.toggle('highlighted');

            // Update the "Check" button state
            updateCheckButtonState();
        }
    });
};


// Update the "Check" button's state based on highlighted tokens or checkbox state
const updateCheckButtonState = () => {
    const highlightedTokens = document.querySelectorAll('.token.highlighted'); // Get all highlighted tokens
    const checkButton = document.getElementById('check-email');
    const checkbox = document.getElementById('mark-legit'); // Get the checkbox element

    // Enable the "Check" button if there are highlighted tokens or the checkbox is checked
    checkButton.disabled = !(highlightedTokens.length > 0 || checkbox.checked);
};

// Event listener to track changes to the checkbox state
document.getElementById('mark-legit').addEventListener('change', () => {
    if (document.getElementById('mark-legit').checked) {
        // Unhighlight all selected tokens when the checkbox is checked
        document.querySelectorAll('.token.highlighted').forEach(token => {
            token.classList.remove('highlighted');
        });
    }
    // Update the "Check" button state
    updateCheckButtonState();
});

// Update navigation buttons' states
const updateNavigationButtons = () => {
    const prevButton = document.getElementById('prev-email');
    const nextButton = document.getElementById('next-email');

    prevButton.disabled = emailIndex === 0;
    nextButton.disabled = emailIndex >= totalEmails - 1;
};

// Navigation buttons
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

// Check button
document.getElementById('check-email').addEventListener('click', () => {
    const highlightedTokens = Array.from(
        document.querySelectorAll('.token.highlighted')
    ).map(token => token.textContent);

    // Get the result message container
    const resultMessage = document.getElementById('result-message');

    // Fetch the label from the email data (assuming it is returned in the response)
    fetch('/get_email_label', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email_index: emailIndex }) // Send the current email index to get the label
    })
        .then(response => response.json())
        .then(data => {
            const isPhishing = data.label === 1; // If the label is 1, it's phishing; else, it's legit
            // Display the result message
            resultMessage.style.display = 'block'; // Show the result message
            if (isPhishing) {
                resultMessage.textContent = 'This email is a phishing attempt.';
                resultMessage.style.color = 'red';  // Set text color to red for phishing
            } else {
                resultMessage.textContent = 'This email is legit.';
                resultMessage.style.color = 'green';  // Set text color to green for legit
            }
        })
        .catch(error => console.error('Error fetching email label:', error));

    // Display highlighted tokens for debugging or later integration
    const resultsPlaceholder = document.getElementById('results-placeholder');
    const highlightedWordsList = document.getElementById('highlighted-words-list');

    // Update the results list with the highlighted tokens
    if (highlightedTokens.length > 0) {
        highlightedWordsList.textContent = highlightedTokens.join(', ');
    } else {
        highlightedWordsList.textContent = 'No words highlighted yet.';
    }

    // Make the results placeholder visible after showing the result message
    resultsPlaceholder.style.display = 'block';
});



// Continue button
document.getElementById('continue-btn').addEventListener('click', function() {
    // Hide the info box and show the email content and navigation buttons
    document.getElementById('info-box').style.display = 'none';
    document.getElementById('email-content').style.display = 'block';
    document.getElementById('nav-buttons').style.display = 'flex'; // Show the navigation buttons
   // Make the info icon visible when the content is shown
   document.getElementById('info-icon').style.display = 'inline-block';
});

// Info button
document.getElementById('info-icon').addEventListener('click', function() {
    // Toggle the info box visibility
    const infoBox = document.getElementById('info-box');
    const emailContent = document.getElementById('email-content');
    const navButtons = document.getElementById('nav-buttons');
    const resultsPlaceholder = document.getElementById('results-placeholder'); // Get the results placeholder

    if (infoBox.style.display === 'none') {
        infoBox.style.display = 'block';  // Show the info box
        emailContent.style.display = 'none';  // Hide the email content
        navButtons.style.display = 'none';  // Hide the navigation buttons
        resultsPlaceholder.style.display = 'none';  // Hide the results placeholder
    } else {
        infoBox.style.display = 'none';  // Hide the info box
        emailContent.style.display = 'block';  // Show the email content
        navButtons.style.display = 'flex';  // Show the navigation buttons
    }
});

// Initial fetch
fetchEmail(emailIndex);

// Initialize event delegation for clickable tokens
initializeEmailBoxClick();
