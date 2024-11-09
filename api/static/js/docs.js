// Function to copy text to clipboard
async function copyToClipboard(text) {
    try {
        await navigator.clipboard.writeText(text);
        return true;
    } catch (err) {
        console.error('Failed to copy text:', err);
        return false;
    }
}

// Add copy buttons to all code blocks
function addCopyButtons() {
    document.querySelectorAll('code.d-block').forEach(codeBlock => {
        if (!codeBlock.nextElementSibling?.classList.contains('copy-btn')) {
            const button = document.createElement('button');
            button.textContent = 'Copy';
            button.className = 'btn btn-sm btn-secondary copy-btn mt-2';
            button.onclick = async () => {
                const success = await copyToClipboard(codeBlock.textContent);
                button.textContent = success ? 'Copied!' : 'Failed to copy';
                setTimeout(() => button.textContent = 'Copy', 2000);
            };
            codeBlock.parentNode.insertBefore(button, codeBlock.nextSibling);
        }
    });
}

// Format JSON with syntax highlighting
function formatJSON(json) {
    if (typeof json !== 'string') {
        json = JSON.stringify(json, null, 2);
    }
    return json.replace(/("(\\u[a-zA-Z0-9]{4}|\\[^u]|[^\\"])*"(\s*:)?|\b(true|false|null)\b|-?\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?)/g, function (match) {
        let cls = 'number';
        if (/^"/.test(match)) {
            if (/:$/.test(match)) {
                cls = 'key';
            } else {
                cls = 'string';
            }
        } else if (/true|false/.test(match)) {
            cls = 'boolean';
        } else if (/null/.test(match)) {
            cls = 'null';
        }
        return '<span class="json-' + cls + '">' + match + '</span>';
    });
}

// Show loading indicator
function showLoading(responseElement) {
    responseElement.innerHTML = `
        <div class="d-flex justify-content-center align-items-center p-4">
            <div class="spinner-border text-primary" role="status">
                <span class="visually-hidden">Loading...</span>
            </div>
        </div>`;
}

// Handle API errors
function handleError(responseElement, error, status) {
    responseElement.innerHTML = `
        <div class="alert alert-danger">
            <h5>Error ${status || ''}</h5>
            <p>${error.message || error}</p>
        </div>`;
}

// Try an endpoint
async function tryEndpoint(endpoint) {
    const responseElement = document.getElementById('response');
    showLoading(responseElement);
    
    try {
        const response = await fetch(endpoint);
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        responseElement.innerHTML = `<code class="json">${formatJSON(data)}</code>`;
    } catch (error) {
        handleError(responseElement, error, error.status);
    }
}

// Initialize documentation page
function initDocs() {
    addCopyButtons();
    
    // Add CSS for JSON highlighting
    const style = document.createElement('style');
    style.textContent = `
        .json-string { color: #7ec699; }
        .json-number { color: #f08d49; }
        .json-boolean { color: #f08d49; }
        .json-null { color: #f08d49; }
        .json-key { color: #5fb3b3; }
    `;
    document.head.appendChild(style);
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', initDocs);
