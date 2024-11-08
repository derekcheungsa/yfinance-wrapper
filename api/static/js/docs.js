async function tryEndpoint(endpoint) {
    const responseElement = document.getElementById('response');
    responseElement.textContent = 'Loading...';
    
    try {
        const response = await fetch(endpoint);
        const data = await response.json();
        responseElement.textContent = JSON.stringify(data, null, 2);
    } catch (error) {
        responseElement.textContent = `Error: ${error.message}`;
    }
}
