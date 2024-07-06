document.addEventListener('DOMContentLoaded', () => {
    const registerForm = document.getElementById('register-form');
    const loginForm = document.getElementById('login-form');
    const logoutButton = document.getElementById('logout-button');
    const pdfUploadForm = document.getElementById('pdf-upload-form');
    const queryForm = document.getElementById('query-form');
    const responseDiv = document.getElementById('response');
    const uploadStatus = document.getElementById('upload-status');

    let token = '';

    registerForm.addEventListener('submit', async (event) => {
        event.preventDefault();
        const formData = new FormData(registerForm);
        const response = await fetch('/auth/register', {
            method: 'POST',
            body: new URLSearchParams(formData)
        });
        const result = await response.json();
        alert(result.message);
        registerForm.reset();
    });

    loginForm.addEventListener('submit', async (event) => {
        event.preventDefault();
        const formData = new FormData(loginForm);
        const response = await fetch('/auth/token', {
            method: 'POST',
            body: new URLSearchParams(formData)
        });
        const result = await response.json();
        if (response.ok) {
            if (result.access_token) {
                token = result.access_token;
                loginForm.style.display = 'none';
                logoutButton.style.display = 'block';
                document.getElementById('pdf-section').style.display = 'block';
                document.getElementById('query-section').style.display = 'block';
            } else {
                alert('Login failed: No access token received');
            }
        } else {
            alert(`Login failed: ${result.detail}`);
        }
        loginForm.reset();
    });

    logoutButton.addEventListener('click', () => {
        token = '';
        loginForm.style.display = 'block';
        logoutButton.style.display = 'none';
        document.getElementById('pdf-section').style.display = 'none';
        document.getElementById('query-section').style.display = 'none';
    });

    pdfUploadForm.addEventListener('submit', async (event) => {
        event.preventDefault();
        const formData = new FormData(pdfUploadForm);
        uploadStatus.textContent = "Uploading files...";
        try {
            const response = await fetch('/pdf/process_pdfs', {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${token}`
                },
                body: formData
            });
            const result = await response.json();
            if (response.ok) {
                alert(result.message);
                uploadStatus.textContent = result.message;
            } else if (result.detail) {
                alert(result.detail);
                uploadStatus.textContent = result.detail;
            } else {
                alert('Upload failed.');
                uploadStatus.textContent = 'Upload failed.';
            }
        } catch (error) {
            alert('An error occurred while uploading files.');
            console.error('Upload error:', error);
            uploadStatus.textContent = "Upload failed.";
        }
        pdfUploadForm.reset();
    });

    queryForm.addEventListener('submit', async (event) => {
        event.preventDefault();
        const formData = new FormData(queryForm);
        try {
            const response = await fetch('/qdrant/generate_response', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${token}`
                },
                body: JSON.stringify({
                    question: formData.get('question')
                })
            });
            const result = await response.json();
            if (response.ok) {
                responseDiv.innerHTML = `<p>${result.response}</p>`;
            } else {
                responseDiv.innerHTML = `<p>Error: ${result.detail || 'An error occurred'}</p>`;
            }
        } catch (error) {
            alert('An error occurred while processing your query.');
            console.error('Query error:', error);
            responseDiv.innerHTML = '<p>An error occurred while processing your query.</p>';
        }
        queryForm.reset();
    });
});
