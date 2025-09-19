/*document.addEventListener('DOMContentLoaded', () => {
    // --- CONFIGURATION ---
    const API_BASE_URL = '';
    const USER_ID = '4'; 

    // --- DOM ELEMENT REFERENCES ---
    const themeToggle = document.getElementById('theme-toggle');
    const uploadForm = document.getElementById('upload-form');
    const fileInput = document.getElementById('file-input');
    const uploadStatus = document.getElementById('upload-status');
    const docSelector = document.getElementById('doc-selector');
    const selectionStatus = document.getElementById('selection-status');
    const chatForm = document.getElementById('chat-form');
    const messageInput = document.getElementById('message-input');
    const sendButton = document.getElementById('send-button');
    const messagesContainer = document.getElementById('messages-container');

    let selectedDocumentId = null;

    // --- THEME MANAGEMENT ---
    const applyTheme = (theme) => {
        if (theme === 'dark') {
            document.documentElement.setAttribute('data-theme', 'dark');
            themeToggle.checked = true;
        } else {
            document.documentElement.removeAttribute('data-theme');
            themeToggle.checked = false;
        }
    };

    const handleThemeToggle = () => {
        const newTheme = themeToggle.checked ? 'dark' : 'light';
        localStorage.setItem('theme', newTheme);
        applyTheme(newTheme);
    };

    // --- CORE FUNCTIONS ---
    
    const fetchDocuments = async () => {
        try {
            const response = await fetch(`${API_BASE_URL}/documents/${USER_ID}`);
            if (!response.ok) throw new Error('Could not fetch documents.');
            const documents = await response.json();
            
            docSelector.innerHTML = '<option value="">-- Please select a document --</option>';
            for (const docId in documents) {
                const option = document.createElement('option');
                option.value = docId;
                option.textContent = documents[docId];
                docSelector.appendChild(option);
            }
        } catch (error) {
            console.error('Error fetching documents:', error);
            selectionStatus.textContent = 'Failed to load documents.';
        }
    };

    const addMessage = (text, sender) => {
        const messageElement = document.createElement('div');
        messageElement.classList.add('message', `${sender}-message`);
        const p = document.createElement('p');
        p.textContent = text;
        messageElement.appendChild(p);
        messagesContainer.appendChild(messageElement);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
        return messageElement;
    };
    
    /**
     * Parses text containing <think> tags and converts it to safe HTML.
     * @param {string} text - The raw text from the stream.
     * @returns {string} - HTML string with thoughts in a styled span.
     */
    /*const parseAndRenderText = (text) => {
        // Escape basic HTML characters to prevent XSS from unexpected API output
        const escapedText = text
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");

        // Now, safely replace our specific <think> tags with styled spans
        return escapedText.replace(/&lt;think&gt;(.*?)&lt;\/think&gt;/gs, '<span class="thought-text">$1</span>');
    };


    const handleFileUpload = async (event) => {
        event.preventDefault();
        const file = fileInput.files[0];
        if (!file) {
            uploadStatus.textContent = 'Please select a file first.';
            return;
        }

        const formData = new FormData();
        formData.append('user_id', USER_ID);
        formData.append('file', file);

        uploadStatus.textContent = `Uploading ${file.name}...`;
        try {
            const response = await fetch(`${API_BASE_URL}/upload/`, { method: 'POST', body: formData });
            if (!response.ok) throw new Error('Upload failed');
            const result = await response.json();
            uploadStatus.textContent = `Successfully uploaded: ${result.filename}`;
            fileInput.value = '';
            await fetchDocuments();
        } catch (error) {
            console.error('Upload error:', error);
            uploadStatus.textContent = 'Upload failed. Please try again.';
        }
    };

    const resetChatForDocument = async (documentId) => {
        if (!documentId) return;
        try {
            const response = await fetch(`${API_BASE_URL}/chat/stream/`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ user_id: USER_ID, document_id: documentId, query: 'reset' })
            });
            if (!response.ok) throw new Error('Failed to reset chat session.');
            
            const reader = response.body.getReader();
            const { value } = await reader.read();
            console.log('Reset confirmation:', new TextDecoder().decode(value));
            
            messagesContainer.innerHTML = '';
            addMessage(`Ready to chat with the selected document!`, 'bot');
        } catch (error) {
            console.error('Error resetting chat:', error);
            addMessage('Could not reset chat session.', 'bot');
        }
    };

    const handleDocumentSelection = async (event) => {
        selectedDocumentId = event.target.value;
        if (selectedDocumentId) {
            selectionStatus.textContent = `Selected: ${event.target.options[event.target.selectedIndex].text}`;
            await resetChatForDocument(selectedDocumentId);
            // messageInput.disabled = false;
            // sendButton.disabled = false;
            messageInput.focus();
        } else {
            selectionStatus.textContent = '';
            // messageInput.disabled = true;
            // sendButton.disabled = true;
            messagesContainer.innerHTML = '';
            addMessage('Hello! Please upload and select a document to begin chatting.', 'bot');
        }
    };

    const handleChatSubmit = async (event) => {
        event.preventDefault();
        const query = messageInput.value.trim();
        if (!query) return;

        payload = { user_id: USER_ID, query: query }
        if (selectedDocumentId) {
            payload.document_id = selectedDocumentId;
            }

        addMessage(query, 'user');
        messageInput.value = '';
        // messageInput.disabled = true;
        // sendButton.disabled = true;

        const botMessageElement = addMessage('', 'bot');
        botMessageElement.classList.add('thinking');
        const botParagraph = botMessageElement.querySelector('p');

        try {
            const response = await fetch(`${API_BASE_URL}/chat/stream/`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let accumulatedText = '';

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                
                accumulatedText += decoder.decode(value, { stream: true });
                // Use innerHTML to render the styled span for thoughts
                botParagraph.innerHTML = parseAndRenderText(accumulatedText);
                messagesContainer.scrollTop = messagesContainer.scrollHeight;
            }
            
        } catch (error) {
            console.error('Streaming error:', error);
            botParagraph.textContent = 'Sorry, an error occurred while getting a response.';
        } finally {
            botMessageElement.classList.remove('thinking');
            // messageInput.disabled = false;
            // sendButton.disabled = false;
            messageInput.focus();
        }
    };

    // --- INITIALIZATION ---
    uploadForm.addEventListener('submit', handleFileUpload);
    docSelector.addEventListener('change', handleDocumentSelection);
    chatForm.addEventListener('submit', handleChatSubmit);
    themeToggle.addEventListener('change', handleThemeToggle);

    // Check for saved theme on page load
    const savedTheme = localStorage.getItem('theme') || 'light';
    applyTheme(savedTheme);
    
    fetchDocuments();
});*/


document.addEventListener('DOMContentLoaded', () => {
    // --- CONFIGURATION ---
    const API_BASE_URL = ''; // 👉 set this to "http://localhost:9082" if needed
    const USER_ID = '4'; 

    // --- DOM ELEMENT REFERENCES ---
    const themeToggle = document.getElementById('theme-toggle');
    const uploadForm = document.getElementById('upload-form');
    const fileInput = document.getElementById('file-input');
    const uploadStatus = document.getElementById('upload-status');
    const docSelector = document.getElementById('doc-selector');
    const selectionStatus = document.getElementById('selection-status');
    const chatForm = document.getElementById('chat-form');
    const messageInput = document.getElementById('message-input');
    const sendButton = document.getElementById('send-button');
    const messagesContainer = document.getElementById('messages-container');

    let selectedDocumentId = null;

    // --- THEME MANAGEMENT ---
    const applyTheme = (theme) => {
        if (theme === 'dark') {
            document.documentElement.setAttribute('data-theme', 'dark');
            themeToggle.checked = true;
        } else {
            document.documentElement.removeAttribute('data-theme');
            themeToggle.checked = false;
        }
    };

    const handleThemeToggle = () => {
        const newTheme = themeToggle.checked ? 'dark' : 'light';
        localStorage.setItem('theme', newTheme);
        applyTheme(newTheme);
    };

    // --- CORE FUNCTIONS ---
    const fetchDocuments = async () => {
        try {
            const response = await fetch(`${API_BASE_URL}/documents/${USER_ID}`);
            if (!response.ok) throw new Error('Could not fetch documents.');
            const documents = await response.json();
            
            docSelector.innerHTML = '<option value="">-- Please select a document --</option>';
            for (const docId in documents) {
                const option = document.createElement('option');
                option.value = docId;
                option.textContent = documents[docId];
                docSelector.appendChild(option);
            }
        } catch (error) {
            console.error('Error fetching documents:', error);
            selectionStatus.textContent = 'Failed to load documents.';
        }
    };

    const addMessage = (text, sender) => {
        const messageElement = document.createElement('div');
        messageElement.classList.add('message', `${sender}-message`);
        const p = document.createElement('p');
        p.textContent = text;
        messageElement.appendChild(p);
        messagesContainer.appendChild(messageElement);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
        return messageElement;
    };
    
    const parseAndRenderText = (text) => {
        const escapedText = text
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
        return escapedText.replace(/&lt;think&gt;(.*?)&lt;\/think&gt;/gs, '<span class="thought-text">$1</span>');
    };

    const handleFileUpload = async (event) => {
        event.preventDefault();
        const file = fileInput.files[0];
        if (!file) {
            uploadStatus.textContent = 'Please select a file first.';
            return;
        }

        const formData = new FormData();
        formData.append('user_id', USER_ID);
        formData.append('file', file);

        uploadStatus.textContent = `Uploading ${file.name}...`;
        try {
            const response = await fetch(`${API_BASE_URL}/upload/`, { method: 'POST', body: formData });
            if (!response.ok) throw new Error('Upload failed');
            const result = await response.json();

            uploadStatus.textContent = `Successfully uploaded: ${result.filename}`;
            fileInput.value = '';

            // 🔥 Save document_id and set it as selected
            selectedDocumentId = result.document_id;
            console.log("Uploaded document_id:", selectedDocumentId);

            // Reset chat to this new document immediately
            await resetChatForDocument(selectedDocumentId);

            // Refresh document list in dropdown
            await fetchDocuments();

            // Auto-select in dropdown too
            if (docSelector) {
                docSelector.value = selectedDocumentId;
                selectionStatus.textContent = `Selected: ${result.filename}`;
            }

        } catch (error) {
            console.error('Upload error:', error);
            uploadStatus.textContent = 'Upload failed. Please try again.';
        }
    };

    const resetChatForDocument = async (documentId) => {
        if (!documentId) return;
        try {
            const response = await fetch(`${API_BASE_URL}/chat/stream/`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ user_id: USER_ID, document_id: documentId, query: 'reset' })
            });
            if (!response.ok) throw new Error('Failed to reset chat session.');
            
            const reader = response.body.getReader();
            const { value } = await reader.read();
            console.log('Reset confirmation:', new TextDecoder().decode(value));
            
            messagesContainer.innerHTML = '';
            addMessage(`Ready to chat with the selected document!`, 'bot');
        } catch (error) {
            console.error('Error resetting chat:', error);
            addMessage('Could not reset chat session.', 'bot');
        }
    };

    const handleDocumentSelection = async (event) => {
        selectedDocumentId = event.target.value;
        if (selectedDocumentId) {
            selectionStatus.textContent = `Selected: ${event.target.options[event.target.selectedIndex].text}`;
            await resetChatForDocument(selectedDocumentId);
            messageInput.focus();
        } else {
            selectionStatus.textContent = '';
            messagesContainer.innerHTML = '';
            addMessage('Hello! Please upload and select a document to begin chatting.', 'bot');
        }
    };

    const handleChatSubmit = async (event) => {
        event.preventDefault();
        const query = messageInput.value.trim();
        if (!query) return;

        let payload = { user_id: USER_ID, query: query }
        if (selectedDocumentId) {
            payload.document_id = selectedDocumentId;
        }

        addMessage(query, 'user');
        messageInput.value = '';

        const botMessageElement = addMessage('', 'bot');
        botMessageElement.classList.add('thinking');
        const botParagraph = botMessageElement.querySelector('p');

        try {
            const response = await fetch(`${API_BASE_URL}/chat/stream/`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let accumulatedText = '';

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                
                accumulatedText += decoder.decode(value, { stream: true });
                botParagraph.innerHTML = parseAndRenderText(accumulatedText);
                messagesContainer.scrollTop = messagesContainer.scrollHeight;
            }
            
        } catch (error) {
            console.error('Streaming error:', error);
            botParagraph.textContent = 'Sorry, an error occurred while getting a response.';
        } finally {
            botMessageElement.classList.remove('thinking');
            messageInput.focus();
        }
    };

    // --- INITIALIZATION ---
    uploadForm.addEventListener('submit', handleFileUpload);
    docSelector.addEventListener('change', handleDocumentSelection);
    chatForm.addEventListener('submit', handleChatSubmit);
    themeToggle.addEventListener('change', handleThemeToggle);

    const savedTheme = localStorage.getItem('theme') || 'light';
    applyTheme(savedTheme);
    
    fetchDocuments();
});
