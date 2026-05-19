document.addEventListener('DOMContentLoaded', () => {
    const chatHistory = document.getElementById('chat-history');
    const chatForm = document.getElementById('chat-form');
    const userInput = document.getElementById('user-input');
    const sendBtn = document.getElementById('send-btn');
    const topKSlider = document.getElementById('top-k-slider');
    const topKValue = document.getElementById('top-k-value');

    // API URL
    const API_URL = 'http://127.0.0.1:8000/api/v1/rag/get_answer';

    // Auto-resize textarea
    userInput.addEventListener('input', function () {
        this.style.height = 'auto';
        this.style.height = (this.scrollHeight) + 'px';

        // Enable/disable send button
        if (this.value.trim().length > 0) {
            sendBtn.removeAttribute('disabled');
        } else {
            sendBtn.setAttribute('disabled', 'true');
        }
    });

    // Handle Enter key (Shift+Enter for new line)
    userInput.addEventListener('keydown', function (e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            if (this.value.trim().length > 0) {
                chatForm.dispatchEvent(new Event('submit'));
            }
        }
    });

    // Handle Slider
    topKSlider.addEventListener('input', function () {
        topKValue.textContent = this.value;
    });

    // Handle form submit
    chatForm.addEventListener('submit', async (e) => {
        e.preventDefault();

        const query = userInput.value.trim();
        if (!query) return;

        const topK = parseInt(topKSlider.value, 10);

        // Reset input
        userInput.value = '';
        userInput.style.height = 'auto';
        sendBtn.setAttribute('disabled', 'true');

        // Add user message
        appendMessage(query, 'user');

        // Show typing indicator
        const typingId = showTypingIndicator();

        try {
            // Check health first or assume it's working
            const response = await fetch(API_URL, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    query: query,
                    top_k: topK
                })
            });

            const data = await response.json();

            // Remove typing indicator
            removeElement(typingId);

            if (response.ok && data.success) {
                appendMessage(data.answer, 'bot');
            } else {
                appendMessage(`Error: ${data.message || 'Something went wrong.'}`, 'bot', true);
            }

        } catch (error) {
            removeElement(typingId);
            appendMessage(`Connection Error: Make sure the FastAPI server is running on localhost:8000. Detail: ${error.message}`, 'bot', true);
        }
    });

    function appendMessage(text, sender, isError = false) {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message', `${sender}-message`);

        const contentDiv = document.createElement('div');
        contentDiv.classList.add('message-content');
        if (isError) {
            contentDiv.style.borderLeft = '3px solid #ef4444';
            contentDiv.style.color = '#fca5a5';
        }

        // Format newlines to <p> tags or <br>
        const formattedText = text.replace(/\n/g, '<br>');
        contentDiv.innerHTML = formattedText;

        messageDiv.appendChild(contentDiv);
        chatHistory.appendChild(messageDiv);

        scrollToBottom();
    }

    function showTypingIndicator() {
        const id = 'typing-' + Date.now();
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message', 'bot-message');
        messageDiv.id = id;

        const contentDiv = document.createElement('div');
        contentDiv.classList.add('message-content');

        contentDiv.innerHTML = `
            <div class="typing-indicator">
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
            </div>
        `;

        messageDiv.appendChild(contentDiv);
        chatHistory.appendChild(messageDiv);

        scrollToBottom();
        return id;
    }

    function removeElement(id) {
        const el = document.getElementById(id);
        if (el) {
            el.remove();
        }
    }

    function scrollToBottom() {
        chatHistory.scrollTo({
            top: chatHistory.scrollHeight,
            behavior: 'smooth'
        });
    }

    // Initial check for backend connection
    checkBackendHealth();

    async function checkBackendHealth() {
        const dot = document.querySelector('.status-indicator .dot');
        const statusText = document.querySelector('.status-indicator span');

        try {
            const response = await fetch('http://127.0.0.1:8000/api/v1/rag/health');
            if (response.ok) {
                dot.classList.add('active');
                statusText.textContent = 'System Online';
            } else {
                throw new Error('Not OK');
            }
        } catch (error) {
            dot.classList.remove('active');
            statusText.textContent = 'System Offline';
        }
    }

    // Poll health every 30 seconds
    setInterval(checkBackendHealth, 30000);
});
