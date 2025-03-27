// DOM Elements
const sendButton = document.getElementById('send-button');
const messageInput = document.getElementById('message-input');
const chatHistory = document.getElementById('chat-history');
const uploadButton = document.getElementById('upload-button');
const fileInput = document.getElementById('file-input');
const uploadOverlay = document.getElementById('upload-overlay');

// Event Listeners for Chat
sendButton.addEventListener('click', sendMessage);
messageInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        e.preventDefault(); // Prevent form submission if inside a form
        sendMessage();
    }
});

// Event Listeners for File Upload
uploadButton.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', () => {
    const file = fileInput.files[0];
    if (file) uploadPDF(file);
    fileInput.value = ''; // Reset input
});

// Drag and Drop Events
document.body.addEventListener('dragenter', (e) => {
    e.preventDefault();
    uploadOverlay.classList.add('active');
});

document.body.addEventListener('dragover', (e) => {
    e.preventDefault();
});

document.body.addEventListener('dragleave', (e) => {
    if (e.relatedTarget === null) {
        uploadOverlay.classList.remove('active');
    }
});

document.body.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadOverlay.classList.remove('active');
    const file = e.dataTransfer.files[0];
    if (file) uploadPDF(file);
});

// Send Chat Message
async function sendMessage() {
    const message = messageInput.value.trim();
    if (!message) return;

    appendMessage('user', message);
    messageInput.value = '';
    const typingIndicator = appendTypingIndicator();

    try {
        const response = await fetch('/chat', {
            method: 'POST',
            body: new URLSearchParams({ message }),
            headers: { 'Content-Type': 'application/x-www-form-urlencoded' }
        });
        const data = await response.json();
        chatHistory.removeChild(typingIndicator);

        if (data.error) {
            appendMessage('bot', `Error: ${data.error}`, 'error');
        } else if (data.character) {
            appendCharacterMessage(data.character, data.response, data.emotion);
        } else {
            appendMessage('bot', data.response);
        }
    } catch (error) {
        chatHistory.removeChild(typingIndicator);
        appendMessage('bot', 'Error: Unable to connect to server', 'error');
    }
}

// Upload PDF
async function uploadPDF(file) {
    if (!file.type.includes('pdf')) {
        appendMessage('bot', 'Error: Please upload a PDF file', 'error');
        return;
    }

    const formData = new FormData();
    formData.append('pdf_file', file);
    appendMessage('bot', 'Uploading PDF...', 'info');

    try {
        const response = await fetch('/chat', {
            method: 'POST',
            body: formData
        });
        const data = await response.json();

        if (data.error) {
            appendMessage('bot', `Error: ${data.error}`, 'error');
        } else {
            displayCharacters(data.characters);
            appendMessage('bot', data.response, 'info');
        }
    } catch (error) {
        appendMessage('bot', 'Error: Unable to upload PDF', 'error');
    }
}

// Append Generic Message
function appendMessage(sender, text, type = '') {
    const messageDiv = document.createElement('div');
    messageDiv.classList.add('message', sender);
    if (type) messageDiv.classList.add(type);

    const textDiv = document.createElement('div');
    textDiv.classList.add('message-text');
    textDiv.textContent = text;
    messageDiv.appendChild(textDiv);

    const timestamp = document.createElement('div');
    timestamp.classList.add('timestamp');
    timestamp.textContent = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    messageDiv.appendChild(timestamp);

    chatHistory.appendChild(messageDiv);
    autoScroll();
}

// Append Character Message
function appendCharacterMessage(character, text, emotion) {
    const messageDiv = document.createElement('div');
    messageDiv.classList.add('message', 'bot', 'character');

    const infoDiv = document.createElement('div');
    infoDiv.classList.add('character-info');

    const nameSpan = document.createElement('span');
    nameSpan.classList.add('character-name');
    nameSpan.textContent = character;

    const emotionSpan = document.createElement('span');
    emotionSpan.classList.add('emotion');
    emotionSpan.textContent = `${getEmotionEmoji(emotion.emotion)} ${emotion.emotion}`;

    infoDiv.appendChild(nameSpan);
    infoDiv.appendChild(emotionSpan);

    const textDiv = document.createElement('div');
    textDiv.classList.add('message-text');
    textDiv.textContent = text;

    const timestamp = document.createElement('div');
    timestamp.classList.add('timestamp');
    timestamp.textContent = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

    messageDiv.appendChild(infoDiv);
    messageDiv.appendChild(textDiv);
    messageDiv.appendChild(timestamp);

    chatHistory.appendChild(messageDiv);
    autoScroll();
}

// Append Typing Indicator
function appendTypingIndicator() {
    const typingDiv = document.createElement('div');
    typingDiv.classList.add('message', 'bot', 'typing');

    const indicator = document.createElement('div');
    indicator.classList.add('typing-indicator');
    indicator.textContent = 'Bot is typing';

    typingDiv.appendChild(indicator);
    chatHistory.appendChild(typingDiv);
    autoScroll();
    return typingDiv;
}

// Display Characters
function displayCharacters(characters) {
    const charactersContent = document.getElementById('characters-content');
    if (!charactersContent) return; // Prevent errors if element is missing
    charactersContent.innerHTML = '<h2>Available Characters</h2>';
    const listDiv = document.createElement('div');
    listDiv.classList.add('character-list');

    characters.forEach(char => {
        const card = document.createElement('div');
        card.classList.add('character-card');
        card.innerHTML = `<h3>${char.name}</h3><p>${char.summary}</p>`;
        listDiv.appendChild(card);
    });

    charactersContent.appendChild(listDiv);
}

// Emotion Emoji Mapping
function getEmotionEmoji(emotion) {
    const emojis = {
        happy: '😊',
        sad: '😢',
        angry: '😠',
        surprised: '😲',
        fearful: '😨',
        disgusted: '🤢'
    };
    return emojis[emotion.toLowerCase()] || '😐';
}

// Auto-Scroll Logic
function autoScroll() {
    const isScrolledToBottom = chatHistory.scrollHeight - chatHistory.clientHeight <= chatHistory.scrollTop + 1;
    if (isScrolledToBottom) {
        chatHistory.scrollTop = chatHistory.scrollHeight;
    }
}