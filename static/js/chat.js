marked.setOptions({ breaks: true, gfm: true });

let currentImageData = null;
let currentImageMime = null;
let currentIsPdf = false;

function initializeChat() {
    loadChatHistory();
    setupEventListeners();
}

async function loadChatHistory() {
    try {
        const res = await fetch("/history");
        const data = await res.json();
        if (data.length > 0) {
            const welcome = document.querySelector(".welcome");
            if (welcome) welcome.remove();
            data.forEach(msg => appendMessage(msg.role, msg.content));
        }
    } catch (e) {
        console.error("Failed to load chat history:", e);
    }
}

function setupEventListeners() {
    const userInput = document.getElementById("user-input");
    if (userInput) {
        userInput.addEventListener("keydown", function (e) {
            if (e.key === "Enter") sendMessage();
        });
    }
}

function handleFileUpload(event) {
    const file = event.target.files[0];
    if (!file) return;

    currentImageMime = file.type;
    currentIsPdf = file.type === "application/pdf";

    const preview = document.getElementById("image-preview");
    const img = document.getElementById("preview-img");
    const name = document.getElementById("preview-name");

    if (currentIsPdf) {
        img.style.display = "none";
        name.textContent = `📄 ${file.name}`;
        name.style.display = "inline";
        preview.style.display = "flex";

        const reader = new FileReader();
        reader.onload = function (e) {
            currentImageData = e.target.result.split(",")[1];
        };
        reader.readAsDataURL(file);
    } else {
        img.style.display = "block";
        name.style.display = "none";
        preview.style.display = "flex";

        const reader = new FileReader();
        reader.onload = function (e) {
            img.src = e.target.result;
            currentImageData = e.target.result.split(",")[1];
        };
        reader.readAsDataURL(file);
    }
}

function clearImage() {
    currentImageData = null;
    currentImageMime = null;
    currentIsPdf = false;
    document.getElementById("image-preview").style.display = "none";
    document.getElementById("preview-img").src = "";
    document.getElementById("preview-name").textContent = "";
    document.getElementById("file-input").value = "";
}

async function sendMessage() {
    const input = document.getElementById("user-input");
    const message = input.value.trim();
    if (!message && !currentImageData) return;

    const welcome = document.querySelector(".welcome");
    if (welcome) welcome.remove();

    input.value = "";
    const sendBtn = document.getElementById("send-btn");
    sendBtn.disabled = true;

    const previewSrc = currentImageData && !currentIsPdf
        ? document.getElementById("preview-img").src
        : null;
    const pdfName = currentIsPdf
        ? document.getElementById("preview-name").textContent
        : null;

    appendMessage("user", message, previewSrc, pdfName);
    const thinking = appendThinking();

    const payload = {
        message: message || (currentIsPdf ? "Analyse this document" : "What is in this image?"),
        image_data: currentImageData,
        image_mime: currentImageMime,
    };

    clearImage();

    try {
        const res = await fetch("/chat/stream", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });

        thinking.remove();

        const box = document.getElementById("chat-box");
        const wrapper = document.createElement("div");
        wrapper.className = "message-wrapper assistant";

        const avatar = document.createElement("div");
        avatar.className = "avatar";
        avatar.textContent = "F";
        wrapper.appendChild(avatar);

        const bubble = document.createElement("div");
        bubble.className = "bubble assistant";

        const textDiv = document.createElement("div");
        bubble.appendChild(textDiv);
        wrapper.appendChild(bubble);
        box.appendChild(wrapper);

        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let fullText = "";

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            const text = decoder.decode(value);
            const lines = text.split("\n");

            for (const line of lines) {
                if (line.startsWith("data: ")) {
                    const chunk = line.slice(6);
                    if (chunk === "[DONE]") break;
                    fullText += chunk;
                    textDiv.innerHTML = marked.parse(fullText);
                    box.scrollTop = box.scrollHeight;
                }
            }
        }

    } catch (err) {
        console.error("Error sending message:", err);
        thinking.remove();
        appendMessage("assistant", "Something went wrong. Please try again.");
    }

    sendBtn.disabled = false;
    input.focus();
}

function appendMessage(role, text, imageSrc = null, pdfName = null) {
    const box = document.getElementById("chat-box");
    const wrapper = document.createElement("div");
    wrapper.className = `message-wrapper ${role}`;

    if (role === "assistant") {
        const avatar = document.createElement("div");
        avatar.className = "avatar";
        avatar.textContent = "F";
        wrapper.appendChild(avatar);
    }

    const bubble = document.createElement("div");
    bubble.className = `bubble ${role}`;

    if (imageSrc) {
        const img = document.createElement("img");
        img.src = imageSrc;
        img.className = "chat-image";
        bubble.appendChild(img);
    }

    if (pdfName) {
        const pdfDiv = document.createElement("div");
        pdfDiv.className = "chat-pdf";
        pdfDiv.textContent = pdfName;
        bubble.appendChild(pdfDiv);
    }

    if (text) {
        const textDiv = document.createElement("div");
        if (role === "assistant") {
            textDiv.innerHTML = marked.parse(text);
        } else {
            textDiv.textContent = text;
        }
        bubble.appendChild(textDiv);
    }

    wrapper.appendChild(bubble);
    box.appendChild(wrapper);
    box.scrollTop = box.scrollHeight;
    return wrapper;
}

function appendThinking() {
    const box = document.getElementById("chat-box");
    const wrapper = document.createElement("div");
    wrapper.className = "message-wrapper assistant";

    const avatar = document.createElement("div");
    avatar.className = "avatar";
    avatar.textContent = "F";
    wrapper.appendChild(avatar);

    const bubble = document.createElement("div");
    bubble.className = "bubble assistant thinking";
    bubble.innerHTML = `
        <div class="thinking-dots">
            <span></span><span></span><span></span>
        </div>
        <span class="thinking-text">Querying 13 models...</span>
    `;
    wrapper.appendChild(bubble);
    box.appendChild(wrapper);
    box.scrollTop = box.scrollHeight;
    return wrapper;
}

async function resetChat() {
    try {
        await fetch("/reset", { method: "POST" });
    } catch (e) {
        console.error("Reset failed:", e);
    }

    clearImage();
    const box = document.getElementById("chat-box");
    box.innerHTML = `
        <div class="welcome">
            <div class="welcome-icon">⚡</div>
            <h2>What can I help you with?</h2>
            <p>Powered by 13 AI models working in parallel to give you the best answer.</p>
            <div class="provider-chips">
                <span>Groq</span><span>Gemini</span><span>Mistral</span>
                <span>Cerebras</span><span>Cohere</span><span>Nvidia</span>
                <span>SambaNova</span><span>OpenRouter</span>
            </div>
        </div>`;
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializeChat);
} else {
    initializeChat();
}