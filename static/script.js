let currentBotMsg = null;

function addMessage(text, type) {
let box = document.getElementById("chat-box");


let div = document.createElement("div");
div.className = "message " + type;
div.innerText = text;

box.appendChild(div);
box.scrollTop = box.scrollHeight;

return div;


}

// typing dots
function showLoader() {
return addMessage("...", "bot");
}

function send() {
let input = document.getElementById("msg");
let msg = input.value;
// let mode = document.getElementById("mode").value;



if (!msg) return;

addMessage(msg, "user");
input.value = "";

currentBotMsg = showLoader();

fetch("/chat", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({
        message: msg,
        // mode: mode
    })
})
.then(res => res.json())
.then(data => {

    // IMAGE MODE
    if (data.image) {
        currentBotMsg.innerHTML = `
            <img src="${data.image}" 
                 style="max-width:100%; border-radius:10px;">
        `;
        return;
    }

    // TEXT MODE SAFE CHECK
    if (data.answer && data.answer.length > 0) {
       
        if (data.image) {
            currentBotMsg.innerHTML = `<img src="${data.image}" style="max-width:100%">`;
            speak("Here is your generated image");
        }
         typeEffect(data.answer);
        speak(data.answer);
    }
     else {
        currentBotMsg.innerText = "⚠️ No response from AI";
    }

})
.catch(err => {
    currentBotMsg.innerText = "❌ Error: " + err.message;
});

}



// typing animation
function typeEffect(text) {
currentBotMsg.innerText = "";
let i = 0;


function typing() {
    if (i < text.length) {
        currentBotMsg.innerText += text.charAt(i);
        i++;
        setTimeout(typing, 15);
    }
}

typing();


}

function newChat() {
document.getElementById("chat-box").innerHTML = "";
}


const micBtn = document.getElementById("micBtn");

const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

if (SpeechRecognition) {
    const recognition = new SpeechRecognition();
    recognition.continuous = false;
    recognition.lang = "en-US";

    micBtn.onclick = () => {
        recognition.start();
        micBtn.innerText = "🎙️ Listening...";
    };

    recognition.onresult = (event) => {
        const text = event.results[0][0].transcript;
        document.getElementById("messageInput").value = text;
        micBtn.innerText = "🎤";
    };

    recognition.onerror = () => {
        micBtn.innerText = "🎤";
    };
}

function speak(text) {
    const speech = new SpeechSynthesisUtterance(text);
    // speech.lang = "en-US";
    speech.lang = "hi-IN";
    speech.rate = 1;
    speech.pitch = 1;

    window.speechSynthesis.speak(speech);
}