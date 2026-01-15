const form = document.getElementById("chatForm");
const input = document.getElementById("chatInput");
const messages = document.getElementById("chatMessages");
const clearBtn = document.querySelector("[data-action='clear']");

function appendMessage(role, text) {
  const wrapper = document.createElement("div");
  wrapper.className = `message message--${role}`;
  const bubble = document.createElement("div");
  bubble.className = "message__bubble";
  bubble.textContent = text;
  wrapper.appendChild(bubble);
  messages.appendChild(wrapper);
  messages.scrollTop = messages.scrollHeight;
}

async function sendMessage(text) {
  appendMessage("user", text);
  appendMessage("assistant", "正在生成回答...");
  const placeholder = messages.lastElementChild;
  try {
    const resp = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text })
    });
    const data = await resp.json();
    if (!resp.ok) {
      throw new Error(data.error || "请求失败");
    }
    placeholder.querySelector(".message__bubble").textContent = data.answer;
  } catch (err) {
    placeholder.querySelector(".message__bubble").textContent = "请求失败，请稍后再试。";
  }
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  const text = input.value.trim();
  if (!text) return;
  input.value = "";
  sendMessage(text);
});

clearBtn.addEventListener("click", () => {
  messages.innerHTML = "";
  appendMessage("assistant", "你好！我可以根据文档内容回答问题。");
});
