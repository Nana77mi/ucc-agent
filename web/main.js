const form = document.getElementById("chatForm");
const input = document.getElementById("chatInput");
const messages = document.getElementById("chatMessages");
const clearBtn = document.querySelector("[data-action='clear']");
const docList = document.getElementById("docList");
const docSearch = document.getElementById("docSearch");
const docTitle = document.getElementById("docTitle");
const docSubtitle = document.getElementById("docSubtitle");
const docMeta = document.getElementById("docMeta");
const docContent = document.getElementById("docContent");

let docs = [];
let activeDocId = null;

function escapeHtml(value) {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function renderDocList(filterText = "") {
  if (!docList) return;
  const keyword = filterText.trim().toLowerCase();
  const filtered = docs.filter((doc) =>
    [doc.title, doc.summary, (doc.tags || []).join(" ")].join(" ").toLowerCase().includes(keyword)
  );
  docList.innerHTML = "";
  filtered.forEach((doc) => {
    const item = document.createElement("button");
    item.type = "button";
    item.className = `docs__item${doc.id === activeDocId ? " docs__item--active" : ""}`;
    item.innerHTML = `<h3>${escapeHtml(doc.title)}</h3><p>${escapeHtml(doc.summary)}</p>`;
    item.addEventListener("click", () => {
      activeDocId = doc.id;
      renderDocList(docSearch?.value || "");
      renderDocDetail(doc);
    });
    docList.appendChild(item);
  });
  if (!filtered.length) {
    const empty = document.createElement("div");
    empty.className = "status";
    empty.textContent = "暂无匹配文档";
    docList.appendChild(empty);
  } else if (!filtered.find((doc) => doc.id === activeDocId)) {
    activeDocId = filtered[0].id;
    renderDocDetail(filtered[0]);
  }
}

function renderDocDetail(doc) {
  if (!doc) return;
  if (docTitle) docTitle.textContent = doc.title || "未命名文档";
  if (docSubtitle) docSubtitle.textContent = doc.subtitle || "暂无补充信息";
  if (docMeta) {
    const tags = doc.tags || [];
    docMeta.innerHTML = tags.map((tag) => `<span>#${escapeHtml(tag)}</span>`).join("");
  }
  if (docContent) {
    const sections = doc.sections || [];
    if (!sections.length) {
      docContent.innerHTML = "<p class=\"status\">暂无可展示内容。</p>";
      return;
    }
    docContent.innerHTML = sections
      .map((section) => {
        const title = section.title ? `<strong>${escapeHtml(section.title)}</strong>` : "";
        const text = section.text ? `<p>${escapeHtml(section.text)}</p>` : "";
        const items = Array.isArray(section.items)
          ? `<ul>${section.items.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul>`
          : "";
        const code = Array.isArray(section.code)
          ? `<pre><code>${escapeHtml(section.code.join("\n"))}</code></pre>`
          : "";
        return `<div>${title}${text}${items}${code}</div>`;
      })
      .join("");
  }
}

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

async function loadDocs() {
  if (!docList) return;
  docList.innerHTML = "<div class=\"status\">正在加载文档...</div>";
  try {
    const resp = await fetch("/web/docs.json");
    if (!resp.ok) {
      throw new Error("文档加载失败");
    }
    const data = await resp.json();
    docs = Array.isArray(data) ? data : data.docs || [];
    if (!docs.length) {
      docList.innerHTML = "<div class=\"status\">暂无文档数据</div>";
      return;
    }
    activeDocId = docs[0].id;
    renderDocList();
    renderDocDetail(docs.find((doc) => doc.id === activeDocId));
  } catch (err) {
    docList.innerHTML = "<div class=\"status\">文档加载失败，请检查 docs.json</div>";
  }
}

if (docSearch) {
  docSearch.addEventListener("input", (event) => {
    renderDocList(event.target.value);
  });
}

loadDocs();
