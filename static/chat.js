const messages = document.querySelector('#messages');
const form = document.querySelector('#chatForm');
const input = document.querySelector('#questionInput');
const sendButton = document.querySelector('#sendButton');
const cancelButton = document.querySelector('#cancelButton');
const healthBadge = document.querySelector('#healthBadge');
const history = [];
const basePath = document.body.dataset.basePath || '';
let activeController = null;

function appUrl(path) {
  const normalized = path.startsWith('/') ? path : `/${path}`;
  return `${basePath}${normalized}`;
}

function escapeHtml(value = '') {
  return String(value).replace(/[&<>'"]/g, (char) => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;'
  }[char]));
}

function formatText(value = '') {
  return escapeHtml(value)
    .replace(/\[([SM])(\d+)\]/g, '<strong class="citation">[$1$2]</strong>')
    .replace(/\n/g, '<br />');
}

function shouldShowLimitation(data) {
  const reason = String(data.answer?.refusal_reason || '').replace(/\s+/g, ' ').trim();
  const summary = String(data.answer?.summary || '').replace(/\s+/g, ' ').trim();
  if (!reason) return false;
  if (summary.includes(reason) || reason.includes(summary)) return false;
  if (data.status === 'partial_answer') return false;
  return !['policy_boundary', 'clarification', 'service_error', 'scope_only']
    .includes(data.fallback_kind);
}

function scrollBottom() {
  messages.scrollTop = messages.scrollHeight;
}

function appendUserMessage(text) {
  const article = document.createElement('article');
  article.className = 'message user';
  article.innerHTML = `<div class="message-content"><p>${formatText(text)}</p></div>`;
  messages.appendChild(article);
  scrollBottom();
}

function appendLoading() {
  const article = document.createElement('article');
  article.className = 'message assistant';
  article.id = 'loadingMessage';
  article.innerHTML = '<div class="avatar">AI</div><div class="message-content"><span id="loadingLabel" class="message-label">正在发送请求</span><div class="typing"><i></i><i></i><i></i></div></div>';
  messages.appendChild(article);
  scrollBottom();
}

function updateLoading(message) {
  const label = document.querySelector('#loadingLabel');
  if (label) label.textContent = message;
}

function sourceLocation(source) {
  const parts = [];
  if (source.page) parts.push(`第 ${source.page} 页`);
  if (source.sheet) parts.push(`工作表 ${source.sheet}`);
  if (source.row_start) parts.push(`第 ${source.row_start}${source.row_end && source.row_end !== source.row_start ? `-${source.row_end}` : ''} 行`);
  return parts.join(' · ') || '位置未标注';
}

function imageLocation(image) {
  const parts = [];
  if (image.page) parts.push(`第 ${image.page} 页`);
  if (image.sheet) parts.push(`工作表 ${image.sheet}`);
  if (image.row_start) parts.push(`第 ${image.row_start}${image.row_end && image.row_end !== image.row_start ? `-${image.row_end}` : ''} 行`);
  return parts.join(' · ');
}

function appendAssistantMessage(data) {
  document.querySelector('#loadingMessage')?.remove();
  const limitationTitle = data.status === 'insufficient_evidence'
    ? '证据校验未通过'
    : data.status === 'service_unavailable'
      ? '服务暂不可用'
      : '拒绝或限制回答的原因';
  const refusalReason = shouldShowLimitation(data) ? `
    <aside class="refusal-reason">
      <strong>${escapeHtml(limitationTitle)}</strong>
      <p>${formatText(data.answer.refusal_reason)}</p>
    </aside>` : '';
  const sections = (data.answer.sections || []).map((section) => `
    <div class="message-section">
      <h3>${escapeHtml(section.heading)}</h3>
      <p>${formatText(section.content)}</p>
    </div>`).join('');
  const stageLabels = { intent: '意图', retrieval: '检索', evidence: '证据', conclusion: '结论' };
  const statusLabels = {
    grounded: '证据支持',
    general_knowledge: '模型通用知识',
    partial_answer: '仅回答允许部分',
    out_of_scope: '超出服务范围',
    clarification_needed: '需要补充信息',
    insufficient_evidence: '证据不足',
    service_unavailable: '服务暂不可用',
    conversational: '对话'
  };
  const fallbackLabels = {
    policy_boundary: '服务范围说明',
    clarification: '需要补充信息',
    missing_evidence: '资料不足',
    service_error: '服务暂不可用',
    scope_only: '仅说明服务范围',
    verified_extractive: '已从审核资料提取回答'
  };
  const evidenceLabels = { supported: '已支持', partial: '部分支持', missing: '缺少证据', not_applicable: '无需检索', model_knowledge: '模型自身知识 · 无文档引用' };
  const analysis = (data.thinking_summary || []).map((step) =>
    `<li><strong>${escapeHtml(stageLabels[step.stage] || step.stage)}</strong> · ${escapeHtml(step.summary)}</li>`
  ).join('');
  const sources = (data.sources || []).map((source) => `
    <li>
      <span class="source-meta">${escapeHtml(source.source_id)} · ${source.source_kind === 'media' ? '图片来源' : '文字来源'} · ${escapeHtml(source.document_title)} · ${escapeHtml(sourceLocation(source))}</span>
      ${escapeHtml(source.excerpt)}
    </li>`).join('');
  const images = (data.images || []).map((item) => {
    const location = imageLocation(item);
    const source = [item.document_title, location, item.source_id].filter(Boolean).join(' · ');
    const imageUrl = appUrl(item.url);
    return `
      <figure class="response-image-card">
        <a href="${escapeHtml(imageUrl)}" target="_blank" rel="noopener" aria-label="查看原图：${escapeHtml(item.alt_text)}">
          <img src="${escapeHtml(imageUrl)}" alt="${escapeHtml(item.alt_text)}" loading="lazy" decoding="async" />
        </a>
        <figcaption>
          <strong>${escapeHtml(item.caption || item.alt_text)}</strong>
          <span>${escapeHtml(source)}</span>
        </figcaption>
      </figure>`;
  }).join('');
  const followups = (data.follow_up_questions || []).map((question) =>
    `<button type="button" data-followup="${escapeHtml(question)}">${escapeHtml(question)}</button>`
  ).join('');
  const article = document.createElement('article');
  article.className = 'message assistant';
  const responseLabel = fallbackLabels[data.fallback_kind]
    || statusLabels[data.status]
    || data.status;
  const technicalDetails = `
    <details class="analysis-card debug-card">
      <summary>技术详情</summary>
      <div class="confidence">证据状态 ${escapeHtml(evidenceLabels[data.evidence_status] || data.evidence_status)} · 策略 ${escapeHtml(data.decision?.policy_version || '-')} / ${escapeHtml(data.decision?.reason_code || '-')} · 检索 ${data.metrics?.retrieval_ms || 0}ms · 生成 ${data.metrics?.generation_ms || 0}ms</div>
      ${analysis ? `<ol class="analysis-list">${analysis}</ol>` : ''}
    </details>`;
  article.innerHTML = `
    <div class="avatar">AI</div>
    <div class="message-content">
      <span class="message-label">智能助手 · ${escapeHtml(responseLabel)}</span>
      <h2>${escapeHtml(data.answer.title)}</h2>
      <p>${formatText(data.answer.summary)}</p>
      ${refusalReason}
      ${sections}
      ${images ? `<section class="response-media"><h3>产品图片</h3><div class="response-image-grid">${images}</div></section>` : ''}
      ${sources ? `<details class="sources-card" open><summary>资料来源（${data.sources.length}）</summary><ol class="source-list">${sources}</ol></details>` : ''}
      ${technicalDetails}
      ${followups ? `<div class="followups">${followups}</div>` : ''}
    </div>`;
  messages.appendChild(article);
  const sectionHistory = (data.answer.sections || [])
    .map((section) => `${section.heading}\n${section.content}`).join('\n');
  const imageHistory = (data.images || [])
    .map((item) => `图片：${item.caption || item.alt_text}${item.product_ids?.length ? `（${item.product_ids.join('、')}）` : ''}`).join('\n');
  history.push({
    role: 'assistant',
    content: [data.answer.title, data.answer.summary, sectionHistory, imageHistory]
      .filter(Boolean).join('\n').slice(0, 8000)
  });
  history.splice(0, Math.max(0, history.length - 10));
  scrollBottom();
}

function appendError(message) {
  document.querySelector('#loadingMessage')?.remove();
  const article = document.createElement('article');
  article.className = 'message assistant';
  article.innerHTML = `<div class="avatar">!</div><div class="message-content"><span class="message-label">请求失败</span><p>${escapeHtml(message)}</p></div>`;
  messages.appendChild(article);
  scrollBottom();
}

function apiErrorMessage(data, response) {
  const detail = data?.detail;
  const base = typeof detail === 'string' ? detail : detail?.message;
  const retryAfter = Number(response.headers.get('Retry-After') || 0);
  if (retryAfter > 0) return `${base || '服务暂时繁忙'}（约 ${retryAfter} 秒后可重试）`;
  return base || '本地模型请求失败';
}

function friendlyNetworkError(error) {
  const message = String(error?.message || '');
  if (/load failed|failed to fetch|networkerror|network request failed/i.test(message)) {
    return '无法连接服务，请检查网络后重试。';
  }
  return message || '无法连接服务';
}

async function readChatEvents(response) {
  if (!response.body) throw new Error('服务未返回进度流，请重试。');
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let completed = false;
  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      buffer = buffer.replace(/\r\n/g, '\n');
      let boundary = buffer.indexOf('\n\n');
      while (boundary !== -1) {
        const block = buffer.slice(0, boundary);
        buffer = buffer.slice(boundary + 2);
        const eventLine = block.split('\n').find((line) => line.startsWith('event: '));
        const dataLines = block.split('\n').filter((line) => line.startsWith('data: '));
        if (eventLine && dataLines.length) {
          const eventName = eventLine.slice(7).trim();
          const data = JSON.parse(dataLines.map((line) => line.slice(6)).join('\n'));
          if (eventName === 'progress') updateLoading(data.message);
          if (eventName === 'error') throw new Error(data.message || '本次请求未能完成。');
          if (eventName === 'result') {
            appendAssistantMessage(data);
            completed = true;
          }
        }
        boundary = buffer.indexOf('\n\n');
      }
    }
  } finally {
    reader.releaseLock();
  }
  if (!completed) throw new Error('连接已结束，但服务未返回完整回答。');
}

async function ask(question) {
  const text = question.trim();
  if (!text || sendButton.disabled) return;
  appendUserMessage(text);
  history.push({ role: 'user', content: text });
  input.value = '';
  input.style.height = 'auto';
  sendButton.disabled = true;
  cancelButton.hidden = false;
  appendLoading();
  const controller = new AbortController();
  activeController = controller;
  try {
    const response = await fetch(appUrl('/api/chat/stream'), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' },
      body: JSON.stringify({ question: text, history: history.slice(0, -1) }),
      signal: controller.signal
    });
    if (!response.ok) {
      let data = {};
      try { data = await response.json(); } catch { /* Use HTTP status below. */ }
      throw new Error(apiErrorMessage(data, response) || `HTTP ${response.status}`);
    }
    await readChatEvents(response);
  } catch (error) {
    const lastHistoryItem = history.at(-1);
    if (lastHistoryItem?.role === 'user' && lastHistoryItem.content === text) history.pop();
    if (error.name === 'AbortError') {
      appendError('已取消本次请求，服务端不会继续处理后续步骤。');
    } else {
      appendError(friendlyNetworkError(error));
    }
  } finally {
    if (activeController === controller) activeController = null;
    cancelButton.hidden = true;
    sendButton.disabled = false;
    input.focus();
  }
}

form.addEventListener('submit', (event) => { event.preventDefault(); ask(input.value); });
cancelButton.addEventListener('click', () => activeController?.abort());
input.addEventListener('keydown', (event) => {
  if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); form.requestSubmit(); }
});
input.addEventListener('input', () => {
  input.style.height = 'auto';
  input.style.height = `${Math.min(input.scrollHeight, 150)}px`;
});
document.addEventListener('click', (event) => {
  const question = event.target.dataset?.question || event.target.dataset?.followup;
  if (question) ask(question);
});

async function checkHealth() {
  try {
    const response = await fetch(appUrl('/api/health'));
    const data = await response.json();
    const ready = data.ollama === 'ok' && data.chat_model_ready && data.embedding_model_ready;
    healthBadge.className = `health-badge ${ready ? 'ready' : 'error'}`;
    healthBadge.innerHTML = `<i></i>${ready ? `${data.knowledge.documents} 份文档` : '模型未就绪'}`;
  } catch {
    healthBadge.className = 'health-badge error';
    healthBadge.innerHTML = '<i></i>服务不可用';
  }
}
checkHealth();
