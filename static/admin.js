const documentList = document.querySelector('#documentList');
const fileInput = document.querySelector('#fileInput');
const dropZone = document.querySelector('#dropZone');
const uploadStatus = document.querySelector('#uploadStatus');
const refreshButton = document.querySelector('#refreshButton');
const healthBadge = document.querySelector('#healthBadge');
const toast = document.querySelector('#toast');
const authDialog = document.querySelector('#authDialog');
const authForm = document.querySelector('#authForm');
const adminTokenInput = document.querySelector('#adminToken');
const authError = document.querySelector('#authError');
const logoutButton = document.querySelector('#logoutButton');
const tokenKey = 'company-ai-agent-admin-token';
let adminToken = sessionStorage.getItem(tokenKey) || '';
const basePath = document.body.dataset.basePath || '';

function appUrl(path) {
  const normalized = path.startsWith('/') ? path : `/${path}`;
  return `${basePath}${normalized}`;
}

function escapeHtml(value = '') {
  return String(value).replace(/[&<>'"]/g, (char) => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;'
  }[char]));
}

function prettyBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 ** 2).toFixed(1)} MB`;
}

function showToast(message, error = false) {
  toast.textContent = message;
  toast.className = `toast show${error ? ' error' : ''}`;
  setTimeout(() => { toast.className = 'toast'; }, 2800);
}

function showAuth(message = '') {
  authError.textContent = message;
  authError.hidden = !message;
  if (!authDialog.open) authDialog.showModal();
  setTimeout(() => adminTokenInput.focus(), 0);
}

async function request(url, options = {}, authenticated = true) {
  const headers = new Headers(options.headers || {});
  if (authenticated && adminToken) headers.set('X-Admin-Token', adminToken);
  const response = await fetch(appUrl(url), { ...options, headers });
  let data = {};
  try { data = await response.json(); } catch { data = {}; }
  if (!response.ok) {
    const detail = data.detail;
    const message = typeof detail === 'string' ? detail : detail?.message;
    const code = typeof detail === 'object' ? detail?.code : '';
    const retryAfter = Number(response.headers.get('Retry-After') || 0);
    const rendered = retryAfter > 0
      ? `${message || '操作失败'}（约 ${retryAfter} 秒后可重试）`
      : (message || '操作失败');
    if (response.status === 401 || code === 'ADMIN_AUTH_RATE_LIMITED') {
      adminToken = '';
      sessionStorage.removeItem(tokenKey);
      logoutButton.hidden = true;
      showAuth(rendered || '管理员令牌无效');
    }
    throw new Error(rendered);
  }
  return data;
}

function domainOptions(selected) {
  const options = [
    ['company_public', '公司公开信息'],
    ['product_public', '自有产品信息']
  ];
  return options.map(([value, label]) =>
    `<option value="${value}" ${value === selected ? 'selected' : ''}>${label}</option>`
  ).join('');
}

function renderDocuments(data) {
  document.querySelector('#documentCount').textContent = data.stats.documents;
  document.querySelector('#chunkCount').textContent = data.stats.chunks;
  if (!data.items.length) {
    documentList.innerHTML = '<div class="empty-state">知识库为空，请上传第一份公司或产品资料。</div>';
    return;
  }
  documentList.innerHTML = data.items.map((item) => {
    const active = Number.isInteger(item.active_version) && !item.is_withdrawn;
    const pending = !active || item.source_version !== item.active_version || item.review_status !== 'approved';
    const stateLabel = item.is_withdrawn
      ? `已撤回 · 草稿 v${item.source_version}`
      : active && pending
        ? `公开 v${item.active_version} · 待审草稿 v${item.source_version}`
        : active ? `已审核发布 v${item.active_version}` : '尚未发布';
    const brandLabels = { gougelila: 'Gougelila', chongqin: 'ChongQin', 'ckin-pet': 'CKIN PET' };
    const ownership = item.knowledge_domain === 'product_public'
      ? ` · 品牌 ${brandLabels[item.brand_id] || '待设置'}${item.product_ids.length ? ` · ${item.product_ids.length} 个产品标识` : ''}`
      : '';
    return `
      <article class="document-row" data-id="${escapeHtml(item.id)}">
        <div class="document-main">
          <span class="file-badge">${escapeHtml(item.file_type)}</span>
          <div>
            <h3>${escapeHtml(item.title)}<span class="status-pill ${item.status.toLowerCase()}">${escapeHtml(item.status)}</span></h3>
            <p>${escapeHtml(item.filename)} · ${prettyBytes(item.size_bytes)} · ${item.chunk_count} 个片段 · ${item.media_count || 0} 张图片 · v${item.source_version}</p>
            <p class="publication-line"><strong>${stateLabel}</strong> · ${escapeHtml(item.visibility)}${ownership}</p>
          </div>
        </div>
        <div class="document-actions">
          <select data-domain aria-label="知识域">${domainOptions(item.knowledge_domain)}</select>
          <a class="button ghost" href="${escapeHtml(appUrl(`/admin/documents/${encodeURIComponent(item.id)}`))}">图片、片段与归属</a>
          <button class="button ghost" data-action="reindex" type="button">重算向量</button>
          <button class="button ${active && !pending ? 'ghost' : 'primary'}" data-action="${active && !pending ? 'withdraw' : 'publish'}" type="button">${active && !pending ? '撤回' : '审核并发布'}</button>
          <button class="button danger" data-action="delete" type="button">删除</button>
        </div>
      </article>`;
  }).join('');
}

async function loadDocuments() {
  if (!adminToken) {
    documentList.innerHTML = '<div class="empty-state">请先登录知识库后台。</div>';
    showAuth();
    return;
  }
  documentList.innerHTML = '<div class="empty-state">正在读取知识库……</div>';
  try {
    const data = await request('/api/admin/documents');
    logoutButton.hidden = false;
    renderDocuments(data);
  } catch (error) {
    documentList.innerHTML = `<div class="empty-state">${escapeHtml(error.message)}</div>`;
  }
}

async function uploadFiles(files) {
  if (!files.length || !adminToken) {
    if (!adminToken) showAuth();
    return;
  }
  uploadStatus.hidden = false;
  uploadStatus.textContent = `正在解析并建立索引：${files.length} 个文件。首次运行会加载 Embedding 模型，请耐心等待……`;
  const body = new FormData();
  [...files].forEach((file) => body.append('files', file));
  body.append('knowledge_domain', document.querySelector('#knowledgeDomain').value);
  try {
    const data = await request('/api/admin/documents', { method: 'POST', body });
    const success = data.items.filter((item) => item.ok).length;
    const errors = data.items.filter((item) => !item.ok).map((item) => `${item.filename}: ${item.error}`);
    uploadStatus.textContent = `完成：${success} 个成功，已进入草稿待审核${errors.length ? `；失败：${errors.join('；')}` : ''}`;
    showToast(`已导入 ${success} 个文件，请审核发布`, errors.length > 0);
    await loadDocuments();
    await checkHealth();
  } catch (error) {
    uploadStatus.textContent = `上传失败：${error.message}`;
    showToast(error.message, true);
  } finally {
    fileInput.value = '';
  }
}

documentList.addEventListener('change', async (event) => {
  const select = event.target.closest('select[data-domain]');
  if (!select) return;
  const row = select.closest('.document-row');
  select.disabled = true;
  try {
    await request(`/api/admin/documents/${row.dataset.id}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ knowledge_domain: select.value })
    });
    showToast('知识域已更新');
  } catch (error) {
    showToast(error.message, true);
    await loadDocuments();
  } finally {
    select.disabled = false;
  }
});

documentList.addEventListener('click', async (event) => {
  const button = event.target.closest('button[data-action]');
  if (!button) return;
  const row = button.closest('.document-row');
  const documentId = row.dataset.id;
  const action = button.dataset.action;
  if (action === 'delete' && !window.confirm('确认删除该文档？原文件会移动到本地回收目录。')) return;
  button.disabled = true;
  try {
    if (action === 'delete') {
      await request(`/api/admin/documents/${documentId}`, { method: 'DELETE' });
      showToast('文档已移入回收目录');
    } else if (action === 'reindex') {
      showToast('正在按当前运营片段重算向量……');
      await request(`/api/admin/documents/${documentId}/reindex`, { method: 'POST' });
      showToast('向量重算完成并进入待审核草稿，当前公开版本未改变');
    } else if (action === 'publish') {
      await request(`/api/admin/documents/${documentId}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ visibility: 'public', review_status: 'approved', publication_status: 'published' })
      });
      showToast('文档已审核发布，可参与公开问答');
    } else if (action === 'withdraw') {
      await request(`/api/admin/documents/${documentId}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ publication_status: 'withdrawn' })
      });
      showToast('文档已撤回，不再参与问答');
    }
    await loadDocuments();
  } catch (error) {
    showToast(error.message, true);
  } finally {
    button.disabled = false;
  }
});

authForm.addEventListener('submit', async (event) => {
  event.preventDefault();
  adminToken = adminTokenInput.value.trim();
  sessionStorage.setItem(tokenKey, adminToken);
  try {
    const data = await request('/api/admin/documents');
    authDialog.close();
    adminTokenInput.value = '';
    logoutButton.hidden = false;
    renderDocuments(data);
  } catch (error) {
    authError.textContent = error.message;
    authError.hidden = false;
  }
});
authDialog.addEventListener('cancel', (event) => event.preventDefault());
logoutButton.addEventListener('click', () => {
  adminToken = '';
  sessionStorage.removeItem(tokenKey);
  logoutButton.hidden = true;
  showAuth('已退出后台');
});

dropZone.addEventListener('click', (event) => {
  if (!event.target.closest('input')) fileInput.click();
});
dropZone.addEventListener('keydown', (event) => {
  if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); fileInput.click(); }
});
['dragenter', 'dragover'].forEach((name) => dropZone.addEventListener(name, (event) => {
  event.preventDefault(); dropZone.classList.add('dragging');
}));
['dragleave', 'drop'].forEach((name) => dropZone.addEventListener(name, (event) => {
  event.preventDefault(); dropZone.classList.remove('dragging');
}));
dropZone.addEventListener('drop', (event) => uploadFiles(event.dataTransfer.files));
fileInput.addEventListener('change', () => uploadFiles(fileInput.files));
refreshButton.addEventListener('click', loadDocuments);

async function checkHealth() {
  try {
    const data = await request('/api/health', {}, false);
    const indexingReady = data.ollama === 'ok' && data.embedding_model_ready;
    const ready = indexingReady && data.chat_model_ready;
    const label = ready ? '本地模型已就绪' : indexingReady ? '索引就绪 · 对话模型下载中' : '模型下载中/未就绪';
    healthBadge.className = `health-badge ${ready ? 'ready' : 'error'}`;
    healthBadge.innerHTML = `<i></i>${label}`;
  } catch {
    healthBadge.className = 'health-badge error';
    healthBadge.innerHTML = '<i></i>服务不可用';
  }
}

loadDocuments();
checkHealth();
