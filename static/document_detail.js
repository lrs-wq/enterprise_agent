const documentId = document.body.dataset.documentId;
const tokenKey = 'company-ai-agent-admin-token';
let adminToken = sessionStorage.getItem(tokenKey) || '';
const basePath = document.body.dataset.basePath || '';

function appUrl(path) {
  const normalized = path.startsWith('/') ? path : `/${path}`;
  return `${basePath}${normalized}`;
}
let currentDocument = null;
let currentChunks = [];
let currentMedia = [];
let currentVersions = [];
let publicationEvents = [];
let previewUrls = [];

const chunkList = document.querySelector('#chunkList');
const mediaGrid = document.querySelector('#mediaGrid');
const versionList = document.querySelector('#versionList');
const toast = document.querySelector('#toast');
const authDialog = document.querySelector('#authDialog');
const authForm = document.querySelector('#authForm');
const adminTokenInput = document.querySelector('#adminToken');
const authError = document.querySelector('#authError');
const logoutButton = document.querySelector('#logoutButton');

function escapeHtml(value = '') {
  return String(value).replace(/[&<>'"]/g, (char) => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;'
  }[char]));
}

function showToast(message, error = false) {
  toast.textContent = message;
  toast.className = `toast show${error ? ' error' : ''}`;
  setTimeout(() => { toast.className = 'toast'; }, 3000);
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

function locationLabel(chunk) {
  const parts = [];
  if (chunk.page) parts.push(`PDF 第 ${chunk.page} 页`);
  if (chunk.sheet) parts.push(`工作表 ${chunk.sheet}`);
  if (chunk.row_start) {
    const rows = chunk.row_end && chunk.row_end !== chunk.row_start
      ? `${chunk.row_start}-${chunk.row_end}` : chunk.row_start;
    parts.push(`第 ${rows} 行`);
  }
  return parts.join(' · ') || '文本片段';
}

function mediaLocation(asset) {
  const parts = [];
  if (asset.page) parts.push(`PDF 第 ${asset.page} 页`);
  if (asset.sheet) parts.push(`工作表 ${asset.sheet}`);
  if (asset.cell_ref) parts.push(`单元格 ${asset.cell_ref}`);
  else if (asset.row_start) parts.push(`第 ${asset.row_start} 行`);
  return parts.join(' · ') || '文档级图片';
}

function localDateTimeValue(value) {
  if (!value) return '';
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return '';
  const local = new Date(parsed.getTime() - parsed.getTimezoneOffset() * 60000);
  return local.toISOString().slice(0, 16);
}

function utcIsoValue(selector) {
  const value = document.querySelector(selector).value;
  return value ? new Date(value).toISOString() : null;
}

function updateProductFields() {
  const visible = document.querySelector('#domainInput').value === 'product_public';
  document.querySelectorAll('.product-field').forEach((field) => {
    field.hidden = !visible;
  });
}

function renderDocument(documentData) {
  currentDocument = documentData;
  document.title = `${documentData.title} · 文档详情`;
  document.querySelector('#documentTitle').textContent = documentData.title;
  document.querySelector('#documentSubtitle').textContent = `${documentData.filename} · ${documentData.chunk_count} 个独立片段 · ${documentData.media_count || 0} 张图片`;
  const hasActive = Number.isInteger(documentData.active_version) && !documentData.is_withdrawn;
  const hasDraft = !hasActive || documentData.source_version !== documentData.active_version || documentData.review_status !== 'approved';
  document.querySelector('#publicationState').textContent = documentData.is_withdrawn
    ? '已撤回'
    : hasActive ? `公开版本 v${documentData.active_version}` : '尚未发布';
  document.querySelector('#versionState').textContent = hasDraft
    ? `当前草稿 v${documentData.source_version} · ${documentData.review_status}`
    : `技术状态 ${documentData.status} · 无待审变更`;
  document.querySelector('#healthBadge').className = `health-badge ${documentData.status === 'READY' ? 'ready' : 'error'}`;
  document.querySelector('#healthBadge').innerHTML = `<i></i>${escapeHtml(documentData.status)}`;
  document.querySelector('#titleInput').value = documentData.title;
  document.querySelector('#domainInput').value = documentData.knowledge_domain;
  document.querySelector('#ownerCompanyInput').value = documentData.owner_company_id || '';
  document.querySelector('#brandInput').value = documentData.brand_id || '';
  document.querySelector('#productIdsInput').value = (documentData.product_ids || []).join(', ');
  document.querySelector('#evidenceKindInput').value = documentData.evidence_kind;
  document.querySelector('#visibilityInput').value = documentData.visibility;
  document.querySelector('#effectiveFromInput').value = localDateTimeValue(documentData.effective_from);
  document.querySelector('#effectiveToInput').value = localDateTimeValue(documentData.effective_to);
  document.querySelector('#publishButton').hidden = hasActive && !hasDraft;
  document.querySelector('#withdrawButton').hidden = !hasActive;
  updateProductFields();
}

function renderVersions() {
  if (!currentVersions.length) {
    versionList.innerHTML = '<div class="empty-state">尚无已发布版本。</div>';
    return;
  }
  const eventByVersion = new Map();
  publicationEvents.forEach((event) => {
    const key = event.version ?? 'none';
    if (!eventByVersion.has(key)) eventByVersion.set(key, []);
    eventByVersion.get(key).push(event);
  });
  versionList.innerHTML = currentVersions.map((version) => {
    const events = eventByVersion.get(version.version) || [];
    const latestEvent = events[0];
    return `<article class="chunk-item">
      <div class="chunk-meta">
        <span><strong>v${version.version}</strong> · ${version.is_active ? '当前公开快照' : '历史快照'}</span>
        <span>${version.chunk_count} 个片段 · ${version.media_count} 张图片</span>
      </div>
      <p>${escapeHtml(version.title)} · ${escapeHtml(version.reviewer_id)} · ${escapeHtml(version.published_at)}</p>
      <div class="chunk-actions">
        <small>${latestEvent ? `最近事件：${escapeHtml(latestEvent.event_type)} · ${escapeHtml(latestEvent.created_at)}` : ''}</small>
        ${version.is_active ? '' : `<button class="button ghost" data-rollback-version="${version.version}" type="button">回滚为公开版本</button>`}
      </div>
    </article>`;
  }).join('');
}

async function hydrateMediaPreviews() {
  await Promise.all([...mediaGrid.querySelectorAll('[data-media-preview]')].map(async (image) => {
    try {
      const response = await fetch(appUrl(`/api/admin/media/${encodeURIComponent(image.dataset.mediaPreview)}/content`), {
        headers: { 'X-Admin-Token': adminToken }
      });
      if (!response.ok) throw new Error('preview failed');
      const objectUrl = URL.createObjectURL(await response.blob());
      previewUrls.push(objectUrl);
      image.src = objectUrl;
      image.closest('.admin-media-preview')?.classList.add('loaded');
    } catch {
      image.alt = '图片预览加载失败';
    }
  }));
}

function renderMedia() {
  previewUrls.forEach((url) => URL.revokeObjectURL(url));
  previewUrls = [];
  document.querySelector('#mediaCount').textContent = String(currentMedia.length);
  if (!currentMedia.length) {
    mediaGrid.innerHTML = '<div class="empty-state">原文中尚未提取到可用图片。可点击“从原文重新提取”。</div>';
    return;
  }
  mediaGrid.innerHTML = currentMedia.map((asset) => `
    <article class="admin-media-card" data-media-id="${escapeHtml(asset.id)}">
      <div class="admin-media-preview">
        <img data-media-preview="${escapeHtml(asset.id)}" alt="${escapeHtml(asset.alt_text || asset.caption || '知识图片')}" />
      </div>
      <div class="admin-media-body">
        <div class="chunk-meta">
          <span>${escapeHtml(mediaLocation(asset))}</span>
          <span>${asset.width}×${asset.height} · ${escapeHtml(asset.source_kind)} · ${escapeHtml(asset.extraction_status || 'current')} / ${escapeHtml(asset.review_status)}</span>
        </div>
        <label class="field">
          <span>图片标题</span>
          <input data-media-caption maxlength="500" value="${escapeHtml(asset.caption)}" />
          ${asset.extracted_caption && asset.extracted_caption !== asset.caption
            ? `<small>本次原文提取标题：${escapeHtml(asset.extracted_caption)}（未覆盖运营标题）</small>` : ''}
        </label>
        <label class="field">
          <span>产品标识</span>
          <input data-media-products maxlength="500" value="${escapeHtml((asset.product_ids || []).join(', '))}" placeholder="例如 F02, water-fountain" />
        </label>
        <label class="field">
          <span>替代文本</span>
          <input data-media-alt maxlength="500" value="${escapeHtml(asset.alt_text)}" required />
        </label>
        <div class="chunk-actions">
          <label class="toggle-label"><input data-media-enabled type="checkbox" ${asset.enabled ? 'checked' : ''} />聊天可展示</label>
          <button class="button primary" data-save-media type="button">保存图片设置</button>
        </div>
      </div>
    </article>`).join('');
  hydrateMediaPreviews();
}

function renderChunks() {
  const query = document.querySelector('#chunkSearch').value.trim().toLowerCase();
  const enabledOnly = document.querySelector('#enabledOnly').checked;
  const filtered = currentChunks.filter((chunk) => {
    if (enabledOnly && !chunk.enabled) return false;
    const haystack = `${locationLabel(chunk)} ${chunk.content}`.toLowerCase();
    return !query || haystack.includes(query);
  });
  document.querySelector('#chunkCount').textContent = `${filtered.length} / ${currentChunks.length}`;
  if (!filtered.length) {
    chunkList.innerHTML = '<div class="empty-state">没有符合当前筛选条件的片段。</div>';
    return;
  }
  chunkList.innerHTML = filtered.map((chunk) => `
    <article id="chunk-${escapeHtml(chunk.id)}" class="chunk-item" data-chunk-id="${escapeHtml(chunk.id)}">
      <div class="chunk-meta">
        <span><strong>#${chunk.position + 1}</strong> · ${escapeHtml(locationLabel(chunk))}</span>
        <span>${chunk.content.length} 字 · ${chunk.enabled ? '已启用' : '已停用'}</span>
      </div>
      <textarea aria-label="知识片段 #${chunk.position + 1}">${escapeHtml(chunk.content)}</textarea>
      <div class="chunk-actions">
        <label class="toggle-label"><input type="checkbox" ${chunk.enabled ? 'checked' : ''} />参与检索</label>
        <button class="button primary" data-save-chunk type="button">保存此片段</button>
      </div>
    </article>`).join('');
}

async function loadDocument() {
  if (!adminToken) {
    showAuth();
    return;
  }
  try {
    const data = await request(`/api/admin/documents/${encodeURIComponent(documentId)}`);
    logoutButton.hidden = false;
    renderDocument(data.document);
    currentChunks = data.chunks;
    currentMedia = data.media || [];
    currentVersions = data.versions || [];
    publicationEvents = data.publication_events || [];
    renderChunks();
    renderMedia();
    renderVersions();
  } catch (error) {
    chunkList.innerHTML = `<div class="empty-state">${escapeHtml(error.message)}</div>`;
  }
}

document.querySelector('#metadataForm').addEventListener('submit', async (event) => {
  event.preventDefault();
  const domain = document.querySelector('#domainInput').value;
  const productIds = document.querySelector('#productIdsInput').value
    .split(/[,，\n]/).map((value) => value.trim()).filter(Boolean);
  const payload = {
    title: document.querySelector('#titleInput').value.trim(),
    knowledge_domain: domain,
    owner_company_id: document.querySelector('#ownerCompanyInput').value.trim(),
    brand_id: domain === 'product_public' ? (document.querySelector('#brandInput').value || null) : null,
    product_ids: domain === 'product_public' ? [...new Set(productIds)] : [],
    evidence_kind: document.querySelector('#evidenceKindInput').value,
    visibility: document.querySelector('#visibilityInput').value,
    effective_from: utcIsoValue('#effectiveFromInput'),
    effective_to: utcIsoValue('#effectiveToInput')
  };
  try {
    const data = await request(`/api/admin/documents/${encodeURIComponent(documentId)}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    renderDocument(data.document);
    showToast('文档信息已保存为待审核草稿；当前公开版本未受影响');
    await loadDocument();
  } catch (error) {
    showToast(error.message, true);
  }
});

mediaGrid.addEventListener('click', async (event) => {
  const button = event.target.closest('[data-save-media]');
  if (!button) return;
  const card = button.closest('.admin-media-card');
  const asset = currentMedia.find((value) => value.id === card.dataset.mediaId);
  const productIds = card.querySelector('[data-media-products]').value
    .split(/[,，\n]/).map((value) => value.trim()).filter(Boolean);
  const payload = {
    caption: card.querySelector('[data-media-caption]').value.trim(),
    alt_text: card.querySelector('[data-media-alt]').value.trim(),
    product_ids: [...new Set(productIds)],
    enabled: card.querySelector('[data-media-enabled]').checked
  };
  if (!payload.alt_text) {
    showToast('替代文本不能为空', true);
    return;
  }
  button.disabled = true;
  try {
    const data = await request(`/api/admin/media/${encodeURIComponent(card.dataset.mediaId)}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    if (asset) Object.assign(asset, data.asset);
    showToast('图片设置已保存为待审核草稿；当前公开图片未受影响');
    await loadDocument();
  } catch (error) {
    showToast(error.message, true);
  } finally {
    button.disabled = false;
  }
});

document.querySelector('#extractMediaButton').addEventListener('click', async (event) => {
  const button = event.currentTarget;
  button.disabled = true;
  try {
    const data = await request(`/api/admin/documents/${encodeURIComponent(documentId)}/extract-media`, { method: 'POST' });
    currentMedia = data.items || [];
    currentDocument.media_count = currentMedia.length;
    renderDocument(currentDocument);
    renderMedia();
    const diff = data.diff || {};
    showToast(`提取完成：新增 ${diff.new || 0}、变化 ${diff.changed || 0}、缺失 ${diff.missing || 0}；运营标题未被覆盖`);
    await loadDocument();
  } catch (error) {
    showToast(error.message, true);
  } finally {
    button.disabled = false;
  }
});

document.querySelector('#domainInput').addEventListener('change', updateProductFields);
document.querySelector('#chunkSearch').addEventListener('input', renderChunks);
document.querySelector('#enabledOnly').addEventListener('change', renderChunks);

chunkList.addEventListener('click', async (event) => {
  const button = event.target.closest('[data-save-chunk]');
  if (!button) return;
  const item = button.closest('.chunk-item');
  const chunk = currentChunks.find((value) => value.id === item.dataset.chunkId);
  const content = item.querySelector('textarea').value;
  const enabled = item.querySelector('input[type="checkbox"]').checked;
  button.disabled = true;
  try {
    const data = await request(`/api/admin/chunks/${encodeURIComponent(item.dataset.chunkId)}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ content, enabled })
    });
    const contentChanged = chunk && chunk.content !== content;
    const enabledChanged = chunk && chunk.enabled !== enabled;
    if (chunk) Object.assign(chunk, data.chunk);
    showToast(contentChanged || enabledChanged
      ? '片段已保存为待审核草稿；当前公开版本未受影响' : '片段没有变化');
    await loadDocument();
  } catch (error) {
    showToast(error.message, true);
  } finally {
    button.disabled = false;
  }
});

async function updatePublication(payload, successMessage) {
  try {
    const data = await request(`/api/admin/documents/${encodeURIComponent(documentId)}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    renderDocument(data.document);
    showToast(successMessage);
    await loadDocument();
  } catch (error) {
    showToast(error.message, true);
  }
}

document.querySelector('#publishButton').addEventListener('click', () => updatePublication(
  { visibility: 'public', review_status: 'approved', publication_status: 'published' },
  '文档已审核发布'
));
document.querySelector('#withdrawButton').addEventListener('click', () => updatePublication(
  { publication_status: 'withdrawn' },
  '文档已撤回公开检索'
));
versionList.addEventListener('click', async (event) => {
  const button = event.target.closest('[data-rollback-version]');
  if (!button) return;
  const version = Number(button.dataset.rollbackVersion);
  if (!window.confirm(`确认将公开服务回滚到 v${version}？当前草稿不会被覆盖。`)) return;
  button.disabled = true;
  try {
    await request(`/api/admin/documents/${encodeURIComponent(documentId)}/rollback/${version}`, { method: 'POST' });
    showToast(`公开版本已回滚到 v${version}，当前草稿仍保留`);
    await loadDocument();
  } catch (error) {
    showToast(error.message, true);
  } finally {
    button.disabled = false;
  }
});
document.querySelector('#reindexButton').addEventListener('click', async (event) => {
  const button = event.currentTarget;
  button.disabled = true;
  try {
    await request(`/api/admin/documents/${encodeURIComponent(documentId)}/reindex`, { method: 'POST' });
    showToast('已按当前片段重算向量并生成待审核草稿；公开版本未受影响');
    await loadDocument();
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
    const data = await request(`/api/admin/documents/${encodeURIComponent(documentId)}`);
    authDialog.close();
    adminTokenInput.value = '';
    logoutButton.hidden = false;
    renderDocument(data.document);
    currentChunks = data.chunks;
    currentMedia = data.media || [];
    currentVersions = data.versions || [];
    publicationEvents = data.publication_events || [];
    renderChunks();
    renderMedia();
    renderVersions();
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

loadDocument();
