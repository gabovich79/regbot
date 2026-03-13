// ==================== State ====================
let currentConversationId = null;
let currentSessionId = null;
let isStreaming = false;

// ==================== Sidebar Toggle (Mobile) ====================

function toggleSidebar() {
    const sidebar = document.getElementById('sidebar');
    const overlay = document.getElementById('sidebar-overlay');
    if (!sidebar) return;

    const isOpen = sidebar.classList.contains('sidebar-open');
    if (isOpen) {
        sidebar.classList.remove('sidebar-open');
        overlay?.classList.add('hidden');
        document.body.style.overflow = '';
    } else {
        sidebar.classList.add('sidebar-open');
        overlay?.classList.remove('hidden');
        document.body.style.overflow = 'hidden';
    }
}

// ==================== Chat ====================

function initChat() {
    const form = document.getElementById('chat-form');
    if (!form) return;

    form.addEventListener('submit', (e) => {
        e.preventDefault();
        sendMessage();
    });

    const input = document.getElementById('question-input');
    if (input) {
        input.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });
        input.addEventListener('input', autoResize);
    }

    document.getElementById('new-chat-btn')?.addEventListener('click', newChat);
    loadConversations();
}

function autoResize() {
    const el = document.getElementById('question-input');
    el.style.height = 'auto';
    el.style.height = Math.min(el.scrollHeight, 120) + 'px';
}

function newChat() {
    currentConversationId = null;
    currentSessionId = null;
    const messages = document.getElementById('chat-messages');
    if (messages) {
        messages.innerHTML = document.getElementById('welcome-message')?.outerHTML || '';
    }
    document.querySelectorAll('.conv-item').forEach(el => el.classList.remove('active'));
    // Close sidebar on mobile
    if (window.innerWidth < 1024) {
        const sidebar = document.getElementById('sidebar');
        if (sidebar?.classList.contains('sidebar-open')) toggleSidebar();
    }
}

async function sendMessage() {
    const input = document.getElementById('question-input');
    const question = input.value.trim();
    if (!question || isStreaming) return;

    // Hide welcome
    const welcome = document.getElementById('welcome-message');
    if (welcome) welcome.remove();

    // Add user bubble
    addMessage('user', question);
    input.value = '';
    input.style.height = 'auto';

    isStreaming = true;
    document.getElementById('send-btn').disabled = true;

    // Create assistant bubble
    const assistantDiv = addMessage('assistant', '', true);
    const contentSpan = assistantDiv.querySelector('.msg-content');

    const formData = new FormData();
    formData.append('question', question);
    if (currentConversationId) formData.append('conversation_id', currentConversationId);
    if (currentSessionId) formData.append('session_id', currentSessionId);

    try {
        const response = await fetch('/api/chat', { method: 'POST', body: formData });
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        let fullText = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop();

            for (const line of lines) {
                if (!line.startsWith('data: ')) continue;
                const data = JSON.parse(line.slice(6));

                if (data.type === 'thinking') {
                    contentSpan.innerHTML = `<span class="thinking-indicator">${escapeHtml(data.text)}</span>`;
                    scrollToBottom();
                } else if (data.type === 'text') {
                    if (!fullText) {
                        contentSpan.innerHTML = ''; // Clear thinking indicator
                    }
                    fullText += data.text;
                    contentSpan.innerHTML = renderMarkdown(fullText);
                    scrollToBottom();
                } else if (data.type === 'usage') {
                    contentSpan.classList.remove('streaming-cursor');
                    if (data.data?.confidence) {
                        addConfidenceBadge(assistantDiv, data.data.confidence);
                    }
                } else if (data.type === 'done') {
                    currentConversationId = data.conversation_id;
                    currentSessionId = data.session_id;
                    loadConversations();
                } else if (data.type === 'error') {
                    contentSpan.textContent = data.text;
                    contentSpan.classList.remove('streaming-cursor');
                    contentSpan.classList.add('text-red-600');
                }
            }
        }
    } catch (err) {
        contentSpan.textContent = 'שגיאה בחיבור לשרת';
        contentSpan.classList.remove('streaming-cursor');
        contentSpan.classList.add('text-red-600');
    }

    isStreaming = false;
    document.getElementById('send-btn').disabled = false;
}

function addMessage(role, content, isStreaming = false) {
    const container = document.getElementById('chat-messages');
    const div = document.createElement('div');
    div.className = `p-4 ${role === 'user' ? 'msg-user' : 'msg-assistant'}`;

    const contentSpan = document.createElement('div');
    contentSpan.className = 'msg-content' + (isStreaming ? ' streaming-cursor' : '');
    contentSpan.innerHTML = role === 'user' ? escapeHtml(content) : renderMarkdown(content);

    div.appendChild(contentSpan);
    container.appendChild(div);
    scrollToBottom();
    return div;
}

function addConfidenceBadge(div, confidence) {
    const badge = document.createElement('span');
    const cls = confidence === 'HIGH' ? 'badge-high' : confidence === 'MEDIUM' ? 'badge-medium' : 'badge-low';
    badge.className = `inline-block mt-2 px-2 py-1 rounded-full text-xs font-medium ${cls}`;
    badge.textContent = confidence;
    div.appendChild(badge);
}

function scrollToBottom() {
    const container = document.getElementById('chat-messages');
    if (container) container.scrollTop = container.scrollHeight;
}

async function loadConversations() {
    const list = document.getElementById('conversations-list');
    if (!list) return;

    try {
        const res = await fetch('/api/conversations');
        const convs = await res.json();
        list.innerHTML = '';

        for (const conv of convs) {
            const div = document.createElement('div');
            div.className = `conv-item p-3 rounded-lg border border-transparent text-sm ${conv.id === currentConversationId ? 'active' : ''}`;
            div.textContent = (conv.first_question || 'שיחה חדשה').substring(0, 50);
            div.onclick = () => loadConversation(conv.id);
            list.appendChild(div);
        }
    } catch (e) { /* ignore */ }
}

async function loadConversation(convId) {
    currentConversationId = convId;
    const container = document.getElementById('chat-messages');
    container.innerHTML = '';

    try {
        const res = await fetch(`/api/conversations/${convId}`);
        const messages = await res.json();

        for (const msg of messages) {
            const div = addMessage(msg.role, msg.content);
            if (msg.role === 'assistant' && msg.confidence) {
                addConfidenceBadge(div, msg.confidence);
            }
        }
    } catch (e) {
        container.innerHTML = '<p class="text-red-500 text-center p-4">שגיאה בטעינת השיחה</p>';
    }

    loadConversations();
    // Close sidebar on mobile after selecting conversation
    if (window.innerWidth < 1024) toggleSidebar();
}

function askSuggestion(btn) {
    const input = document.getElementById('question-input');
    if (input) {
        input.value = btn.textContent.trim();
        sendMessage();
    }
}

// ==================== Admin - Documents ====================

async function loadDocuments() {
    const table = document.getElementById('documents-table');
    const noDocs = document.getElementById('no-docs');
    if (!table) return;

    try {
        const res = await fetch('/api/documents');
        const docs = await res.json();

        if (docs.length === 0) {
            table.innerHTML = '';
            noDocs?.classList.remove('hidden');
            return;
        }

        noDocs?.classList.add('hidden');
        table.innerHTML = docs.map(doc => `
            <tr class="border-b border-gray-100 hover:bg-gray-50">
                <td class="px-4 py-3 font-medium">${escapeHtml(doc.title)}</td>
                <td class="px-4 py-3">
                    <span class="px-2 py-1 rounded-full text-xs bg-gray-100">${doc.source_type}</span>
                </td>
                <td class="px-4 py-3">${(doc.token_count || 0).toLocaleString()}</td>
                <td class="px-4 py-3 text-gray-500">${formatDate(doc.added_at)}</td>
                <td class="px-4 py-3">
                    <button onclick="deleteDoc(${doc.id})" class="text-red-500 hover:text-red-700 text-xs py-2 px-2 min-h-[44px] inline-flex items-center">🗑 הסר</button>
                </td>
            </tr>
        `).join('');

        // Update stats
        const activeDocs = docs.filter(d => d.is_active);
        document.getElementById('doc-count').textContent = activeDocs.length;
        const totalTokens = activeDocs.reduce((sum, d) => sum + (d.token_count || 0), 0);
        document.getElementById('total-tokens').textContent = totalTokens.toLocaleString();

        const warning = document.getElementById('token-warning');
        if (warning) {
            totalTokens > 150000 ? warning.classList.remove('hidden') : warning.classList.add('hidden');
        }
    } catch (e) {
        table.innerHTML = '<tr><td colspan="5" class="text-center py-4 text-red-500">שגיאה בטעינה</td></tr>';
    }
}

async function deleteDoc(id) {
    if (!confirm('בטוח להסיר מסמך זה?')) return;
    try {
        await fetch(`/api/documents/${id}`, { method: 'DELETE' });
        loadDocuments();
    } catch (e) {
        alert('שגיאה בהסרת המסמך');
    }
}

function setupUpload() {
    const zone = document.getElementById('upload-zone');
    const fileInput = document.getElementById('file-input');
    if (!zone || !fileInput) return;

    zone.addEventListener('dragover', (e) => { e.preventDefault(); zone.classList.add('drag-over'); });
    zone.addEventListener('dragleave', () => zone.classList.remove('drag-over'));
    zone.addEventListener('drop', (e) => {
        e.preventDefault();
        zone.classList.remove('drag-over');
        handleFiles(e.dataTransfer.files);
    });

    fileInput.addEventListener('change', () => handleFiles(fileInput.files));

    // URL form
    document.getElementById('url-form')?.addEventListener('submit', async (e) => {
        e.preventDefault();
        const url = document.getElementById('url-input').value.trim();
        const title = document.getElementById('url-title').value.trim();
        if (!url) return;

        showUploadModal('מוריד ומעבד מסמך...');
        try {
            const formData = new FormData();
            formData.append('url', url);
            if (title) formData.append('title', title);

            const res = await fetch('/api/documents/url', { method: 'POST', body: formData });
            const data = await res.json();
            if (!res.ok) throw new Error(data.detail || 'שגיאה');

            hideUploadModal();
            alert(data.message);
            if (data.warning) alert('⚠️ ' + data.warning);
            document.getElementById('url-input').value = '';
            document.getElementById('url-title').value = '';
            loadDocuments();
        } catch (err) {
            hideUploadModal();
            alert('שגיאה: ' + err.message);
        }
    });
}

async function handleFiles(files) {
    for (const file of files) {
        showUploadModal(`מעבד: ${file.name}...`);
        try {
            const formData = new FormData();
            formData.append('file', file);
            const res = await fetch('/api/documents/upload', { method: 'POST', body: formData });
            const data = await res.json();
            if (!res.ok) throw new Error(data.detail || 'שגיאה');
            if (data.warning) alert('⚠️ ' + data.warning);
        } catch (err) {
            alert(`שגיאה בעיבוד ${file.name}: ${err.message}`);
        }
    }
    hideUploadModal();
    loadDocuments();
}

function showUploadModal(text) {
    document.getElementById('upload-status').textContent = text;
    document.getElementById('upload-modal')?.classList.remove('hidden');
}

function hideUploadModal() {
    document.getElementById('upload-modal')?.classList.add('hidden');
}

// ==================== Admin - Costs ====================

async function loadCosts() {
    try {
        const res = await fetch('/api/costs');
        const data = await res.json();

        const el = (id) => document.getElementById(id);
        if (el('cost-today')) el('cost-today').textContent = '$' + (data.summary.today || 0).toFixed(4);
        if (el('cost-month')) el('cost-month').textContent = '$' + (data.summary.this_month || 0).toFixed(4);
        if (el('cost-total')) el('cost-total').textContent = '$' + (data.summary.total || 0).toFixed(4);

        // Chart
        const chart = el('cost-chart');
        if (chart && data.daily.length > 0) {
            const maxCost = Math.max(...data.daily.map(d => d.total_cost || 0), 0.001);
            chart.innerHTML = data.daily.reverse().map(d => {
                const height = Math.max(((d.total_cost || 0) / maxCost) * 100, 4);
                return `
                    <div class="flex-1 flex flex-col items-center gap-1">
                        <div class="text-xs text-gray-500">$${(d.total_cost || 0).toFixed(3)}</div>
                        <div class="w-full bg-blue-500 rounded-t" style="height:${height}%"></div>
                        <div class="text-xs text-gray-400">${d.date?.slice(5) || ''}</div>
                    </div>`;
            }).join('');
        } else if (chart) {
            chart.innerHTML = '<div class="text-gray-400 text-center w-full py-8">אין נתונים עדיין</div>';
        }
    } catch (e) { /* ignore */ }
}

function switchTab(tab) {
    document.getElementById('panel-docs')?.classList.toggle('hidden', tab !== 'docs');
    document.getElementById('panel-costs')?.classList.toggle('hidden', tab !== 'costs');
    document.getElementById('panel-instructions')?.classList.toggle('hidden', tab !== 'instructions');
    document.querySelectorAll('.tab-btn').forEach(btn => {
        const isActive = btn.id === `tab-${tab}`;
        btn.classList.toggle('bg-white', isActive);
        btn.classList.toggle('shadow', isActive);
        btn.classList.toggle('text-blue-700', isActive);
        btn.classList.toggle('text-gray-600', !isActive);
    });
    if (tab === 'instructions') loadInstructions();
}

// ==================== Admin - Instructions ====================

async function loadInstructions() {
    const editor = document.getElementById('instructions-editor');
    if (!editor) return;

    try {
        const res = await fetch('/api/settings/instructions');
        const data = await res.json();

        editor.value = data.instructions;
        updateInstructionsUI(data.is_custom);
        updateCharCount();
    } catch (e) {
        editor.value = 'שגיאה בטעינת ההוראות';
    }

    editor.addEventListener('input', updateCharCount);
}

function updateInstructionsUI(isCustom) {
    const customBadge = document.getElementById('instructions-custom-badge');
    const defaultBadge = document.getElementById('instructions-default-badge');
    if (isCustom) {
        customBadge?.classList.remove('hidden');
        defaultBadge?.classList.add('hidden');
    } else {
        customBadge?.classList.add('hidden');
        defaultBadge?.classList.remove('hidden');
    }
}

function updateCharCount() {
    const editor = document.getElementById('instructions-editor');
    const counter = document.getElementById('instructions-char-count');
    if (editor && counter) {
        counter.textContent = `${editor.value.length} תווים`;
    }
}

function showInstructionsStatus(message, isError) {
    const el = document.getElementById('instructions-status');
    if (!el) return;
    el.textContent = message;
    el.className = `mb-3 px-4 py-2 rounded-lg text-sm ${isError ? 'bg-red-50 text-red-700 border border-red-200' : 'bg-green-50 text-green-700 border border-green-200'}`;
    el.classList.remove('hidden');
    setTimeout(() => el.classList.add('hidden'), 4000);
}

async function saveInstructions() {
    const editor = document.getElementById('instructions-editor');
    if (!editor) return;

    const instructions = editor.value.trim();
    if (!instructions) {
        showInstructionsStatus('ההוראות לא יכולות להיות ריקות', true);
        return;
    }

    try {
        const formData = new FormData();
        formData.append('instructions', instructions);

        const res = await fetch('/api/settings/instructions', { method: 'PUT', body: formData });
        const data = await res.json();

        if (!res.ok) throw new Error(data.detail || 'שגיאה');

        showInstructionsStatus('✅ ההוראות נשמרו בהצלחה — ייכנסו לתוקף בשאלה הבאה', false);
        updateInstructionsUI(true);
    } catch (e) {
        showInstructionsStatus('שגיאה בשמירה: ' + e.message, true);
    }
}

async function resetInstructions() {
    if (!confirm('לאפס את ההוראות לברירת המחדל? השינויים שלך יימחקו.')) return;

    try {
        const res = await fetch('/api/settings/instructions', { method: 'DELETE' });
        const data = await res.json();

        if (!res.ok) throw new Error(data.detail || 'שגיאה');

        document.getElementById('instructions-editor').value = data.instructions;
        showInstructionsStatus('↩️ ההוראות אופסו לברירת מחדל', false);
        updateInstructionsUI(false);
        updateCharCount();
    } catch (e) {
        showInstructionsStatus('שגיאה באיפוס: ' + e.message, true);
    }
}

// ==================== Logs ====================

let currentLogPage = 1;

async function loadLogs(page) {
    currentLogPage = page;
    const table = document.getElementById('logs-table');
    const noLogs = document.getElementById('no-logs');
    if (!table) return;

    const dateFrom = document.getElementById('filter-from')?.value || '';
    const dateTo = document.getElementById('filter-to')?.value || '';

    let url = `/api/logs?page=${page}`;
    if (dateFrom) url += `&date_from=${dateFrom}`;
    if (dateTo) url += `&date_to=${dateTo}`;

    try {
        const res = await fetch(url);
        const data = await res.json();

        if (data.logs.length === 0) {
            table.innerHTML = '';
            noLogs?.classList.remove('hidden');
            document.getElementById('pagination').innerHTML = '';
            return;
        }

        noLogs?.classList.add('hidden');
        table.innerHTML = data.logs.map(log => `
            <tr class="border-b border-gray-100 hover:bg-gray-50 cursor-pointer" onclick="showLogDetail(${log.conversation_id})">
                <td class="px-4 py-3 text-gray-500 text-xs">${formatDate(log.created_at)}</td>
                <td class="px-4 py-3">${escapeHtml((log.question || '').substring(0, 60))}${(log.question || '').length > 60 ? '...' : ''}</td>
                <td class="px-4 py-3">
                    ${log.confidence ? `<span class="px-2 py-1 rounded-full text-xs badge-${log.confidence.toLowerCase()}">${log.confidence}</span>` : '-'}
                </td>
                <td class="px-4 py-3 text-gray-500">${log.response_time_ms ? (log.response_time_ms / 1000).toFixed(1) + 's' : '-'}</td>
                <td class="px-4 py-3 text-gray-500">${log.cost_usd ? '$' + log.cost_usd.toFixed(4) : '-'}</td>
            </tr>
        `).join('');

        // Pagination
        const totalPages = Math.ceil(data.total / data.per_page);
        const pagination = document.getElementById('pagination');
        if (pagination) {
            pagination.innerHTML = '';
            for (let i = 1; i <= totalPages; i++) {
                const btn = document.createElement('button');
                btn.textContent = i;
                btn.className = `px-3 py-2 rounded text-sm min-h-[44px] min-w-[44px] ${i === page ? 'bg-blue-600 text-white' : 'bg-white border border-gray-300 hover:bg-gray-50'}`;
                btn.onclick = () => loadLogs(i);
                pagination.appendChild(btn);
            }
        }
    } catch (e) {
        table.innerHTML = '<tr><td colspan="5" class="text-center py-4 text-red-500">שגיאה בטעינה</td></tr>';
    }
}

async function showLogDetail(conversationId) {
    const modal = document.getElementById('message-modal');
    const content = document.getElementById('modal-content');
    if (!modal || !content) return;

    try {
        const res = await fetch(`/api/conversations/${conversationId}`);
        const messages = await res.json();

        content.innerHTML = messages.map(msg => `
            <div class="p-3 rounded-lg ${msg.role === 'user' ? 'bg-blue-50 border border-blue-200' : 'bg-gray-50 border border-gray-200'}">
                <div class="text-xs text-gray-500 mb-1">${msg.role === 'user' ? '👤 שאלה' : '🤖 תשובה'}</div>
                <div class="text-sm">${msg.role === 'user' ? escapeHtml(msg.content) : renderMarkdown(msg.content)}</div>
                ${msg.confidence ? `<span class="inline-block mt-2 px-2 py-1 rounded-full text-xs badge-${msg.confidence.toLowerCase()}">${msg.confidence}</span>` : ''}
            </div>
        `).join('');

        modal.classList.remove('hidden');
    } catch (e) {
        alert('שגיאה בטעינת השיחה');
    }
}

function closeModal(e) {
    if (e.target === e.currentTarget) {
        e.target.classList.add('hidden');
    }
}

function exportCSV() {
    window.open('/api/logs/export', '_blank');
}

// ==================== Utils ====================

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function renderMarkdown(text) {
    if (!text) return '';
    return text
        .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
        .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.+?)\*/g, '<em>$1</em>')
        .replace(/`(.+?)`/g, '<code>$1</code>')
        .replace(/^### (.+)$/gm, '<h3 class="text-base font-bold mt-3 mb-1">$1</h3>')
        .replace(/^## (.+)$/gm, '<h2 class="text-lg font-bold mt-3 mb-1">$1</h2>')
        .replace(/^# (.+)$/gm, '<h1 class="text-xl font-bold mt-3 mb-1">$1</h1>')
        .replace(/^---$/gm, '<hr>')
        .replace(/^- (.+)$/gm, '<li>$1</li>')
        .replace(/(<li>.*<\/li>)/gs, '<ul class="list-disc pr-6 my-1">$1</ul>')
        .replace(/^\d+\. (.+)$/gm, '<li>$1</li>')
        .replace(/\n/g, '<br>');
}

function formatDate(dateStr) {
    if (!dateStr) return '-';
    try {
        const d = new Date(dateStr);
        return d.toLocaleDateString('he-IL') + ' ' + d.toLocaleTimeString('he-IL', { hour: '2-digit', minute: '2-digit' });
    } catch {
        return dateStr;
    }
}

// ==================== Init ====================

document.addEventListener('DOMContentLoaded', () => {
    if (document.getElementById('chat-form')) {
        initChat();
    }
});
