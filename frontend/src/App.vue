<script setup lang="ts">
import { computed, nextTick, onMounted, ref } from "vue"
import {
  BookOpen,
  Bot,
  Check,
  ChevronDown,
  CircleUserRound,
  CloudUpload,
  Database,
  FileText,
  Globe2,
  LoaderCircle,
  LogOut,
  Menu,
  MessageSquarePlus,
  Paperclip,
  Search,
  Send,
  Sparkles,
  Trash2,
  X,
} from "lucide-vue-next"
import { authApi, createImportTask, getImportTask, listImportTasks, sessionApi, streamQuery } from "./api"
import type { ChatMessage, ImportTask, ProgressStage, SessionSummary, Source, User } from "./types"

const user = ref<User | null>(null)
const booting = ref(true)
const authError = ref("")
const DEFAULT_USERNAME = "admin"
const DEFAULT_PASSWORD = "admin"
const username = ref(DEFAULT_USERNAME)
const password = ref(DEFAULT_PASSWORD)
const loggingIn = ref(false)
const sessions = ref<SessionSummary[]>([])
const sessionId = ref("")
const messages = ref<ChatMessage[]>([])
const query = ref("")
const mode = ref<"fast" | "deep">("fast")
const querying = ref(false)
const queryError = ref("")
const sources = ref<Source[]>([])
const sourceTab = ref<"internal" | "web">("internal")
const mobileSidebar = ref(false)
const deleteTarget = ref<SessionSummary | null>(null)
const deleting = ref(false)
const deleteError = ref("")
const importOpen = ref(false)
const importTask = ref<ImportTask | null>(null)
const importTasks = ref<ImportTask[]>([])
const importTab = ref<"upload" | "history">("upload")
const importing = ref(false)
const importError = ref("")
const fileInput = ref<HTMLInputElement | null>(null)
const messageList = ref<HTMLElement | null>(null)
let importEventSource: EventSource | null = null

const stageLabels: Record<string, string> = {
  item_confirm: "确认商品主体",
  embedding: "检索内部知识",
  hyde: "生成假设文档",
  web: "检索联网资料",
  rrf: "融合多路结果",
  rerank: "精排检索证据",
  answer: "生成最终答案",
}

const initialQueryStages = (): ProgressStage[] =>
  Object.entries(stageLabels).map(([key, label]) => ({ key, label, status: "pending" }))

const queryStages = ref<ProgressStage[]>(initialQueryStages())

const activeStageLabel = computed(() => {
  const running = queryStages.value.find((stage) => stage.status === "running")
  if (running) return running.label
  const pending = queryStages.value.find((stage) => stage.status === "pending")
  return pending?.label || "正在检索"
})

const completedStageCount = computed(
  () =>
    queryStages.value.filter((stage) => stage.status === "completed" || stage.status === "skipped")
      .length,
)

const filteredSources = computed(() =>
  sources.value.filter((source) => source.source_type === sourceTab.value),
)

const hasConversation = computed(() => messages.value.length > 0)

function scrollToBottom() {
  nextTick(() => messageList.value?.scrollTo({ top: messageList.value.scrollHeight, behavior: "smooth" }))
}

async function loadSessions() {
  sessions.value = await sessionApi.list()
}

async function signIn() {
  authError.value = ""
  loggingIn.value = true
  try {
    user.value = await authApi.login(username.value, password.value)
    password.value = ""
    await loadSessions()
    await resumeImport()
  } catch (error) {
    authError.value = error instanceof Error ? error.message : "登录失败"
  } finally {
    loggingIn.value = false
  }
}

async function signOut() {
  await authApi.logout()
  user.value = null
  sessions.value = []
  messages.value = []
  sessionId.value = ""
  username.value = DEFAULT_USERNAME
  password.value = DEFAULT_PASSWORD
}

function newConversation() {
  sessionId.value = ""
  messages.value = []
  sources.value = []
  queryStages.value = initialQueryStages()
  queryError.value = ""
  mobileSidebar.value = false
}

async function openSession(id: string) {
  const response = await sessionApi.messages(id)
  sessionId.value = id
  messages.value = response.messages as ChatMessage[]
  const lastAssistant = [...messages.value].reverse().find((message) => message.role === "assistant")
  const savedSources = lastAssistant?.metadata?.sources
  sources.value = Array.isArray(savedSources) ? (savedSources as Source[]) : []
  queryStages.value = initialQueryStages().map((stage) => ({ ...stage, status: "completed" }))
  mobileSidebar.value = false
  scrollToBottom()
}

function requestDelete(session: SessionSummary) {
  deleteTarget.value = session
  deleteError.value = ""
}

function cancelDelete() {
  if (deleting.value) return
  deleteTarget.value = null
  deleteError.value = ""
}

async function confirmDelete() {
  const target = deleteTarget.value
  if (!target || deleting.value) return
  deleting.value = true
  deleteError.value = ""
  try {
    await sessionApi.remove(target.session_id)
    deleteTarget.value = null
    if (sessionId.value === target.session_id) newConversation()
    await loadSessions()
  } catch (error) {
    deleteError.value = error instanceof Error ? error.message : "删除失败"
  } finally {
    deleting.value = false
  }
}

function updateQueryStage(data: Record<string, unknown>) {
  const key = String(data.stage || "")
  const target = queryStages.value.find((stage) => stage.key === key)
  if (!target) return
  target.status = String(data.status || "pending") as ProgressStage["status"]
  if (typeof data.count === "number") target.count = data.count
}

async function submitQuery() {
  const content = query.value.trim()
  if (!content || querying.value) return
  query.value = ""
  queryError.value = ""
  querying.value = true
  sources.value = []
  queryStages.value = initialQueryStages()
  const messageId = `message-${crypto.randomUUID().replaceAll("-", "")}`
  messages.value.push({ message_id: messageId, role: "user", content })
  const answerMessage: ChatMessage = {
    message_id: `${messageId}:answer`,
    role: "assistant",
    content: "",
    content_html: "",
    pending: true,
  }
  messages.value.push(answerMessage)
  const answerIndex = messages.value.length - 1
  scrollToBottom()

  try {
    await streamQuery(
      {
        query: content,
        session_id: sessionId.value || undefined,
        message_id: messageId,
        mode: mode.value,
      },
      (event, data) => {
        if (event === "start") sessionId.value = String(data.session_id || "")
        if (event === "progress") updateQueryStage(data)
        if (event === "answer") {
          messages.value[answerIndex].content += String(data.content || "")
          scrollToBottom()
        }
        if (event === "final") {
          messages.value[answerIndex].content = String(
            data.answer || messages.value[answerIndex].content,
          )
          messages.value[answerIndex].content_html = String(data.answer_html || "")
          messages.value[answerIndex].pending = false
          sources.value = Array.isArray(data.sources) ? (data.sources as Source[]) : []
        }
        if (event === "error") throw new Error(String(data.message || "查询失败"))
      },
    )
    await loadSessions()
  } catch (error) {
    messages.value[answerIndex].pending = false
    queryError.value = error instanceof Error ? error.message : "查询失败"
    if (!messages.value[answerIndex].content) messages.value.pop()
  } finally {
    querying.value = false
    scrollToBottom()
  }
}

function subscribeImport(taskId: string, afterRevision = 0) {
  importEventSource?.close()
  const eventSource = new EventSource(
    `/api/v1/imports/tasks/${encodeURIComponent(taskId)}/events?after_revision=${afterRevision}`,
    { withCredentials: true },
  )
  importEventSource = eventSource
  const update = (event: MessageEvent) => {
    const task = JSON.parse(event.data) as ImportTask
    importTask.value = task
    const index = importTasks.value.findIndex((item) => item.task_id === task.task_id)
    if (index >= 0) importTasks.value[index] = task
    else importTasks.value.unshift(task)
  }
  eventSource.addEventListener("progress", update)
  eventSource.addEventListener("done", (event) => {
    update(event as MessageEvent)
    importing.value = false
    localStorage.removeItem("activeImportTaskId")
    eventSource.close()
    importEventSource = null
  })
  eventSource.onerror = () => {
    eventSource.close()
    importEventSource = null
    if (importTask.value?.status === "running") {
      importError.value = "进度连接已断开，可关闭窗口后重新打开继续查看。"
    }
  }
}

async function uploadDocument() {
  const file = fileInput.value?.files?.[0]
  if (!file || importing.value) return
  importError.value = ""
  importing.value = true
  try {
    importTask.value = await createImportTask(file)
    importTasks.value = [
      importTask.value,
      ...importTasks.value.filter((task) => task.task_id !== importTask.value?.task_id),
    ]
    localStorage.setItem("activeImportTaskId", importTask.value.task_id)
    subscribeImport(importTask.value.task_id, importTask.value.revision)
  } catch (error) {
    importing.value = false
    importError.value = error instanceof Error ? error.message : "导入失败"
  }
}

async function openImportModal(tab: "upload" | "history" = "upload") {
  importOpen.value = true
  importTab.value = tab
  importError.value = ""
  try {
    importTasks.value = await listImportTasks()
  } catch (error) {
    importError.value = error instanceof Error ? error.message : "导入历史加载失败"
  }
}

function showImportTask(task: ImportTask) {
  importTask.value = task
  importing.value = task.status === "queued" || task.status === "running"
  if (importing.value) {
    localStorage.setItem("activeImportTaskId", task.task_id)
    subscribeImport(task.task_id, task.revision)
  }
}

function showImportHistory() {
  importTask.value = null
  importing.value = false
  importTab.value = "history"
}

function statusText(status: ImportTask["status"]) {
  return {
    queued: "等待中",
    running: "导入中",
    succeeded: "已完成",
    failed: "失败",
  }[status]
}

function formatDate(value?: string) {
  if (!value) return ""
  return new Intl.DateTimeFormat("zh-CN", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  }).format(new Date(value))
}

async function resumeImport() {
  let taskId = localStorage.getItem("activeImportTaskId")
  try {
    if (!taskId) {
      const recentTasks = await listImportTasks()
      importTasks.value = recentTasks
      taskId = recentTasks.find((task) => task.status === "queued" || task.status === "running")?.task_id || null
    }
    if (!taskId) return
    const task = await getImportTask(taskId)
    importTask.value = task
    importOpen.value = true
    importing.value = task.status === "queued" || task.status === "running"
    if (importing.value) subscribeImport(task.task_id, task.revision)
    else localStorage.removeItem("activeImportTaskId")
  } catch {
    localStorage.removeItem("activeImportTaskId")
  }
}

onMounted(async () => {
  try {
    user.value = await authApi.me()
    await Promise.all([loadSessions(), resumeImport()])
  } catch {
    user.value = null
  } finally {
    booting.value = false
  }
})
</script>

<template>
  <div v-if="booting" class="grid min-h-screen place-items-center bg-slate-50 text-slate-500">
    <LoaderCircle class="size-7 animate-spin text-blue-600" />
  </div>

  <main v-else-if="!user" class="login-shell">
    <section class="login-card">
      <div class="brand-mark"><BookOpen class="size-7" /></div>
      <p class="eyebrow">ENTERPRISE KNOWLEDGE</p>
      <h1>登录知识库助手</h1>
      <p class="login-copy">使用内部账号进入企业知识检索与文档导入平台。</p>
      <form class="mt-8 space-y-4" @submit.prevent="signIn">
        <label class="field-label">用户名<input v-model="username" autocomplete="username" required minlength="3" /></label>
        <label class="field-label">密码<input v-model="password" type="password" autocomplete="current-password" required minlength="5" /></label>
        <p v-if="authError" class="error-text">{{ authError }}</p>
        <button class="primary-button w-full" :disabled="loggingIn">
          <LoaderCircle v-if="loggingIn" class="size-4 animate-spin" />
          {{ loggingIn ? "正在登录" : "登录" }}
        </button>
      </form>
      <p class="mt-6 text-center text-xs text-slate-400">账号由管理员创建，不开放自主注册</p>
    </section>
  </main>

  <div v-else class="app-shell">
    <aside :class="['sidebar', mobileSidebar && 'sidebar-open']">
      <div class="flex items-center gap-3 px-5 py-5">
        <div class="brand-mark small"><BookOpen class="size-5" /></div>
        <div><h1 class="text-base font-bold text-slate-900">知识库助手</h1><p class="text-[11px] text-slate-400">企业知识 · 智能问答</p></div>
        <button class="ml-auto lg:hidden" aria-label="关闭菜单" @click="mobileSidebar = false"><X class="size-5" /></button>
      </div>
      <div class="px-4"><button class="primary-button w-full" @click="newConversation"><MessageSquarePlus class="size-4" />新建对话</button></div>
      <div class="mt-7 flex-1 overflow-y-auto px-3">
        <p class="px-2 text-[11px] font-semibold tracking-wider text-slate-400">最近对话</p>
        <div v-for="session in sessions" :key="session.session_id" class="session-item-wrap">
          <button
            :class="['session-item', sessionId === session.session_id && 'active']"
            @click="openSession(session.session_id)"
          >
            <span class="truncate font-medium">{{ session.title }}</span>
            <span class="truncate text-xs text-slate-400">{{ session.last_message }}</span>
          </button>
          <button
            class="session-delete"
            title="删除对话"
            aria-label="删除对话"
            :disabled="querying && sessionId === session.session_id"
            @click.stop="requestDelete(session)"
          >
            <Trash2 class="size-3.5" />
          </button>
        </div>
        <p v-if="!sessions.length" class="px-2 py-8 text-center text-xs text-slate-400">还没有历史会话</p>
      </div>
      <div class="border-t border-slate-200 p-4">
        <div class="flex items-center gap-3 rounded-xl bg-white p-2.5">
          <CircleUserRound class="size-8 text-blue-500" />
          <span class="min-w-0 flex-1 truncate text-sm font-medium">{{ user.username }}</span>
          <button title="退出登录" class="icon-button" @click="signOut"><LogOut class="size-4" /></button>
        </div>
      </div>
    </aside>

    <section class="workspace">
      <header class="topbar">
        <button class="icon-button lg:hidden" aria-label="打开菜单" @click="mobileSidebar = true"><Menu class="size-5" /></button>
        <div><h2 class="font-bold text-slate-900">知识问答</h2><div class="mt-0.5 flex items-center gap-1.5 text-xs text-emerald-600"><span class="size-1.5 rounded-full bg-emerald-500"></span>知识库已连接</div></div>
        <button class="secondary-button ml-auto" @click="openImportModal()"><CloudUpload class="size-4" />导入文档</button>
      </header>

      <div class="content-grid">
        <section class="chat-column">
          <div ref="messageList" class="message-list">
            <div v-if="!hasConversation" class="empty-state">
              <div class="brand-mark"><Sparkles class="size-7" /></div>
              <h2>今天想了解什么？</h2>
              <p>询问设备使用、故障处理或产品参数，我会从内部资料中查找依据。</p>
            </div>
            <article v-for="message in messages" :key="message.message_id" :class="['message-row', message.role]">
              <div v-if="message.role === 'assistant'" class="avatar"><Bot class="size-4" /></div>
              <div :class="['message-bubble', message.role]">
                <div v-if="message.role === 'assistant' && message.pending && !message.content" class="retrieval-status">
                  <LoaderCircle class="size-3.5 animate-spin" />
                  <span>{{ activeStageLabel }}…</span>
                  <span class="retrieval-count">{{ completedStageCount }}/{{ queryStages.length }}</span>
                </div>
                <div v-else-if="message.role === 'assistant' && message.content_html && !message.pending" class="markdown-body" v-html="message.content_html"></div>
                <div v-else-if="message.content" class="whitespace-pre-wrap">{{ message.content }}<span v-if="message.pending" class="typing-cursor"></span></div>
              </div>
              <div v-if="message.role === 'user'" class="avatar user-avatar" aria-label="用户">
                <CircleUserRound class="size-4" />
              </div>
            </article>
            <p v-if="queryError" class="mx-auto max-w-2xl rounded-xl bg-red-50 px-4 py-3 text-sm text-red-700">{{ queryError }}</p>
          </div>

          <form class="composer" @submit.prevent="submitQuery">
            <div class="composer-box">
              <textarea v-model="query" rows="2" placeholder="输入产品问题…" @keydown.enter.exact.prevent="submitQuery"></textarea>
              <div class="flex items-center gap-2 border-t border-slate-100 pt-2">
                <Paperclip class="size-4 text-slate-400" />
                <div class="mode-switch">
                  <button type="button" :class="mode === 'fast' && 'active'" @click="mode = 'fast'">快速问答</button>
                  <button type="button" :class="mode === 'deep' && 'active'" @click="mode = 'deep'">深度检索</button>
                </div>
                <button class="send-button ml-auto" :disabled="querying || !query.trim()" aria-label="发送"><LoaderCircle v-if="querying" class="size-4 animate-spin" /><Send v-else class="size-4" /></button>
              </div>
            </div>
            <p class="mt-2 text-center text-[11px] text-slate-400">答案由企业资料与检索结果生成，重要信息请核验。</p>
          </form>
        </section>

        <aside class="evidence-panel">
          <div class="panel-heading"><div><h3>检索依据</h3><p>回答所引用的摘要</p></div><Search class="size-4 text-slate-400" /></div>
          <div class="source-tabs">
            <button :class="sourceTab === 'internal' && 'active'" @click="sourceTab = 'internal'"><Database class="size-3.5" />内部知识（{{ sources.filter(s => s.source_type === 'internal').length }}）</button>
            <button :class="sourceTab === 'web' && 'active'" @click="sourceTab = 'web'"><Globe2 class="size-3.5" />联网资料（{{ sources.filter(s => s.source_type === 'web').length }}）</button>
          </div>
          <div class="source-list">
            <article v-for="source in filteredSources" :key="`${source.source_type}-${source.index}`" class="source-card">
              <div class="flex items-start gap-2"><FileText class="mt-0.5 size-4 shrink-0 text-blue-500" /><h4>{{ source.title || `资料 ${source.index}` }}</h4><span v-if="source.score != null" class="score">{{ source.score.toFixed(2) }}</span></div>
              <p>{{ source.summary || "该资料暂无摘要" }}</p>
            </article>
            <div v-if="!filteredSources.length" class="panel-empty"><Search class="size-7" /><p>发送问题后将在这里展示检索依据</p></div>
          </div>
          <div class="progress-panel">
            <div class="flex items-center justify-between"><h3>检索过程</h3><ChevronDown class="size-4 text-slate-400" /></div>
            <div class="mt-4 space-y-3">
              <div v-for="stage in queryStages" :key="stage.key" class="stage-row">
                <span :class="['stage-dot', stage.status]"><LoaderCircle v-if="stage.status === 'running'" class="size-3 animate-spin" /><Check v-else-if="stage.status === 'completed'" class="size-3" /></span>
                <span class="flex-1">{{ stage.label }}</span>
                <span v-if="stage.count != null" class="text-xs text-slate-400">{{ stage.count }} 条</span>
                <span v-else-if="stage.status === 'skipped'" class="text-xs text-slate-400">已跳过</span>
              </div>
            </div>
          </div>
        </aside>
      </div>
    </section>

    <div v-if="importOpen" class="modal-backdrop" @click.self="importOpen = false">
      <section class="modal-card">
        <div class="flex items-start justify-between"><div><p class="eyebrow">KNOWLEDGE IMPORT</p><h2>导入知识文档</h2><p>支持 PDF、Markdown，关闭窗口不会中断后台任务。</p></div><button class="icon-button" @click="importOpen = false"><X class="size-5" /></button></div>
        <div v-if="!importTask" class="import-tabs">
          <button :class="importTab === 'upload' && 'active'" @click="importTab = 'upload'">上传文档</button>
          <button :class="importTab === 'history' && 'active'" @click="importTab = 'history'">导入历史（{{ importTasks.length }}）</button>
        </div>
        <template v-if="!importTask">
          <template v-if="importTab === 'upload'">
            <label class="upload-zone"><CloudUpload class="size-8 text-blue-500" /><span class="font-semibold text-slate-700">选择需要导入的文档</span><span class="text-xs text-slate-400">单个文件最大 100 MB</span><input ref="fileInput" type="file" accept=".pdf,.md" /></label>
            <p v-if="importError" class="error-text">{{ importError }}</p>
            <button class="primary-button w-full" @click="uploadDocument">开始导入</button>
          </template>
          <template v-else>
            <div class="import-history">
              <button v-for="task in importTasks" :key="task.task_id" class="import-history-item" @click="showImportTask(task)">
                <FileText class="size-5 shrink-0 text-blue-500" />
                <span class="min-w-0 flex-1 text-left"><strong>{{ task.file_name }}</strong><small>{{ formatDate(task.created_at) }} · {{ task.progress }}%</small></span>
                <span :class="['task-status', task.status]">{{ statusText(task.status) }}</span>
              </button>
              <div v-if="!importTasks.length" class="panel-empty"><FileText class="size-7" /><p>当前用户还没有导入记录</p></div>
            </div>
            <p v-if="importError" class="error-text">{{ importError }}</p>
          </template>
        </template>
        <template v-else>
          <button class="mb-2 text-xs font-semibold text-blue-600" @click="showImportHistory">← 返回导入历史</button>
          <div class="mt-6 rounded-2xl bg-slate-50 p-4"><div class="flex justify-between gap-4 text-sm"><span class="truncate font-medium">{{ importTask.file_name }}</span><span class="font-semibold text-blue-600">{{ importTask.progress }}%</span></div><div class="mt-3 h-2 overflow-hidden rounded-full bg-slate-200"><div class="h-full rounded-full bg-blue-600 transition-all" :style="{ width: `${importTask.progress}%` }"></div></div></div>
          <div class="my-6 space-y-3">
            <div v-for="stage in importTask.stages" :key="stage.key" class="stage-row text-sm"><span :class="['stage-dot', stage.status]"><LoaderCircle v-if="stage.status === 'running'" class="size-3 animate-spin" /><Check v-else-if="stage.status === 'completed'" class="size-3" /></span><span>{{ stage.label }}</span><span class="ml-auto text-xs text-slate-400">{{ stage.status === 'completed' ? '完成' : stage.status === 'running' ? '处理中' : stage.status === 'skipped' ? '跳过' : '等待' }}</span></div>
          </div>
          <p v-if="importTask.status === 'succeeded'" class="success-text">文档已成功写入知识库，共 {{ importTask.result?.chunk_count || 0 }} 个切片。</p>
          <p v-if="importTask.status === 'failed'" class="error-text">{{ importTask.error || "导入失败" }}</p>
          <p v-if="importError" class="error-text">{{ importError }}</p>
          <button v-if="!importing" class="secondary-button w-full justify-center" @click="importTask = null; importTab = 'upload'; importError = ''">继续导入</button>
        </template>
      </section>
    </div>

    <div v-if="deleteTarget" class="modal-backdrop" @click.self="cancelDelete">
      <section class="modal-card">
        <div class="flex items-start justify-between gap-4">
          <div>
            <p class="eyebrow">DELETE CONVERSATION</p>
            <h2>删除这个对话？</h2>
            <p>“{{ deleteTarget.title }}”的全部消息将被永久删除，无法恢复。</p>
          </div>
          <button class="icon-button" aria-label="关闭" @click="cancelDelete"><X class="size-5" /></button>
        </div>
        <p v-if="deleteError" class="error-text mt-5">{{ deleteError }}</p>
        <div class="mt-6 flex gap-3">
          <button class="secondary-button flex-1 justify-center" :disabled="deleting" @click="cancelDelete">取消</button>
          <button class="danger-button flex-1 justify-center" :disabled="deleting" @click="confirmDelete">
            <LoaderCircle v-if="deleting" class="size-4 animate-spin" />
            {{ deleting ? "删除中" : "删除" }}
          </button>
        </div>
      </section>
    </div>
  </div>
</template>
