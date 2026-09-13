import type { ImportTask, SessionSummary, User } from "./types"

const jsonHeaders = { "Content-Type": "application/json" }

async function parseError(response: Response): Promise<string> {
  try {
    const body = await response.json()
    return body.detail || body.message || `请求失败 (${response.status})`
  } catch {
    return `请求失败 (${response.status})`
  }
}

export async function apiFetch<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, { credentials: "include", ...init })
  if (!response.ok) throw new Error(await parseError(response))
  if (response.status === 204) return undefined as T
  return response.json() as Promise<T>
}

export const authApi = {
  me: () => apiFetch<User>("/api/v1/auth/me"),
  login: (username: string, password: string) =>
    apiFetch<User>("/api/v1/auth/login", {
      method: "POST",
      headers: jsonHeaders,
      body: JSON.stringify({ username, password }),
    }),
  logout: () => apiFetch<void>("/api/v1/auth/logout", { method: "POST" }),
}

export const sessionApi = {
  list: () => apiFetch<SessionSummary[]>("/api/v1/sessions"),
  messages: (id: string) =>
    apiFetch<{ session_id: string; messages: unknown[] }>(
      `/api/v1/sessions/${encodeURIComponent(id)}/messages`,
    ),
}

export async function createImportTask(file: File): Promise<ImportTask> {
  const form = new FormData()
  form.append("file", file)
  return apiFetch<ImportTask>("/api/v1/imports/tasks", {
    method: "POST",
    body: form,
  })
}

export async function getImportTask(taskId: string): Promise<ImportTask> {
  return apiFetch<ImportTask>(`/api/v1/imports/tasks/${taskId}`)
}

export async function listImportTasks(): Promise<ImportTask[]> {
  return apiFetch<ImportTask[]>("/api/v1/imports/tasks")
}

export async function streamQuery(
  body: Record<string, unknown>,
  onEvent: (event: string, data: Record<string, unknown>) => void,
): Promise<void> {
  const response = await fetch("/api/v1/queries/stream", {
    method: "POST",
    credentials: "include",
    headers: jsonHeaders,
    body: JSON.stringify(body),
  })
  if (!response.ok) throw new Error(await parseError(response))
  if (!response.body) throw new Error("浏览器未提供流式响应")

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ""
  while (true) {
    const { value, done } = await reader.read()
    buffer += decoder.decode(value || new Uint8Array(), { stream: !done })
    const blocks = buffer.split("\n\n")
    buffer = blocks.pop() || ""
    for (const block of blocks) {
      let event = "message"
      const dataLines: string[] = []
      for (const line of block.split("\n")) {
        if (line.startsWith("event:")) event = line.slice(6).trim()
        if (line.startsWith("data:")) dataLines.push(line.slice(5).trim())
      }
      if (!dataLines.length) continue
      onEvent(event, JSON.parse(dataLines.join("\n")))
    }
    if (done) break
  }
}
