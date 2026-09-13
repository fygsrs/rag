export interface User {
  id: string
  username: string
  created_at?: string
}

export interface Source {
  index: number
  id?: string | number
  title: string
  url: string
  score?: number
  source: string
  source_type: "internal" | "web"
  summary: string
  metadata: Record<string, unknown>
}

export interface ChatMessage {
  message_id: string
  role: "user" | "assistant"
  content: string
  content_html?: string
  metadata?: Record<string, unknown>
  pending?: boolean
}

export interface SessionSummary {
  session_id: string
  title: string
  last_message: string
  updated_at?: string
  message_count: number
}

export interface ProgressStage {
  key: string
  label: string
  status: "pending" | "running" | "completed" | "skipped" | "failed"
  count?: number
}

export interface ImportTask {
  task_id: string
  file_name: string
  status: "queued" | "running" | "succeeded" | "failed"
  current_stage: string
  progress: number
  stages: ProgressStage[]
  result?: Record<string, unknown>
  error: string
  revision: number
  created_at?: string
  updated_at?: string
}
