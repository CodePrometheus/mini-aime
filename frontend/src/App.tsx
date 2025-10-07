import { useState, useEffect, useRef } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { Send, Loader2 } from 'lucide-react'
import EventList from './components/EventList'
import TaskTree from './components/TaskTree'
import './App.css'

interface UserEvent {
  type: string
  title: string
  content: string
  timestamp: string
  level: string
  collapsible: boolean
  icon: string
  metadata?: {
    agent_id?: string
    task_id?: string
  }
  details?: Record<string, any>
}

interface Task {
  id: string
  description: string
  status: 'pending' | 'in_progress' | 'completed' | 'failed'
  subtasks?: Task[]
}

function App() {
  const [inputValue, setInputValue] = useState('')
  const [taskId, setTaskId] = useState<string | null>(null)
  const [events, setEvents] = useState<UserEvent[]>([])
  const [tasks, setTasks] = useState<Task[]>([])
  const [result, setResult] = useState<string>('')
  const [loading, setLoading] = useState(false)
  const [leftWidth, setLeftWidth] = useState(35) // 左侧宽度百分比
  const [isDragging, setIsDragging] = useState(false)
  const [isTaskTreeCollapsed, setIsTaskTreeCollapsed] = useState(true)
  const containerRef = useRef<HTMLDivElement>(null)
  const eventSourceRef = useRef<EventSource | null>(null)

  // 页面加载时，尝试恢复上次的任务
  useEffect(() => {
    const savedTaskId = localStorage.getItem('activeTaskId')
    if (savedTaskId && !taskId) {
      console.log('Restoring task from localStorage:', savedTaskId)
      setTaskId(savedTaskId)
      setLoading(true)
    }
  }, [])

  // 订阅事件流（StrictMode 下防重复订阅）
  useEffect(() => {
    if (!taskId) return
    
    // StrictMode 会导致 effect 运行两次，使用 ref 防止重复创建
    if (eventSourceRef.current) {
      console.log('EventSource already exists, skipping creation')
      return
    }

    console.log('Creating EventSource connection for task:', taskId)
    const eventSource = new EventSource(`/api/tasks/${taskId}/events`)
    eventSourceRef.current = eventSource

    eventSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data)
        
        // 连接确认
        if (data.type === 'connected') {
          console.log('Event stream connected:', data.task_id)
          return
        }
        
        // 流结束
        if (data.type === 'stream_ended') {
          console.log('Event stream ended:', data.task_status)
          eventSource.close()
          eventSourceRef.current = null
          fetchResult(taskId)
          setLoading(false)
          // 任务完成，清除 localStorage
          localStorage.removeItem('activeTaskId')
          return
        }
        
        // 超时
        if (data.type === 'timeout') {
          console.log('Event stream timeout')
          eventSource.close()
          eventSourceRef.current = null
          setLoading(false)
          // 超时也清除
          localStorage.removeItem('activeTaskId')
          return
        }
        
        // 用户事件
        if (data.type === 'user_event' && data.data) {
          const userEvent = data.data
          
          // 添加事件到列表（去重：检查是否已存在相同timestamp和type的事件）
          setEvents((prev) => {
            const isDuplicate = prev.some(
              (e) => e.timestamp === userEvent.timestamp && 
                     e.type === userEvent.type &&
                     e.title === userEvent.title
            )
            return isDuplicate ? prev : [...prev, userEvent]
          })
          
          // 如果是任务更新事件，解析并更新任务树
          if (userEvent.type === 'task_update') {
            try {
              const taskData = JSON.parse(userEvent.content)
              setTasks(taskData)
            } catch (e) {
              console.error('Failed to parse task update:', e)
            }
          }
        }
      } catch (error) {
        console.error('Failed to parse event:', error)
      }
    }

    eventSource.onerror = (error) => {
      console.error('EventSource error:', error)
      eventSource.close()
      eventSourceRef.current = null
      fetchResult(taskId)
      setLoading(false)
      // 只有在任务真正完成或失败时才清除 localStorage
      // 不要因为连接错误就清除，这样页面刷新后还能恢复
    }

    return () => {
      if (eventSourceRef.current) {
        eventSourceRef.current.close()
        eventSourceRef.current = null
      }
    }
  }, [taskId])

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (!isDragging || !containerRef.current) return

      const containerRect = containerRef.current.getBoundingClientRect()
      const newLeftWidth = ((e.clientX - containerRect.left) / containerRect.width) * 100
      
      // 限制在 20% - 80% 之间
      if (newLeftWidth >= 20 && newLeftWidth <= 80) {
        setLeftWidth(newLeftWidth)
      }
    }

    const handleMouseUp = () => {
      setIsDragging(false)
    }

    if (isDragging) {
      document.addEventListener('mousemove', handleMouseMove)
      document.addEventListener('mouseup', handleMouseUp)
      document.body.style.cursor = 'col-resize'
      document.body.style.userSelect = 'none'
    }

    return () => {
      document.removeEventListener('mousemove', handleMouseMove)
      document.removeEventListener('mouseup', handleMouseUp)
      document.body.style.cursor = ''
      document.body.style.userSelect = ''
    }
  }, [isDragging])

  const fetchResult = async (id: string) => {
    try {
      const response = await fetch(`/api/tasks/${id}/result`)
      const data = await response.json()
      if (data.content) {
        setResult(data.content)
      }
    } catch (error) {
      console.error('Failed to fetch result:', error)
    } finally {
      setLoading(false)
    }
  }

  const handleSubmit = async () => {
    const goal = inputValue.trim()
    if (!goal || loading) return

    setLoading(true)
    setEvents([])
    setTasks([])
    setResult('')
    setTaskId(null)
    setInputValue('')

    try {
      const response = await fetch('/api/tasks/submit', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ goal }),
      })

      const data = await response.json()
      const newTaskId = data.task_id
      
      // 保存到 localStorage，刷新后可以恢复
      localStorage.setItem('activeTaskId', newTaskId)
      setTaskId(newTaskId)
    } catch (error) {
      console.error('Failed to submit task:', error)
      alert('提交任务失败')
      setLoading(false)
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit()
    }
  }



  return (
    <div className="app" ref={containerRef}>
      {/* 左侧面板 */}
      <div className="left-panel" style={{ width: `${leftWidth}%` }}>
        {/* 事件流容器 */}
        <EventList events={events} autoScroll={true} />

        {/* 底部区域：任务树 + 输入框 */}
        <div className="bottom-section">
          {/* 任务树 */}
          {tasks.length > 0 && (
            <div className="task-tree-wrapper">
              <TaskTree 
                tasks={tasks}
                isCollapsed={isTaskTreeCollapsed}
                onToggle={() => setIsTaskTreeCollapsed(!isTaskTreeCollapsed)}
              />
            </div>
          )}

          {/* 输入框 */}
          <div className="input-section">
            <div className="input-wrapper">
              <textarea
                className="chat-input transition-smooth"
                placeholder="Ask Mini-Aime"
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                onKeyDown={handleKeyDown}
                disabled={loading}
                rows={1}
              />
              <button 
                className="send-button transition-smooth"
                onClick={handleSubmit}
                disabled={loading || !inputValue.trim()}
                title="发送消息"
              >
                {loading ? (
                  <Loader2 className="animate-spin" size={16} strokeWidth={2.5} />
                ) : (
                  <Send size={16} strokeWidth={2.5} />
                )}
              </button>
            </div>
            {loading && (
              <div className="input-hint animate-pulse">
                <Loader2 className="animate-spin" size={14} />
                <span>正在执行任务...</span>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* 可拖拽的分割线 */}
      <div
        className={`resizer ${isDragging ? 'active' : ''}`}
        onMouseDown={() => setIsDragging(true)}
      />

      {/* 右侧面板 */}
      <div className="right-panel" style={{ width: `${100 - leftWidth}%` }}>
        <div className="result-container">
          {loading && !result && (
            <div className="empty-message">
              <div className="empty-icon animate-fade-up">
                <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                  <defs>
                    <linearGradient id="brand-grad" x1="0" y1="0" x2="1" y2="1">
                      <stop offset="0%" stopColor="#667eea" />
                      <stop offset="100%" stopColor="#764ba2" />
                    </linearGradient>
                  </defs>
                  <g stroke="url(#brand-grad)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                    <path d="M14 2v6h6" />
                    <path d="M16 13H8" />
                    <path d="M16 17H8" />
                    <path d="M10 9H8" />
                  </g>
                </svg>
              </div>
              Waiting for task to complete...
            </div>
          )}
          {!loading && !result && (
            <div className="empty-message">
              <div className="empty-icon animate-fade-up">
                <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                  <defs>
                    <linearGradient id="brand-grad-2" x1="0" y1="0" x2="1" y2="1">
                      <stop offset="0%" stopColor="#667eea" />
                      <stop offset="100%" stopColor="#764ba2" />
                    </linearGradient>
                  </defs>
                  <g stroke="url(#brand-grad-2)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                    <path d="M14 2v6h6" />
                    <path d="M16 13H8" />
                    <path d="M16 17H8" />
                    <path d="M10 9H8" />
                  </g>
                </svg>
              </div>
              The result will appear here once the task is complete
            </div>
          )}
          {result && (
            <div className="markdown-content">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>{result}</ReactMarkdown>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default App