import { useState } from 'react'
import { getStatusIcon } from './icons/EventIcons'
import { ChevronRight } from 'lucide-react'
import './TaskTree.css'

interface Task {
  id: string
  description: string
  status: 'pending' | 'in_progress' | 'completed' | 'failed'
  subtasks?: Task[]
}

interface TaskTreeProps {
  tasks: Task[]
  isCollapsed?: boolean
  onToggle?: () => void
}

const statusLabels: Record<Task['status'], string> = {
  pending: '待处理',
  in_progress: '进行中',
  completed: '已完成',
  failed: '已失败'
}

interface TaskNodeProps {
  task: Task
  level?: number
  displayIndex?: string
}

function TaskNode({ task, level = 0, displayIndex }: TaskNodeProps) {
  const [isExpanded, setIsExpanded] = useState(false)

  const getStatusClass = (status: string) => {
    return `task-status-${status}`
  }
  
  const StatusIconComponent = getStatusIcon(task.status)
  const hasSubtasks = task.subtasks && task.subtasks.length > 0
  const isTopLevel = level === 0
  const statusLabel = statusLabels[task.status]

  return (
    <div className={`task-node level-${level} animate-fade-in ${isTopLevel ? 'task-node-root' : ''}`}>
      <div className={`task-item transition-smooth ${getStatusClass(task.status)} ${isTopLevel ? 'task-item-root' : ''}`}>
        {hasSubtasks && (
          <button
            className={`expand-button ${isExpanded ? 'expanded' : 'collapsed'} transition-smooth`}
            onClick={() => setIsExpanded(!isExpanded)}
            aria-label={isExpanded ? '折叠' : '展开'}
          >
            <ChevronRight size={14} />
          </button>
        )}
        <span className={`task-icon-wrapper ${isTopLevel ? 'task-icon-root' : ''}`}>
          <StatusIconComponent size={16} />
        </span>
        {isTopLevel ? (
          <div className="task-card-content">
            <div className="task-card-header">
              {displayIndex && <span className="task-index">{displayIndex}</span>}
              <span className="task-title">{task.description}</span>
              <span className={`task-status-pill status-${task.status}`}>{statusLabel}</span>
            </div>
            {hasSubtasks && (
              <div className="task-card-meta">
                <span className="task-card-meta-item">{task.subtasks!.length} 个子任务</span>
              </div>
            )}
          </div>
        ) : (
          <span className="task-description">{task.description}</span>
        )}
      </div>
      
      {hasSubtasks && isExpanded && (
        <div className="task-children">
          {task.subtasks!.map((subtask) => (
            <TaskNode key={subtask.id} task={subtask} level={level + 1} />
          ))}
        </div>
      )}
    </div>
  )
}

function TaskTree({ tasks, isCollapsed = true, onToggle }: TaskTreeProps) {
  const getTaskStats = (tasks: Task[]): { total: number; completed: number } => {
    let total = 0
    let completed = 0

    const countTasks = (taskList: Task[]) => {
      taskList.forEach(task => {
        total++
        if (task.status === 'completed') completed++
        if (task.subtasks) countTasks(task.subtasks)
      })
    }

    countTasks(tasks)
    return { total, completed }
  }

  const stats = getTaskStats(tasks)
  const progress = stats.total > 0 ? Math.round((stats.completed / stats.total) * 100) : 0

  return (
    <div className={`task-tree ${isCollapsed ? 'collapsed' : 'expanded'}`}>
      <div className="task-tree-header transition-smooth" onClick={onToggle}>
        <div className="task-tree-header-left">
          <button className={`collapse-button ${isCollapsed ? 'collapsed' : 'expanded'} transition-smooth`}>
            <ChevronRight size={16} />
          </button>
          <span className="task-tree-title">任务进度</span>
        </div>
        <span className="task-tree-stats">
          {stats.completed}/{stats.total} ({progress}%)
        </span>
      </div>

      {!isCollapsed && (
        <div className="task-tree-content">
          {tasks.length === 0 ? (
            <div className="empty-tasks">
              <p>暂无任务</p>
            </div>
          ) : (
            <>
              {/* 进度条 */}
              <div className="progress-bar-wrapper">
                <div className="progress-bar">
                  <div 
                    className="progress-fill" 
                    style={{ width: `${progress}%` }}
                  />
                </div>
              </div>
              {/* 任务列表 */}
              {tasks.map((task, index) => (
                <TaskNode key={task.id} task={task} displayIndex={`${index + 1}`.padStart(2, '0')} />
              ))}
            </>
          )}
        </div>
      )}
    </div>
  )
}

export default TaskTree

