import {
  useRef,
  useEffect,
  useState,
  Children,
  cloneElement,
  isValidElement,
  Fragment,
  type ReactNode,
} from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { ChevronDown, CheckCircle2, Loader2, ClipboardList } from 'lucide-react'
import { getEventIcon } from './icons/EventIcons'
import './EventList.css'

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

interface EventListProps {
  events: UserEvent[]
  autoScroll?: boolean
}

interface EventCardProps {
  event: UserEvent
  index: number
}

function EventCard({ event, index }: EventCardProps) {
  const [isExpanded, setIsExpanded] = useState(false)
  const MAX_PREVIEW_LENGTH = 200 // 预览最多显示200个字符
  const emojiRegex = /(✅|⏳|📝)/g

  const emojiIconMap: Record<string, { Icon: any; className: string }> = {
    '✅': { Icon: CheckCircle2, className: 'text-green-500' },
    '⏳': { Icon: Loader2, className: 'text-blue-500' },
    '📝': { Icon: ClipboardList, className: 'text-amber-500' },
  }

  const getEventClassName = (event: UserEvent) => {
    const baseClass = 'event-card animate-slide-in'
    const levelClass = `level-${event.level}`
    const typeClass = `type-${event.type}`
    return `${baseClass} ${levelClass} ${typeClass}`
  }
  
  // 获取事件图标组件
  const IconComponent = getEventIcon(event.type);

  const renderTextWithIcons = (text: string, keyPrefix: string): ReactNode[] => {
    const segments = text.split(emojiRegex)
    return segments.map((segment, segmentIndex) => {
      const mapping = emojiIconMap[segment]
      if (mapping) {
        const { Icon, className } = mapping
        return (
          <Icon
            key={`${keyPrefix}-icon-${segmentIndex}`}
            className={`emoji-icon ${className}`}
            size={16}
            strokeWidth={2}
          />
        )
      }
      if (!segment) {
        return <Fragment key={`${keyPrefix}-empty-${segmentIndex}`} />
      }
      return (
        <Fragment key={`${keyPrefix}-text-${segmentIndex}`}>
          {segment}
        </Fragment>
      )
    })
  }

  const renderNodesWithIcons = (nodes: ReactNode, keyPrefix: string): ReactNode => {
    return Children.map(nodes, (child, childIndex) => {
      if (typeof child === 'string') {
        return renderTextWithIcons(child, `${keyPrefix}-${childIndex}`)
      }
      if (isValidElement(child)) {
        const props = child.props as any
        const updatedChildren = props?.children
          ? renderNodesWithIcons(props.children, `${keyPrefix}-${childIndex}`)
          : props?.value
            ? renderTextWithIcons(String(props.value), `${keyPrefix}-${childIndex}`)
            : undefined

        if (updatedChildren) {
          return cloneElement(child, {
            children: updatedChildren,
          } as any)
        }
      }
      return child
    })
  }

  const formatTime = (timestamp: string) => {
    const date = new Date(timestamp)
    return date.toLocaleTimeString('zh-CN', { 
      hour: '2-digit', 
      minute: '2-digit',
      second: '2-digit'
    })
  }

  // 判断内容是否需要折叠
  const needsCollapse = event.content.length > MAX_PREVIEW_LENGTH

  // 获取显示的内容（特殊处理任务更新事件）
  const getDisplayContent = () => {
    // 如果是任务更新事件，显示友好摘要
    if (event.type === 'task_update' && !isExpanded) {
      try {
        const tasks = JSON.parse(event.content)
        const taskCount = tasks.length
        return `已更新 ${taskCount} 个任务，点击展开查看详情`
      } catch (e) {
        // JSON 解析失败，使用默认逻辑
      }
    }

    // 如果是任务更新事件且已展开，优先使用 Markdown 内容
    if (event.type === 'task_update' && isExpanded && event.details?.markdown_content) {
      return event.details.markdown_content
    }

    if (!needsCollapse || isExpanded) {
      return event.content
    }
    // 显示前200个字符 + "..."
    return event.content.slice(0, MAX_PREVIEW_LENGTH) + '...'
  }

  return (
    <div key={index} className={getEventClassName(event)}>
      <div className="event-header">
        <div className="event-icon-wrapper">
          {IconComponent && <IconComponent size={18} className={`icon-color-${event.type}`} />}
        </div>
        <span className="event-title">{event.title}</span>
        <span className="event-time">{formatTime(event.timestamp)}</span>
      </div>
      <div className="event-content markdown-content">
        <ReactMarkdown
          remarkPlugins={[remarkGfm]}
          components={{
            p: ({ children }) => <p>{renderNodesWithIcons(children, `p-${index}`)}</p>,
            li: ({ children }) => <li>{renderNodesWithIcons(children, `li-${index}`)}</li>,
            h3: ({ children }) => <h3>{renderNodesWithIcons(children, `h3-${index}`)}</h3>,
          }}
        >
          {getDisplayContent()}
        </ReactMarkdown>
      </div>
      {needsCollapse && (
        <button 
          className="event-toggle transition-smooth"
          onClick={() => setIsExpanded(!isExpanded)}
        >
          {isExpanded ? '收起' : '展开查看更多'}
        </button>
      )}
    </div>
  )
}

function EventList({ events, autoScroll = true }: EventListProps) {
  const endRef = useRef<HTMLDivElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const [userScrolledUp, setUserScrolledUp] = useState(false)

  // 监听用户滚动行为
  useEffect(() => {
    const container = containerRef.current
    if (!container) return

    const handleScroll = () => {
      const { scrollTop, scrollHeight, clientHeight } = container
      const distanceFromBottom = scrollHeight - scrollTop - clientHeight
      
      // 如果距离底部超过 100px，认为用户向上滚动了
      if (distanceFromBottom > 100) {
        setUserScrolledUp(true)
      } else {
        setUserScrolledUp(false)
      }
    }

    container.addEventListener('scroll', handleScroll)
    return () => container.removeEventListener('scroll', handleScroll)
  }, [])

  // 智能自动滚动：只有当用户在底部时才滚动
  useEffect(() => {
    if (!autoScroll || userScrolledUp || !endRef.current) return
    
    endRef.current.scrollIntoView({ behavior: 'smooth' })
  }, [events, autoScroll, userScrolledUp])

  return (
    <div className="event-list" ref={containerRef}>
      {events.length === 0 ? (
        <div className="empty-state">
          <p>Waiting for events...</p>
        </div>
      ) : (
        <>
          {events.map((event, index) => (
            <EventCard 
              key={`${event.timestamp}-${event.type}-${index}`} 
              event={event} 
              index={index} 
            />
          ))}
          <div ref={endRef} />
        </>
      )}
      
      {/* 回到底部按钮 - 智能定位方式 */}
      {userScrolledUp && (
        <div className="scroll-to-bottom-container">
          <button 
            className="scroll-to-bottom"
            onClick={() => {
              endRef.current?.scrollIntoView({ behavior: 'smooth' })
              setUserScrolledUp(false)
            }}
            title="回到底部"
            aria-label="回到底部"
          >
            <ChevronDown size={16} />
          </button>
        </div>
      )}
    </div>
  )
}

export default EventList

