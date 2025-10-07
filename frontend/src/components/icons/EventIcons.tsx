/**
 * Mini-Aime 图标组件库
 * 基于 Lucide Icons，统一管理所有图标样式
 */

import {
  Award,
  AlertCircle,
  CheckCircle,
  CheckCircle2,
  Loader2,
  Circle,
  XCircle,
  Clock,
  Pause,
  Download,
  Upload,
  Save,
  Share2,
  Copy,
  Link2,
  Send,
  Paperclip,
  Mic,
  Image,
  Folder,
  BarChart3,
  LineChart,
  PieChart,
  TrendingUp,
  Sparkles,
  Wand2,
  BrainCircuit,
  ChevronDown,
  ChevronRight,
  ChevronUp,
  ChevronLeft,
  ArrowLeft,
  Menu,
  Edit3,
  Trash2,
  Plus,
  Minus,
  Settings,
  Filter,
  Search,
  MoreVertical,
  X,
  Brain,
  Lightbulb,
  Zap,
  Rocket,
  Eye,
  FileSearch,
  Trophy,
  Flag,
  FileText,
  type LucideProps,
} from 'lucide-react'

/**
 * 图标组件的统一属性接口
 */
export interface IconProps extends Omit<LucideProps, 'ref'> {
  className?: string
  size?: number
}

/**
 * ReAct 流程图标
 * 用于展示 AI 的思考、行动、观察过程
 */
export const EventIcons = {
  /**
   * 思考图标 - 紫色灯泡
   */
  Thought: ({ className = '', size = 20, ...props }: IconProps) => (
    <Lightbulb 
      size={size}
      className={`text-purple-500 ${className}`}
      strokeWidth={2}
      {...props}
    />
  ),
  
  /**
   * 动作图标 - 蓝色闪电
   */
  Action: ({ className = '', size = 20, ...props }: IconProps) => (
    <Zap 
      size={size}
      className={`text-blue-500 ${className}`}
      strokeWidth={2}
      fill="currentColor"
      {...props}
    />
  ),
  
  /**
   * 观察图标 - 绿色眼睛
   */
  Observation: ({ className = '', size = 20, ...props }: IconProps) => (
    <Eye 
      size={size}
      className={`text-green-500 ${className}`}
      strokeWidth={2}
      {...props}
    />
  ),
  
  /**
   * 里程碑图标 - 橙色奖杯
   */
  Milestone: ({ className = '', size = 20, ...props }: IconProps) => (
    <Trophy 
      size={size}
      className={`text-orange-500 ${className}`}
      strokeWidth={2}
      {...props}
    />
  ),
  
  /**
   * 错误图标 - 红色警告
   */
  Error: ({ className = '', size = 20, ...props }: IconProps) => (
    <AlertCircle 
      size={size}
      className={`text-red-500 ${className}`}
      strokeWidth={2}
      {...props}
    />
  ),
  
  /**
   * 信息图标 - 蓝色火花
   */
  Info: ({ className = '', size = 20, ...props }: IconProps) => (
    <Sparkles 
      size={size}
      className={`text-blue-500 ${className}`}
      strokeWidth={2}
      {...props}
    />
  ),
  
  /**
   * 成功图标 - 绿色勾选
   */
  Success: ({ className = '', size = 20, ...props }: IconProps) => (
    <CheckCircle 
      size={size}
      className={`text-green-500 ${className}`}
      strokeWidth={2}
      {...props}
    />
  ),
}

/**
 * 任务状态图标
 * 用于展示任务的执行状态
 */
export const StatusIcons = {
  /**
   * 待执行 - 灰色圆圈
   */
  Pending: ({ className = '', size = 16, ...props }: IconProps) => (
    <Circle 
      size={size}
      className={`text-gray-400 ${className}`}
      strokeWidth={2}
      {...props}
    />
  ),
  
  /**
   * 进行中 - 蓝色旋转
   */
  InProgress: ({ className = '', size = 16, ...props }: IconProps) => (
    <Loader2 
      size={size}
      className={`text-blue-500 animate-spin ${className}`}
      strokeWidth={2}
      {...props}
    />
  ),
  
  /**
   * 已完成 - 绿色勾选
   */
  Completed: ({ className = '', size = 16, ...props }: IconProps) => (
    <CheckCircle2 
      size={size}
      className={`text-green-500 ${className}`}
      strokeWidth={2}
      {...props}
    />
  ),
  
  /**
   * 失败 - 红色叉号
   */
  Failed: ({ className = '', size = 16, ...props }: IconProps) => (
    <XCircle 
      size={size}
      className={`text-red-500 ${className}`}
      strokeWidth={2}
      {...props}
    />
  ),
  
  /**
   * 计划中 - 灰色时钟
   */
  Scheduled: ({ className = '', size = 16, ...props }: IconProps) => (
    <Clock 
      size={size}
      className={`text-gray-500 ${className}`}
      strokeWidth={2}
      {...props}
    />
  ),
  
  /**
   * 已暂停 - 灰色暂停
   */
  Paused: ({ className = '', size = 16, ...props }: IconProps) => (
    <Pause 
      size={size}
      className={`text-gray-500 ${className}`}
      strokeWidth={2}
      {...props}
    />
  ),
}

/**
 * 工具栏图标
 * 用于各种操作按钮
 */
export const ToolbarIcons = {
  // 文件操作
  Download,
  Upload,
  Save,
  FileText,
  Folder,
  
  // 分享与复制
  Share2,
  Copy,
  Link2,
  
  // 数据与分析
  BarChart3,
  LineChart,
  PieChart,
  TrendingUp,
  
  // 交互操作
  Send,
  Paperclip,
  Mic,
  Image,
  
  // AI 特性
  Sparkles,
  Wand2,
  BrainCircuit,
  Brain,
  Lightbulb,
  Rocket,
  
  // 导航
  ChevronDown,
  ChevronRight,
  ChevronUp,
  ChevronLeft,
  ArrowLeft,
  Menu,
  
  // 编辑
  Edit3,
  Trash2,
  Plus,
  Minus,
  X,
  
  // 设置与搜索
  Settings,
  Filter,
  Search,
  FileSearch,
  Eye,
  MoreVertical,
  
  // 奖励与成就
  Trophy,
  Flag,
  Award,
  Zap,
}

/**
 * 带背景的图标组件
 * 用于需要突出显示的场景
 */
export interface IconWithBackgroundProps extends IconProps {
  variant?: 'rounded' | 'circle'
  bgColor?: string
}

export const IconWithBackground = ({
  icon: Icon,
  variant = 'rounded',
  bgColor = 'bg-gray-50',
  className = '',
  size = 20,
  ...props
}: IconWithBackgroundProps & { icon: React.ComponentType<IconProps> }) => {
  const containerClass = variant === 'circle' ? 'rounded-full' : 'rounded-lg'
  
  return (
    <div className={`flex items-center justify-center w-10 h-10 ${containerClass} ${bgColor} transition-smooth hover:scale-110`}>
      <Icon size={size} className={className} {...props} />
    </div>
  )
}

/**
 * 根据事件类型获取对应图标
 */
export const getEventIcon = (eventType: string): React.ComponentType<IconProps> => {
  const iconMap: Record<string, React.ComponentType<IconProps>> = {
    thought: EventIcons.Thought,
    action: EventIcons.Action,
    observation: EventIcons.Observation,
    milestone: EventIcons.Milestone,
    error: EventIcons.Error,
    info: EventIcons.Info,
    success: EventIcons.Success,
  }
  
  return iconMap[eventType] || EventIcons.Info
}

/**
 * 根据任务状态获取对应图标
 */
export const getStatusIcon = (status: string): React.ComponentType<IconProps> => {
  const iconMap: Record<string, React.ComponentType<IconProps>> = {
    pending: StatusIcons.Pending,
    in_progress: StatusIcons.InProgress,
    completed: StatusIcons.Completed,
    failed: StatusIcons.Failed,
    scheduled: StatusIcons.Scheduled,
    paused: StatusIcons.Paused,
  }
  
  return iconMap[status] || StatusIcons.Pending
}

