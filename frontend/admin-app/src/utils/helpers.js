export const formatDate = (dateString) => {
  if (!dateString) return 'N/A'
  const date = new Date(dateString)
  return date.toLocaleDateString() + ' ' + date.toLocaleTimeString()
}

export const formatBytes = (bytes) => {
  if (bytes === 0) return '0 Bytes'
  const k = 1024
  const sizes = ['Bytes', 'KB', 'MB', 'GB']
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i]
}

export const truncate = (str, length = 100) => {
  if (!str) return ''
  return str.length > length ? str.substring(0, length) + '...' : str
}

export const getStatusColor = (status) => {
  const colors = {
    pending: '#f59e0b',
    processing: '#3b82f6',
    completed: '#10b981',
    indexed: '#10b981',
    failed: '#ef4444'
  }
  return colors[status] || '#64748b'
}