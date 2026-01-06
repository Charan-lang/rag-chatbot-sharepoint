import React from 'react'
import { useChat } from '../hooks/useChat'
import MessageList from './MessageList'
import MessageInput from './MessageInput'

export default function ChatInterface() {
  const { messages, loading, error, sendMessage, clearMessages } = useChat()

  return (
    <div style={{ 
      height: '100%', 
      display: 'flex', 
      flexDirection: 'column',
      background: '#f9fafb'
    }}>
      <MessageList messages={messages} loading={loading} />
      <MessageInput onSend={sendMessage} loading={loading} onClear={clearMessages} />
      
      {error && (
        <div style={{ 
          position: 'fixed', 
          top: '80px', 
          left: '50%', 
          transform: 'translateX(-50%)',
          background: '#fee2e2',
          color: '#dc2626',
          padding: '12px 24px',
          borderRadius: '8px',
          boxShadow: '0 4px 6px rgba(0,0,0,0.1)',
          zIndex: 1000
        }}>
          {error}
        </div>
      )}
    </div>
  )
}