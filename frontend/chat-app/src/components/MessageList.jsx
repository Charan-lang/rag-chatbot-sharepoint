import React, { useEffect, useRef } from 'react'
import ReactMarkdown from 'react-markdown'
import SourcesList from './SourcesList'
import { Bot, User } from 'lucide-react'

export default function MessageList({ messages, loading }) {
  const messagesEndRef = useRef(null)

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  if (messages.length === 0) {
    return (
      <div style={{ 
        flex: 1, 
        display: 'flex', 
        flexDirection: 'column',
        justifyContent: 'center', 
        alignItems: 'center',
        padding: '40px'
      }}>
        <Bot size={64} color="#9ca3af" style={{ marginBottom: '24px' }} />
        <h2 style={{ fontSize: '24px', fontWeight: '600', color: '#1f2937', marginBottom: '12px' }}>
          Welcome to RAG Chat
        </h2>
        <p style={{ fontSize: '16px', color: '#6b7280', textAlign: 'center', maxWidth: '500px' }}>
          Ask me anything about your documents. I'll search through your accessible content and provide answers with source citations.
        </p>
        <div style={{ marginTop: '32px', display: 'flex', flexDirection: 'column', gap: '12px', width: '100%', maxWidth: '500px' }}>
          <div style={{ padding: '16px', background: 'white', borderRadius: '8px', border: '1px solid #e5e7eb' }}>
            <p style={{ fontSize: '14px', color: '#374151' }}>
              💡 <strong>Tip:</strong> I only search documents you have permission to access
            </p>
          </div>
          <div style={{ padding: '16px', background: 'white', borderRadius: '8px', border: '1px solid #e5e7eb' }}>
            <p style={{ fontSize: '14px', color: '#374151' }}>
              🔒 <strong>Secure:</strong> All responses are grounded in your authorized content
            </p>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div style={{ 
      flex: 1, 
      overflow: 'auto', 
      padding: '24px',
      display: 'flex',
      flexDirection: 'column',
      gap: '20px'
    }}>
      {messages.map((message) => (
        <div 
          key={message.id}
          style={{ 
            display: 'flex',
            gap: '16px',
            alignItems: 'flex-start',
            ...(message.role === 'user' ? { flexDirection: 'row-reverse' } : {})
          }}
        >
          {/* Avatar */}
          <div style={{
            width: '40px',
            height: '40px',
            borderRadius: '50%',
            background: message.role === 'user' ? '#eff6ff' : '#f3f4f6',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            flexShrink: 0
          }}>
            {message.role === 'user' ? (
              <User size={20} color="#2563eb" />
            ) : (
              <Bot size={20} color="#6b7280" />
            )}
          </div>

          {/* Message Content */}
          <div style={{ 
            flex: 1,
            maxWidth: '70%',
            ...(message.role === 'user' ? { textAlign: 'right' } : {})
          }}>
            <div style={{
              display: 'inline-block',
              padding: '16px 20px',
              borderRadius: '12px',
              background: message.role === 'user' ? '#2563eb' : 'white',
              color: message.role === 'user' ? 'white' : '#1f2937',
              textAlign: 'left',
              boxShadow: '0 1px 3px rgba(0,0,0,0.1)'
            }}>
              <div className="message-content">
                <ReactMarkdown>{message.content}</ReactMarkdown>
              </div>
              
              {message.sources && message.sources.length > 0 && (
                <SourcesList sources={message.sources} />
              )}
            </div>
            
            <div style={{ 
              fontSize: '12px', 
              color: '#9ca3af', 
              marginTop: '8px',
              ...(message.role === 'user' ? { textAlign: 'right' } : {})
            }}>
              {message.timestamp?.toLocaleTimeString()}
            </div>
          </div>
        </div>
      ))}
      
      {loading && (
        <div style={{ display: 'flex', gap: '16px', alignItems: 'flex-start' }}>
          <div style={{
            width: '40px',
            height: '40px',
            borderRadius: '50%',
            background: '#f3f4f6',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center'
          }}>
            <Bot size={20} color="#6b7280" />
          </div>
          
          <div style={{
            padding: '16px 20px',
            borderRadius: '12px',
            background: 'white',
            boxShadow: '0 1px 3px rgba(0,0,0,0.1)'
          }}>
            <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
              <div className="typing-dot" style={{ width: '8px', height: '8px', background: '#9ca3af', borderRadius: '50%', animation: 'typing 1.4s infinite' }}></div>
              <div className="typing-dot" style={{ width: '8px', height: '8px', background: '#9ca3af', borderRadius: '50%', animation: 'typing 1.4s infinite 0.2s' }}></div>
              <div className="typing-dot" style={{ width: '8px', height: '8px', background: '#9ca3af', borderRadius: '50%', animation: 'typing 1.4s infinite 0.4s' }}></div>
            </div>
          </div>
        </div>
      )}
      
      <div ref={messagesEndRef} />
    </div>
  )
}