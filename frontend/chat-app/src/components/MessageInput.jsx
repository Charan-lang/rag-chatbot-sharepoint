import React, { useState } from 'react'
import { Send, Trash2 } from 'lucide-react'

export default function MessageInput({ onSend, loading, onClear }) {
  const [input, setInput] = useState('')

  const handleSubmit = (e) => {
    e.preventDefault()
    if (input.trim() && !loading) {
      onSend(input.trim())
      setInput('')
    }
  }

  return (
    <div style={{ 
      padding: '20px 24px', 
      background: 'white', 
      borderTop: '1px solid #e5e7eb' 
    }}>
      <form onSubmit={handleSubmit} style={{ display: 'flex', gap: '12px', alignItems: 'center' }}>
        <button
          type="button"
          onClick={onClear}
          style={{
            padding: '12px',
            background: '#f3f4f6',
            border: 'none',
            borderRadius: '8px',
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center'
          }}
          title="Clear conversation"
        >
          <Trash2 size={20} color="#6b7280" />
        </button>
        
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask a question about your documents..."
          disabled={loading}
          style={{
            flex: 1,
            padding: '14px 18px',
            border: '1px solid #d1d5db',
            borderRadius: '10px',
            fontSize: '15px',
            outline: 'none',
            transition: 'border-color 0.2s'
          }}
          onFocus={(e) => e.target.style.borderColor = '#2563eb'}
          onBlur={(e) => e.target.style.borderColor = '#d1d5db'}
        />
        
        <button
          type="submit"
          disabled={!input.trim() || loading}
          style={{
            padding: '14px 24px',
            background: input.trim() && !loading ? '#2563eb' : '#d1d5db',
            color: 'white',
            border: 'none',
            borderRadius: '10px',
            fontSize: '15px',
            fontWeight: '600',
            cursor: input.trim() && !loading ? 'pointer' : 'not-allowed',
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            transition: 'all 0.2s'
          }}
        >
          <Send size={18} />
          Send
        </button>
      </form>
    </div>
  )
}