import { useState } from 'react'
import { useAuth } from '../contexts/AuthContext'
import api from '../services/api'

export const useChat = () => {
  const { acquireToken } = useAuth()
  const [messages, setMessages] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const sendMessage = async (query) => {
    const userMessage = {
      id: Date.now(),
      role: 'user',
      content: query,
      timestamp: new Date()
    }

    setMessages(prev => [...prev, userMessage])
    setLoading(true)
    setError(null)

    try {
      // Ensure we have a fresh token before making the API call
      console.log('Acquiring fresh token before API call...')
      await acquireToken()
      
      console.log('Sending chat query to API...')
      const response = await api.post('/chat/query', {
        query,
        conversation_history: messages.map(m => ({
          role: m.role,
          content: m.content
        })),
        use_vector_search: true
      })

      const assistantMessage = {
        id: Date.now() + 1,
        role: 'assistant',
        content: response.data.answer,
        sources: response.data.sources,
        timestamp: new Date()
      }

      setMessages(prev => [...prev, assistantMessage])
    } catch (err) {
      console.error('Chat query error:', err)
      const errorDetail = err.response?.data?.detail || 'Failed to get response'
      setError(errorDetail)
      
      const errorMessage = {
        id: Date.now() + 1,
        role: 'assistant',
        content: `Sorry, I encountered an error: ${errorDetail}`,
        error: true,
        timestamp: new Date()
      }
      
      setMessages(prev => [...prev, errorMessage])
    } finally {
      setLoading(false)
    }
  }

  const clearMessages = () => {
    setMessages([])
    setError(null)
  }

  return {
    messages,
    loading,
    error,
    sendMessage,
    clearMessages
  }
}