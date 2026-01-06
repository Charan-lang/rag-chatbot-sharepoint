import React from 'react'
import { useAuth } from '../contexts/AuthContext'
import UserProfile from './UserProfile'
import { MessageSquare, LogOut } from 'lucide-react'

export default function Layout({ children }) {
  const { user, login, logout, loading } = useAuth()

  if (loading) {
    return (
      <div style={{ 
        display: 'flex', 
        justifyContent: 'center', 
        alignItems: 'center', 
        height: '100vh',
        background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)'
      }}>
        <div style={{ textAlign: 'center', color: 'white' }}>
          <div style={{ fontSize: '24px', marginBottom: '16px' }}>Loading...</div>
        </div>
      </div>
    )
  }

  if (!user) {
    return (
      <div style={{ 
        display: 'flex', 
        justifyContent: 'center', 
        alignItems: 'center', 
        height: '100vh',
        background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)'
      }}>
        <div style={{ 
          background: 'white', 
          padding: '60px 40px', 
          borderRadius: '16px', 
          boxShadow: '0 20px 60px rgba(0,0,0,0.3)',
          textAlign: 'center',
          maxWidth: '400px'
        }}>
          <div style={{ 
            width: '80px', 
            height: '80px', 
            background: '#667eea', 
            borderRadius: '50%',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            margin: '0 auto 24px'
          }}>
            <MessageSquare size={40} color="white" />
          </div>
          
          <h1 style={{ fontSize: '28px', fontWeight: 'bold', marginBottom: '12px', color: '#1f2937' }}>
            RAG Chat
          </h1>
          <p style={{ fontSize: '16px', color: '#6b7280', marginBottom: '32px' }}>
            AI-powered document search with permissions
          </p>
          
          <button
            onClick={login}
            style={{
              width: '100%',
              padding: '14px 24px',
              background: '#667eea',
              color: 'white',
              border: 'none',
              borderRadius: '8px',
              fontSize: '16px',
              fontWeight: '600',
              cursor: 'pointer',
              transition: 'all 0.3s'
            }}
            onMouseEnter={(e) => e.target.style.background = '#5568d3'}
            onMouseLeave={(e) => e.target.style.background = '#667eea'}
          >
            Sign in with Microsoft
          </button>
          
          <p style={{ fontSize: '13px', color: '#9ca3af', marginTop: '24px' }}>
            Secure authentication via Microsoft Entra ID
          </p>
        </div>
      </div>
    )
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100vh' }}>
      {/* Header */}
      <header style={{ 
        background: 'white', 
        borderBottom: '1px solid #e5e7eb', 
        padding: '12px 24px',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <div style={{ 
            width: '40px', 
            height: '40px', 
            background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)', 
            borderRadius: '8px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center'
          }}>
            <MessageSquare size={24} color="white" />
          </div>
          <div>
            <h1 style={{ fontSize: '18px', fontWeight: '600', color: '#1f2937' }}>RAG Chat</h1>
            <p style={{ fontSize: '12px', color: '#6b7280' }}>Permissions-Aware AI Assistant</p>
          </div>
        </div>
        
        <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
          <UserProfile user={user} />
          <button
            onClick={logout}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
              padding: '8px 16px',
              background: '#fee2e2',
              border: 'none',
              borderRadius: '6px',
              color: '#dc2626',
              fontSize: '14px',
              fontWeight: '500',
              cursor: 'pointer'
            }}
          >
            <LogOut size={16} />
            Logout
          </button>
        </div>
      </header>

      {/* Main Content */}
      <main style={{ flex: 1, overflow: 'hidden' }}>
        {children}
      </main>
    </div>
  )
}