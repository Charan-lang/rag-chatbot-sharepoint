import React from 'react'
import { AuthProvider } from './contexts/AuthContext'
import Layout from './components/Layout'
import ChatInterface from './components/ChatInterface'

function App() {
  return (
    <AuthProvider>
      <Layout>
        <ChatInterface />
      </Layout>
    </AuthProvider>
  )
}

export default App