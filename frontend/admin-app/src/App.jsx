import React from 'react'
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom'
import { AuthProvider } from './contexts/AuthContext'

import Login from './components/Login'
import Layout from './components/Layout'
import Dashboard from './components/Dashboard'
import DocumentList from './components/DocumentList'
import ChunkViewer from './components/ChunkViewer'
import PermissionManager from './components/PermissionManager'
import IngestionPanel from './components/IngestionPanel'
import AuditLog from './components/AuditLog'
import PermissionSync from './components/PermissionSync'

function ProtectedRoute({ children }) {
  const token = localStorage.getItem('admin_token')
  return token ? children : <Navigate to="/login" />
}

export default function App() {
  return (
    <AuthProvider>
      <Router>
        <Routes>

          <Route path="/login" element={<Login />} />

          <Route
            path="/"
            element={
              <ProtectedRoute>
                <Layout />
              </ProtectedRoute>
            }
          >
            <Route index element={<Dashboard />} />
            <Route path="documents" element={<DocumentList />} />
            <Route path="documents/:id/chunks" element={<ChunkViewer />} />
            <Route path="documents/:id/permissions" element={<PermissionManager />} />
            <Route path="ingestion" element={<IngestionPanel />} />
            <Route path="permission-sync" element={<PermissionSync />} />
            <Route path="audit" element={<AuditLog />} />
          </Route>

        </Routes>
      </Router>
    </AuthProvider>
  )
}
