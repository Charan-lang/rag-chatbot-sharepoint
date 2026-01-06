import React, { useState } from 'react'
import { Link, useLocation, useNavigate, Outlet } from 'react-router-dom'
import { useAuth } from '../contexts/AuthContext'
import { 
  LayoutDashboard, FileText, Upload, Shield, 
  ClipboardList, LogOut, Menu, X, RefreshCw 
} from 'lucide-react'

export default function Layout() {
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const { user, logout } = useAuth()
  const location = useLocation()
  const navigate = useNavigate()

  const handleLogout = () => {
    logout()
    navigate('/login')
  }

  const navItems = [
    { path: '/', icon: LayoutDashboard, label: 'Dashboard' },
    { path: '/documents', icon: FileText, label: 'Documents' },
    { path: '/ingestion', icon: Upload, label: 'Ingestion' },
    { path: '/permission-sync', icon: RefreshCw, label: 'Permission Sync' },
    { path: '/audit', icon: ClipboardList, label: 'Audit Logs' },
  ]

  return (
    <div style={{ display: 'flex', height: '100vh', background: '#f9fafb' }}>
      {/* Sidebar */}
      <aside style={{ 
        width: sidebarOpen ? '250px' : '80px', 
        background: 'white', 
        borderRight: '1px solid #e5e7eb',
        transition: 'width 0.3s',
        overflow: 'hidden'
      }}>
        <div style={{ padding: '20px', borderBottom: '1px solid #e5e7eb', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          {sidebarOpen && <h2 style={{ fontSize: '18px', fontWeight: 'bold', color: '#1f2937' }}>RAG Admin</h2>}
          <button 
            onClick={() => setSidebarOpen(!sidebarOpen)}
            style={{ background: 'none', border: 'none', cursor: 'pointer', padding: '8px' }}
          >
            {sidebarOpen ? <X size={20} /> : <Menu size={20} />}
          </button>
        </div>

        <nav style={{ padding: '16px' }}>
          {navItems.map((item) => {
            const Icon = item.icon
            const isActive = location.pathname === item.path
            return (
              <Link
                key={item.path}
                to={item.path}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  padding: '12px',
                  marginBottom: '8px',
                  borderRadius: '6px',
                  textDecoration: 'none',
                  color: isActive ? '#2563eb' : '#6b7280',
                  background: isActive ? '#eff6ff' : 'transparent',
                  transition: 'all 0.2s'
                }}
              >
                <Icon size={20} />
                {sidebarOpen && <span style={{ marginLeft: '12px', fontSize: '14px', fontWeight: '500' }}>{item.label}</span>}
              </Link>
            )
          })}
        </nav>

        <div style={{ position: 'absolute', bottom: '20px', left: '16px', right: '16px' }}>
          <button
            onClick={handleLogout}
            style={{
              display: 'flex',
              alignItems: 'center',
              width: '100%',
              padding: '12px',
              background: '#fee2e2',
              border: 'none',
              borderRadius: '6px',
              color: '#dc2626',
              cursor: 'pointer',
              fontSize: '14px',
              fontWeight: '500'
            }}
          >
            <LogOut size={20} />
            {sidebarOpen && <span style={{ marginLeft: '12px' }}>Logout</span>}
          </button>
        </div>
      </aside>

      {/* Main Content */}
      <main style={{ flex: 1, overflow: 'auto' }}>
        {/* Header */}
        <header style={{ background: 'white', borderBottom: '1px solid #e5e7eb', padding: '16px 24px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <h1 style={{ fontSize: '20px', fontWeight: '600', color: '#1f2937' }}>
              {navItems.find(item => item.path === location.pathname)?.label || 'Dashboard'}
            </h1>
            <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
              <Shield size={20} color="#10b981" />
              <span style={{ fontSize: '14px', color: '#6b7280' }}>
                {user?.name || 'Admin User'}
              </span>
            </div>
          </div>
        </header>

        {/* Page Content */}
        <div style={{ padding: '24px' }}>
          <Outlet />
        </div>
      </main>
    </div>
  )
}