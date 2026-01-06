import React, { useState, useEffect } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import api from '../services/api'
import { ArrowLeft, Users, UserPlus, X, Mail, Building } from 'lucide-react'

export default function PermissionManager() {
  const { id } = useParams()
  const navigate = useNavigate()
  const [permissions, setPermissions] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [newUser, setNewUser] = useState('')
  const [newGroup, setNewGroup] = useState('')

  useEffect(() => {
    fetchPermissions()
  }, [id])

  const fetchPermissions = async () => {
    try {
      const response = await api.get(`/admin/documents/${id}/permissions`)
      setPermissions(response.data)
    } catch (err) {
      setError('Failed to load permissions')
    } finally {
      setLoading(false)
    }
  }

  const addUser = () => {
    if (!newUser.trim()) return
    // In real implementation, call API to add user
    alert(`Would add user: ${newUser}`)
    setNewUser('')
  }

  const addGroup = () => {
    if (!newGroup.trim()) return
    // In real implementation, call API to add group
    alert(`Would add group: ${newGroup}`)
    setNewGroup('')
  }

  // Helper to get display text for user (email or ID)
  const getUserDisplay = (user) => {
    if (typeof user === 'object') {
      return user.email || user.displayName || user.id
    }
    return user
  }

  // Helper to get display text for group (name or ID)
  const getGroupDisplay = (group) => {
    if (typeof group === 'object') {
      return group.displayName || group.mail || group.id
    }
    return group
  }

  if (loading) return <div className="loading">Loading permissions...</div>
  if (error) return <div className="error">{error}</div>

  return (
    <div>
      <div style={{ marginBottom: '20px' }}>
        <button 
          className="btn btn-secondary"
          onClick={() => navigate('/documents')}
          style={{ display: 'flex', alignItems: 'center', gap: '8px' }}
        >
          <ArrowLeft size={16} />
          Back to Documents
        </button>
      </div>

      <div className="card" style={{ marginBottom: '20px' }}>
        <h2 style={{ fontSize: '20px', fontWeight: '600', marginBottom: '8px', color: '#1f2937' }}>
          Document Permissions
        </h2>
        <p style={{ fontSize: '14px', color: '#6b7280' }}>
          Document ID: {permissions?.document_id}
        </p>
        <p style={{ fontSize: '14px', color: '#6b7280', marginTop: '4px', display: 'flex', alignItems: 'center', gap: '4px' }}>
          <Mail size={14} />
          Owner: {permissions?.owner_email || permissions?.owner_id}
        </p>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>
        {/* Users */}
        <div className="card">
          <h3 style={{ fontSize: '16px', fontWeight: '600', marginBottom: '16px', color: '#1f2937', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Users size={20} />
            Allowed Users ({permissions?.allowed_users?.length || 0})
          </h3>

          <div style={{ marginBottom: '16px', display: 'flex', gap: '8px' }}>
            <input
              type="text"
              value={newUser}
              onChange={(e) => setNewUser(e.target.value)}
              placeholder="User email"
              style={{ flex: 1, padding: '8px', border: '1px solid #d1d5db', borderRadius: '4px', fontSize: '14px' }}
            />
            <button 
              className="btn btn-primary"
              onClick={addUser}
              style={{ display: 'flex', alignItems: 'center', gap: '4px' }}
            >
              <UserPlus size={16} />
              Add
            </button>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
            {permissions?.allowed_users?.map((user, idx) => (
              <div 
                key={typeof user === 'object' ? user.id : user}
                style={{ 
                  display: 'flex', 
                  justifyContent: 'space-between', 
                  alignItems: 'center',
                  padding: '8px 12px',
                  background: '#f9fafb',
                  borderRadius: '4px'
                }}
              >
                <div style={{ display: 'flex', flexDirection: 'column' }}>
                  <span style={{ fontSize: '14px', color: '#374151', display: 'flex', alignItems: 'center', gap: '4px' }}>
                    <Mail size={14} style={{ color: '#6b7280' }} />
                    {getUserDisplay(user)}
                  </span>
                  {typeof user === 'object' && user.displayName && user.email && (
                    <span style={{ fontSize: '12px', color: '#9ca3af' }}>
                      {user.displayName}
                    </span>
                  )}
                </div>
                <button
                  style={{ background: 'none', border: 'none', cursor: 'pointer', color: '#ef4444' }}
                  onClick={() => alert(`Would remove user: ${getUserDisplay(user)}`)}
                >
                  <X size={16} />
                </button>
              </div>
            ))}
          </div>
        </div>

        {/* Groups */}
        <div className="card">
          <h3 style={{ fontSize: '16px', fontWeight: '600', marginBottom: '16px', color: '#1f2937', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Building size={20} />
            Allowed Groups ({permissions?.allowed_groups?.length || 0})
          </h3>

          <div style={{ marginBottom: '16px', display: 'flex', gap: '8px' }}>
            <input
              type="text"
              value={newGroup}
              onChange={(e) => setNewGroup(e.target.value)}
              placeholder="Group name"
              style={{ flex: 1, padding: '8px', border: '1px solid #d1d5db', borderRadius: '4px', fontSize: '14px' }}
            />
            <button 
              className="btn btn-primary"
              onClick={addGroup}
              style={{ display: 'flex', alignItems: 'center', gap: '4px' }}
            >
              <UserPlus size={16} />
              Add
            </button>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
            {permissions?.allowed_groups?.map((group, idx) => (
              <div 
                key={typeof group === 'object' ? group.id : group}
                style={{ 
                  display: 'flex', 
                  justifyContent: 'space-between', 
                  alignItems: 'center',
                  padding: '8px 12px',
                  background: '#f9fafb',
                  borderRadius: '4px'
                }}
              >
                <div style={{ display: 'flex', flexDirection: 'column' }}>
                  <span style={{ fontSize: '14px', color: '#374151', display: 'flex', alignItems: 'center', gap: '4px' }}>
                    <Building size={14} style={{ color: '#6b7280' }} />
                    {getGroupDisplay(group)}
                  </span>
                  {typeof group === 'object' && group.mail && group.displayName !== group.mail && (
                    <span style={{ fontSize: '12px', color: '#9ca3af' }}>
                      {group.mail}
                    </span>
                  )}
                </div>
                <button
                  style={{ background: 'none', border: 'none', cursor: 'pointer', color: '#ef4444' }}
                  onClick={() => alert(`Would remove group: ${getGroupDisplay(group)}`)}
                >
                  <X size={16} />
                </button>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}