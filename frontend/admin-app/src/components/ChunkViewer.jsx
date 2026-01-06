import React, { useState, useEffect } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import api from '../services/api'
import { ArrowLeft, Copy, Check, Mail, Building, ChevronDown, ChevronRight } from 'lucide-react'

export default function ChunkViewer() {
  const { id } = useParams()
  const navigate = useNavigate()
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [copiedId, setCopiedId] = useState(null)
  const [expandedChunks, setExpandedChunks] = useState({})

  useEffect(() => {
    fetchChunks()
  }, [id])

  const fetchChunks = async () => {
    try {
      const response = await api.get(`/admin/documents/${id}/chunks`)
      setData(response.data)
    } catch (err) {
      setError('Failed to load chunks')
    } finally {
      setLoading(false)
    }
  }

  const copyToClipboard = (text, chunkId) => {
    navigator.clipboard.writeText(text)
    setCopiedId(chunkId)
    setTimeout(() => setCopiedId(null), 2000)
  }

  const toggleExpanded = (chunkId) => {
    setExpandedChunks(prev => ({
      ...prev,
      [chunkId]: !prev[chunkId]
    }))
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

  if (loading) return <div className="loading">Loading chunks...</div>
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
          {data?.document_title}
        </h2>
        <p style={{ fontSize: '14px', color: '#6b7280' }}>
          Total Chunks: {data?.total_chunks}
        </p>
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
        {data?.chunks.map((chunk) => (
          <div key={chunk.id} className="card">
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'start', marginBottom: '12px' }}>
              <div>
                <span style={{ 
                  padding: '4px 8px', 
                  borderRadius: '4px', 
                  background: '#eff6ff',
                  color: '#2563eb',
                  fontSize: '12px',
                  fontWeight: '600'
                }}>
                  Chunk #{chunk.chunk_index}
                </span>
              </div>
              <button
                className="btn btn-secondary"
                onClick={() => copyToClipboard(chunk.content, chunk.id)}
                style={{ padding: '6px 12px', fontSize: '12px', display: 'flex', alignItems: 'center', gap: '4px' }}
              >
                {copiedId === chunk.id ? <Check size={14} /> : <Copy size={14} />}
                {copiedId === chunk.id ? 'Copied' : 'Copy'}
              </button>
            </div>

            <div style={{ 
              background: '#f9fafb', 
              padding: '12px', 
              borderRadius: '6px',
              marginBottom: '12px',
              maxHeight: '200px',
              overflow: 'auto'
            }}>
              <p style={{ fontSize: '14px', color: '#374151', lineHeight: '1.6', whiteSpace: 'pre-wrap' }}>
                {chunk.content}
              </p>
            </div>

            {/* Permissions Section */}
            <div 
              style={{ 
                cursor: 'pointer',
                padding: '8px',
                background: '#f3f4f6',
                borderRadius: '4px'
              }}
              onClick={() => toggleExpanded(chunk.id)}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '12px', color: '#6b7280' }}>
                {expandedChunks[chunk.id] ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
                <span style={{ fontWeight: '600' }}>Permissions:</span>
                <span>{chunk.allowed_users?.length || 0} Users, {chunk.allowed_groups?.length || 0} Groups</span>
              </div>
            </div>
            
            {expandedChunks[chunk.id] && (
              <div style={{ marginTop: '12px', display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
                {/* Users */}
                <div style={{ background: '#f9fafb', padding: '8px', borderRadius: '4px' }}>
                  <div style={{ fontSize: '12px', fontWeight: '600', color: '#374151', marginBottom: '8px', display: 'flex', alignItems: 'center', gap: '4px' }}>
                    <Mail size={12} />
                    Allowed Users ({chunk.allowed_users?.length || 0})
                  </div>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
                    {chunk.allowed_users?.map((user, idx) => (
                      <div key={typeof user === 'object' ? user.id : user} style={{ fontSize: '12px', color: '#6b7280' }}>
                        {getUserDisplay(user)}
                      </div>
                    ))}
                    {(!chunk.allowed_users || chunk.allowed_users.length === 0) && (
                      <div style={{ fontSize: '12px', color: '#9ca3af', fontStyle: 'italic' }}>No users</div>
                    )}
                  </div>
                </div>
                
                {/* Groups */}
                <div style={{ background: '#f9fafb', padding: '8px', borderRadius: '4px' }}>
                  <div style={{ fontSize: '12px', fontWeight: '600', color: '#374151', marginBottom: '8px', display: 'flex', alignItems: 'center', gap: '4px' }}>
                    <Building size={12} />
                    Allowed Groups ({chunk.allowed_groups?.length || 0})
                  </div>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
                    {chunk.allowed_groups?.map((group, idx) => (
                      <div key={typeof group === 'object' ? group.id : group} style={{ fontSize: '12px', color: '#6b7280' }}>
                        {getGroupDisplay(group)}
                      </div>
                    ))}
                    {(!chunk.allowed_groups || chunk.allowed_groups.length === 0) && (
                      <div style={{ fontSize: '12px', color: '#9ca3af', fontStyle: 'italic' }}>No groups</div>
                    )}
                  </div>
                </div>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  )
}