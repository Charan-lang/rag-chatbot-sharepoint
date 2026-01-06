import React, { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import api from '../services/api'
import { formatDate } from '../utils/helpers'
import { Eye, Shield, Trash2, RefreshCw } from 'lucide-react'

export default function DocumentList() {
  const [documents, setDocuments] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const navigate = useNavigate()

  useEffect(() => {
    fetchDocuments()
  }, [])

  const fetchDocuments = async () => {
    try {
      const response = await api.get('/admin/documents')
      setDocuments(response.data.documents)
    } catch (err) {
      setError('Failed to load documents')
    } finally {
      setLoading(false)
    }
  }

  const deleteDocument = async (docId) => {
    if (!confirm('Are you sure you want to delete this document?')) return

    try {
      await api.delete(`/admin/documents/${docId}`)
      setDocuments(documents.filter(doc => doc.id !== docId))
    } catch (err) {
      alert('Failed to delete document')
    }
  }

  if (loading) return <div className="loading">Loading documents...</div>
  if (error) return <div className="error">{error}</div>

  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
        <h2 style={{ fontSize: '20px', fontWeight: '600', color: '#1f2937' }}>
          Documents ({documents.length})
        </h2>
        <button 
          className="btn btn-primary"
          onClick={fetchDocuments}
          style={{ display: 'flex', alignItems: 'center', gap: '8px' }}
        >
          <RefreshCw size={16} />
          Refresh
        </button>
      </div>

      {documents.length === 0 ? (
        <div className="card" style={{ textAlign: 'center', padding: '60px' }}>
          <p style={{ fontSize: '16px', color: '#6b7280' }}>No documents found. Start by ingesting documents from SharePoint.</p>
        </div>
      ) : (
        <div className="card" style={{ padding: 0, overflow: 'hidden' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead>
              <tr style={{ background: '#f9fafb', borderBottom: '1px solid #e5e7eb' }}>
                <th style={{ padding: '12px 16px', textAlign: 'left', fontSize: '14px', fontWeight: '600', color: '#374151' }}>Title</th>
                <th style={{ padding: '12px 16px', textAlign: 'left', fontSize: '14px', fontWeight: '600', color: '#374151' }}>Source</th>
                <th style={{ padding: '12px 16px', textAlign: 'center', fontSize: '14px', fontWeight: '600', color: '#374151' }}>Chunks</th>
                <th style={{ padding: '12px 16px', textAlign: 'center', fontSize: '14px', fontWeight: '600', color: '#374151' }}>Users</th>
                <th style={{ padding: '12px 16px', textAlign: 'center', fontSize: '14px', fontWeight: '600', color: '#374151' }}>Groups</th>
                <th style={{ padding: '12px 16px', textAlign: 'center', fontSize: '14px', fontWeight: '600', color: '#374151' }}>Created</th>
                <th style={{ padding: '12px 16px', textAlign: 'center', fontSize: '14px', fontWeight: '600', color: '#374151' }}>Actions</th>
              </tr>
            </thead>
            <tbody>
              {documents.map((doc) => (
                <tr key={doc.id} style={{ borderBottom: '1px solid #e5e7eb' }}>
                  <td style={{ padding: '12px 16px', fontSize: '14px', color: '#1f2937' }}>
                    {doc.title}
                  </td>
                  <td style={{ padding: '12px 16px', fontSize: '14px', color: '#6b7280' }}>
                    <span style={{ 
                      padding: '4px 8px', 
                      borderRadius: '4px', 
                      background: '#f3f4f6',
                      fontSize: '12px'
                    }}>
                      {doc.source}
                    </span>
                  </td>
                  <td style={{ padding: '12px 16px', textAlign: 'center', fontSize: '14px', color: '#6b7280' }}>
                    {doc.chunk_count}
                  </td>
                  <td style={{ padding: '12px 16px', textAlign: 'center', fontSize: '14px', color: '#6b7280' }}>
                    {doc.allowed_users_count}
                  </td>
                  <td style={{ padding: '12px 16px', textAlign: 'center', fontSize: '14px', color: '#6b7280' }}>
                    {doc.allowed_groups_count}
                  </td>
                  <td style={{ padding: '12px 16px', textAlign: 'center', fontSize: '14px', color: '#6b7280' }}>
                    {formatDate(doc.created_at)}
                  </td>
                  <td style={{ padding: '12px 16px', textAlign: 'center' }}>
                    <div style={{ display: 'flex', gap: '8px', justifyContent: 'center' }}>
                      <button
                        className="btn btn-secondary"
                        onClick={() => navigate(`/documents/${doc.id}/chunks`)}
                        style={{ padding: '6px 12px', fontSize: '12px', display: 'flex', alignItems: 'center', gap: '4px' }}
                      >
                        <Eye size={14} />
                        Chunks
                      </button>
                      <button
                        className="btn btn-secondary"
                        onClick={() => navigate(`/documents/${doc.id}/permissions`)}
                        style={{ padding: '6px 12px', fontSize: '12px', display: 'flex', alignItems: 'center', gap: '4px' }}
                      >
                        <Shield size={14} />
                        Permissions
                      </button>
                      <button
                        className="btn btn-danger"
                        onClick={() => deleteDocument(doc.id)}
                        style={{ padding: '6px 12px', fontSize: '12px' }}
                      >
                        <Trash2 size={14} />
                      </button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}