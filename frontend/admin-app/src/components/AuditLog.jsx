import React, { useState, useEffect } from 'react'
import api from '../services/api'
import { formatDate } from '../utils/helpers'
import { Activity, RefreshCw } from 'lucide-react'

export default function AuditLog() {
  const [logs, setLogs] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  useEffect(() => {
    fetchLogs()
  }, [])

  const fetchLogs = async () => {
    try {
      const response = await api.get('/admin/audit-logs')
      setLogs(response.data.logs)
    } catch (err) {
      setError('Failed to load audit logs')
    } finally {
      setLoading(false)
    }
  }

  if (loading) return <div className="loading">Loading audit logs...</div>
  if (error) return <div className="error">{error}</div>

  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
        <h2 style={{ fontSize: '20px', fontWeight: '600', color: '#1f2937', display: 'flex', alignItems: 'center', gap: '8px' }}>
          <Activity size={24} />
          Audit Logs ({logs.length})
        </h2>
        <button 
          className="btn btn-primary"
          onClick={fetchLogs}
          style={{ display: 'flex', alignItems: 'center', gap: '8px' }}
        >
          <RefreshCw size={16} />
          Refresh
        </button>
      </div>

      {logs.length === 0 ? (
        <div className="card" style={{ textAlign: 'center', padding: '60px' }}>
          <p style={{ fontSize: '16px', color: '#6b7280' }}>No audit logs found</p>
        </div>
      ) : (
        <div className="card" style={{ padding: 0, overflow: 'hidden' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead>
              <tr style={{ background: '#f9fafb', borderBottom: '1px solid #e5e7eb' }}>
                <th style={{ padding: '12px 16px', textAlign: 'left', fontSize: '14px', fontWeight: '600', color: '#374151' }}>Timestamp</th>
                <th style={{ padding: '12px 16px', textAlign: 'left', fontSize: '14px', fontWeight: '600', color: '#374151' }}>User</th>
                <th style={{ padding: '12px 16px', textAlign: 'left', fontSize: '14px', fontWeight: '600', color: '#374151' }}>Action</th>
                <th style={{ padding: '12px 16px', textAlign: 'left', fontSize: '14px', fontWeight: '600', color: '#374151' }}>Resource Type</th>
                <th style={{ padding: '12px 16px', textAlign: 'left', fontSize: '14px', fontWeight: '600', color: '#374151' }}>Resource ID</th>
              </tr>
            </thead>
            <tbody>
              {logs.map((log, index) => (
                <tr key={index} style={{ borderBottom: '1px solid #e5e7eb' }}>
                  <td style={{ padding: '12px 16px', fontSize: '14px', color: '#6b7280' }}>
                    {formatDate(log.timestamp)}
                  </td>
                  <td style={{ padding: '12px 16px', fontSize: '14px', color: '#1f2937' }}>
                    {log.user_id}
                  </td>
                  <td style={{ padding: '12px 16px', fontSize: '14px' }}>
                    <span style={{ 
                      padding: '4px 8px', 
                      borderRadius: '4px', 
                      background: '#eff6ff',
                      color: '#2563eb',
                      fontSize: '12px',
                      fontWeight: '500'
                    }}>
                      {log.action}
                    </span>
                  </td>
                  <td style={{ padding: '12px 16px', fontSize: '14px', color: '#6b7280' }}>
                    {log.resource_type}
                  </td>
                  <td style={{ padding: '12px 16px', fontSize: '14px', color: '#6b7280', fontFamily: 'monospace' }}>
                    {log.resource_id}
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