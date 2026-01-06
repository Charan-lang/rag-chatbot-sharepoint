import React, { useState, useEffect } from 'react'
import api from '../services/api'
import { FileText, Database, Users, Activity } from 'lucide-react'

export default function Dashboard() {
  const [stats, setStats] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  useEffect(() => {
    fetchStats()
  }, [])

  const fetchStats = async () => {
    try {
      const response = await api.get('/admin/stats')
      setStats(response.data)
    } catch (err) {
      setError('Failed to load statistics')
    } finally {
      setLoading(false)
    }
  }

  if (loading) return <div className="loading">Loading statistics...</div>
  if (error) return <div className="error">{error}</div>

  const statCards = [
    {
      title: 'Total Documents',
      value: stats?.total_documents || 0,
      icon: FileText,
      color: '#3b82f6',
      bgColor: '#eff6ff'
    },
    {
      title: 'Total Chunks',
      value: stats?.total_chunks || 0,
      icon: Database,
      color: '#8b5cf6',
      bgColor: '#f5f3ff'
    },
    {
      title: 'Avg Chunks/Doc',
      value: stats?.average_chunks_per_document || 0,
      icon: Activity,
      color: '#10b981',
      bgColor: '#dcfce7'
    },
    {
      title: 'Audit Entries',
      value: stats?.total_audit_entries || 0,
      icon: Users,
      color: '#f59e0b',
      bgColor: '#fef3c7'
    }
  ]

  return (
    <div>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: '20px', marginBottom: '30px' }}>
        {statCards.map((card, index) => {
          const Icon = card.icon
          return (
            <div
              key={index}
              className="card"
              style={{ 
                display: 'flex', 
                alignItems: 'center',
                transition: 'transform 0.2s',
                cursor: 'pointer'
              }}
              onMouseEnter={(e) => e.currentTarget.style.transform = 'translateY(-4px)'}
              onMouseLeave={(e) => e.currentTarget.style.transform = 'translateY(0)'}
            >
              <div style={{ 
                width: '60px', 
                height: '60px', 
                borderRadius: '12px', 
                background: card.bgColor,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                marginRight: '16px'
              }}>
                <Icon size={28} color={card.color} />
              </div>
              <div>
                <p style={{ fontSize: '14px', color: '#6b7280', marginBottom: '4px' }}>{card.title}</p>
                <p style={{ fontSize: '28px', fontWeight: 'bold', color: '#1f2937' }}>{card.value}</p>
              </div>
            </div>
          )
        })}
      </div>

      <div className="card">
        <h2 style={{ fontSize: '18px', fontWeight: '600', marginBottom: '16px', color: '#1f2937' }}>
          System Information
        </h2>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
          <div>
            <p style={{ fontSize: '14px', color: '#6b7280', marginBottom: '4px' }}>Index Name</p>
            <p style={{ fontSize: '16px', fontWeight: '500', color: '#1f2937' }}>{stats?.index_name || 'N/A'}</p>
          </div>
          <div>
            <p style={{ fontSize: '14px', color: '#6b7280', marginBottom: '4px' }}>Status</p>
            <span style={{ 
              padding: '4px 12px', 
              borderRadius: '12px', 
              background: '#dcfce7', 
              color: '#16a34a',
              fontSize: '14px',
              fontWeight: '500'
            }}>
              Operational
            </span>
          </div>
        </div>
      </div>
    </div>
  )
}