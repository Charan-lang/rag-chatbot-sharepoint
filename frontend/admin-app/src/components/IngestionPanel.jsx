import React, { useState, useEffect } from 'react'
import api from '../services/api'
import { Upload, Play, RefreshCw, CheckCircle, XCircle, Clock } from 'lucide-react'
import { getStatusColor } from '../utils/helpers'

export default function IngestionPanel() {
  const [jobs, setJobs] = useState([])
  const [loading, setLoading] = useState(false)
  const [libraryName, setLibraryName] = useState('Documents')
  const [error, setError] = useState('')
  const [success, setSuccess] = useState('')

  useEffect(() => {
    fetchJobs()
  }, [])

  const fetchJobs = async () => {
    try {
      const response = await api.get('/ingestion/jobs')
      setJobs(response.data.jobs)
    } catch (err) {
      console.error('Failed to load jobs', err)
    }
  }

  const startIngestion = async () => {
    setLoading(true)
    setError('')
    setSuccess('')

    try {
      const response = await api.post('/ingestion/start', {
        source: 'sharepoint',
        library_name: libraryName
      })
      setSuccess(`Ingestion started! Job ID: ${response.data.job_id}`)
      setTimeout(() => fetchJobs(), 2000)
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to start ingestion')
    } finally {
      setLoading(false)
    }
  }

  const getStatusIcon = (status) => {
    switch (status) {
      case 'completed':
        return <CheckCircle size={16} color="#10b981" />
      case 'failed':
        return <XCircle size={16} color="#ef4444" />
      case 'processing':
        return <Clock size={16} color="#3b82f6" />
      default:
        return <Clock size={16} color="#6b7280" />
    }
  }

  return (
    <div>
      <div className="card" style={{ marginBottom: '24px' }}>
        <h2 style={{ fontSize: '20px', fontWeight: '600', marginBottom: '16px', color: '#1f2937', display: 'flex', alignItems: 'center', gap: '8px' }}>
          <Upload size={24} />
          Start New Ingestion
        </h2>

        {error && <div className="error">{error}</div>}
        {success && <div className="success">{success}</div>}

        <div style={{ display: 'flex', gap: '12px', marginBottom: '16px' }}>
          <div style={{ flex: 1 }}>
            <label style={{ display: 'block', marginBottom: '8px', fontSize: '14px', fontWeight: '500', color: '#374151' }}>
              SharePoint Library Name
            </label>
            <input
              type="text"
              value={libraryName}
              onChange={(e) => setLibraryName(e.target.value)}
              placeholder="Documents"
              style={{ width: '100%', padding: '10px', border: '1px solid #d1d5db', borderRadius: '6px', fontSize: '14px' }}
            />
          </div>
        </div>

        <button
          className="btn btn-primary"
          onClick={startIngestion}
          disabled={loading}
          style={{ display: 'flex', alignItems: 'center', gap: '8px' }}
        >
          <Play size={16} />
          {loading ? 'Starting...' : 'Start Ingestion'}
        </button>
      </div>

      <div className="card">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
          <h2 style={{ fontSize: '18px', fontWeight: '600', color: '#1f2937' }}>
            Ingestion Jobs
          </h2>
          <button 
            className="btn btn-secondary"
            onClick={fetchJobs}
            style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '14px' }}
          >
            <RefreshCw size={14} />
            Refresh
          </button>
        </div>

        {jobs.length === 0 ? (
          <p style={{ textAlign: 'center', color: '#6b7280', padding: '40px' }}>No jobs found</p>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
            {jobs.map((job) => (
              <div 
                key={job.job_id}
                style={{ 
                  padding: '16px', 
                  border: '1px solid #e5e7eb', 
                  borderRadius: '8px',
                  background: '#fafafa'
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '12px' }}>
                  <div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '4px' }}>
                      {getStatusIcon(job.status)}
                      <span style={{ fontSize: '14px', fontWeight: '600', color: '#1f2937' }}>
                        {job.source.toUpperCase()}
                      </span>
                      <span style={{ 
                        padding: '2px 8px', 
                        borderRadius: '12px', 
                        background: getStatusColor(job.status) + '20',
                        color: getStatusColor(job.status),
                        fontSize: '12px',
                        fontWeight: '500'
                      }}>
                        {job.status}
                      </span>
                    </div>
                    <p style={{ fontSize: '12px', color: '#6b7280' }}>Job ID: {job.job_id}</p>
                  </div>
                  <div style={{ textAlign: 'right' }}>
                    <p style={{ fontSize: '12px', color: '#6b7280' }}>
                      {job.started_at && new Date(job.started_at).toLocaleString()}
                    </p>
                  </div>
                </div>

                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '12px' }}>
                  <div>
                    <p style={{ fontSize: '12px', color: '#6b7280', marginBottom: '4px' }}>Total</p>
                    <p style={{ fontSize: '18px', fontWeight: '600', color: '#1f2937' }}>{job.total_documents}</p>
                  </div>
                  <div>
                    <p style={{ fontSize: '12px', color: '#6b7280', marginBottom: '4px' }}>Processed</p>
                    <p style={{ fontSize: '18px', fontWeight: '600', color: '#10b981' }}>{job.processed_documents}</p>
                  </div>
                  <div>
                    <p style={{ fontSize: '12px', color: '#6b7280', marginBottom: '4px' }}>Failed</p>
                    <p style={{ fontSize: '18px', fontWeight: '600', color: '#ef4444' }}>{job.failed_documents}</p>
                  </div>
                </div>

                {job.error_message && (
                  <div style={{ marginTop: '12px', padding: '8px', background: '#fee2e2', borderRadius: '4px' }}>
                    <p style={{ fontSize: '12px', color: '#dc2626' }}>{job.error_message}</p>
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}