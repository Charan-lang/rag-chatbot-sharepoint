import React, { useState, useEffect } from 'react'
import api from '../services/api'

export default function PermissionSync() {
  const [logicAppStatus, setLogicAppStatus] = useState(null)
  const [syncLoading, setSyncLoading] = useState(false)
  const [message, setMessage] = useState(null)

  useEffect(() => {
    fetchLogicAppStatus()
  }, [])

  const fetchLogicAppStatus = async () => {
    try {
      const response = await api.get('/logic-app/sync-status')
      setLogicAppStatus(response.data)
    } catch (err) {
      console.error('Error fetching Logic App status:', err)
    }
  }

  const handleTriggerLogicAppSync = async () => {
    setSyncLoading(true)
    setMessage(null)
    try {
      const response = await api.post('/logic-app/trigger-full-sync')
      setMessage({ 
        type: 'success', 
        text: `Full sync started! Processing ${response.data.documentsCount || 0} documents.` 
      })
      setTimeout(fetchLogicAppStatus, 3000)
    } catch (err) {
      setMessage({ 
        type: 'error', 
        text: err.response?.data?.detail || 'Failed to trigger sync' 
      })
    } finally {
      setSyncLoading(false)
    }
  }

  const handleResetSyncState = async () => {
    if (!confirm('Reset sync state? This will cause the next sync to start fresh.')) return
    try {
      await api.delete('/logic-app/reset-sync-state')
      setMessage({ type: 'success', text: 'Sync state reset successfully!' })
      fetchLogicAppStatus()
    } catch (err) {
      setMessage({ 
        type: 'error', 
        text: err.response?.data?.detail || 'Failed to reset' 
      })
    }
  }

  return (
    <div style={{ padding: '20px' }}>
      <h2 style={{ marginBottom: '20px' }}>🔄 Permission Sync Dashboard</h2>
      
      {message && (
        <div style={{
          padding: '12px 16px',
          marginBottom: '20px',
          borderRadius: '8px',
          background: message.type === 'error' ? '#fee2e2' : '#d1fae5',
          color: message.type === 'error' ? '#dc2626' : '#059669'
        }}>
          {message.text}
        </div>
      )}

      {/* Logic App Status */}
      <div style={{
        background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
        borderRadius: '12px',
        padding: '24px',
        marginBottom: '20px',
        color: 'white'
      }}>
        <h3 style={{ marginBottom: '16px', display: 'flex', alignItems: 'center', gap: '8px' }}>
          <span>⚡</span> Azure Logic App Sync Status
        </h3>
        <p style={{ opacity: 0.9, marginBottom: '16px', fontSize: '14px' }}>
          Logic App runs every 5 minutes to sync SharePoint permissions automatically using Microsoft Graph Delta Query.
        </p>
        
        {logicAppStatus ? (
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: '16px' }}>
            <div style={{ background: 'rgba(255,255,255,0.1)', borderRadius: '8px', padding: '16px' }}>
              <div style={{ fontSize: '24px', fontWeight: 'bold' }}>{logicAppStatus.totalSyncs || 0}</div>
              <div style={{ fontSize: '12px', opacity: 0.8 }}>Total Syncs</div>
            </div>
            <div style={{ background: 'rgba(255,255,255,0.1)', borderRadius: '8px', padding: '16px' }}>
              <div style={{ fontSize: '24px', fontWeight: 'bold' }}>{logicAppStatus.totalFilesProcessed || 0}</div>
              <div style={{ fontSize: '12px', opacity: 0.8 }}>Files Processed</div>
            </div>
            <div style={{ background: 'rgba(255,255,255,0.1)', borderRadius: '8px', padding: '16px' }}>
              <div style={{ fontSize: '24px', fontWeight: 'bold' }}>{logicAppStatus.totalErrors || 0}</div>
              <div style={{ fontSize: '12px', opacity: 0.8 }}>Errors</div>
            </div>
            <div style={{ background: 'rgba(255,255,255,0.1)', borderRadius: '8px', padding: '16px' }}>
              <div style={{ fontSize: '12px', fontWeight: 'bold' }}>
                {logicAppStatus.lastSyncTimestamp 
                  ? new Date(logicAppStatus.lastSyncTimestamp).toLocaleString()
                  : 'Never'}
              </div>
              <div style={{ fontSize: '12px', opacity: 0.8 }}>Last Sync</div>
            </div>
          </div>
        ) : (
          <p style={{ opacity: 0.7 }}>Loading status...</p>
        )}
        
        <div style={{ marginTop: '16px', display: 'flex', gap: '12px', flexWrap: 'wrap' }}>
          <button
            onClick={fetchLogicAppStatus}
            style={{
              padding: '8px 16px',
              background: 'rgba(255,255,255,0.2)',
              color: 'white',
              border: '1px solid rgba(255,255,255,0.3)',
              borderRadius: '6px',
              cursor: 'pointer'
            }}
          >
            🔄 Refresh Status
          </button>
          <button
            onClick={handleTriggerLogicAppSync}
            disabled={syncLoading}
            style={{
              padding: '8px 16px',
              background: 'white',
              color: '#764ba2',
              border: 'none',
              borderRadius: '6px',
              cursor: syncLoading ? 'not-allowed' : 'pointer',
              fontWeight: '500'
            }}
          >
            {syncLoading ? 'Starting...' : '🚀 Trigger Full Sync'}
          </button>
          <button
            onClick={handleResetSyncState}
            style={{
              padding: '8px 16px',
              background: 'rgba(255,255,255,0.1)',
              color: 'white',
              border: '1px solid rgba(255,255,255,0.3)',
              borderRadius: '6px',
              cursor: 'pointer'
            }}
          >
            ↺ Reset State
          </button>
        </div>
      </div>

      {/* How It Works */}
      <div style={{
        background: 'white',
        borderRadius: '12px',
        padding: '24px',
        marginBottom: '20px',
        boxShadow: '0 1px 3px rgba(0,0,0,0.1)'
      }}>
        <h3 style={{ marginBottom: '16px' }}>🔧 How It Works</h3>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '16px' }}>
          <div style={{ padding: '16px', background: '#f8fafc', borderRadius: '8px' }}>
            <div style={{ fontSize: '24px', marginBottom: '8px' }}>⏰</div>
            <h4 style={{ marginBottom: '8px', color: '#1f2937' }}>Scheduled Sync</h4>
            <p style={{ fontSize: '13px', color: '#666' }}>
              Logic App runs every 5 minutes to check for permission changes in SharePoint.
            </p>
          </div>
          <div style={{ padding: '16px', background: '#f8fafc', borderRadius: '8px' }}>
            <div style={{ fontSize: '24px', marginBottom: '8px' }}>📊</div>
            <h4 style={{ marginBottom: '8px', color: '#1f2937' }}>Delta Query</h4>
            <p style={{ fontSize: '13px', color: '#666' }}>
              Uses Microsoft Graph Delta Query to efficiently detect only changed files.
            </p>
          </div>
          <div style={{ padding: '16px', background: '#f8fafc', borderRadius: '8px' }}>
            <div style={{ fontSize: '24px', marginBottom: '8px' }}>🔐</div>
            <h4 style={{ marginBottom: '8px', color: '#1f2937' }}>Permission Sync</h4>
            <p style={{ fontSize: '13px', color: '#666' }}>
              Fetches file permissions and updates Azure AI Search index for security trimming.
            </p>
          </div>
        </div>
      </div>

      {/* Logic App Setup Guide */}
      <div style={{
        background: 'white',
        borderRadius: '12px',
        padding: '24px',
        marginBottom: '20px',
        boxShadow: '0 1px 3px rgba(0,0,0,0.1)'
      }}>
        <h3 style={{ marginBottom: '16px' }}>📘 Logic App Setup Guide</h3>
        <ol style={{ color: '#666', fontSize: '14px', lineHeight: '1.8', paddingLeft: '20px' }}>
          <li><strong>Deploy Logic App:</strong> Use the Bicep template in <code>azure-resources/deployment/logic-app-permission-sync.bicep</code></li>
          <li><strong>Configure Parameters:</strong>
            <ul style={{ marginTop: '8px', paddingLeft: '20px' }}>
              <li><code>backendApiUrl</code>: Your backend API URL (e.g., https://your-app.azurewebsites.net)</li>
              <li><code>backendApiKey</code>: Your SECRET_KEY from .env</li>
              <li><code>tenantId</code>: Your Azure AD tenant ID</li>
              <li><code>clientId</code>: App registration client ID</li>
              <li><code>clientSecret</code>: App registration client secret</li>
              <li><code>sharePointSiteId</code>: Your SharePoint site ID</li>
              <li><code>driveId</code>: Your SharePoint document library drive ID</li>
            </ul>
          </li>
          <li><strong>Enable the Logic App:</strong> It will start running every 5 minutes automatically</li>
          <li><strong>Monitor:</strong> Check the status above or use Azure Portal for detailed logs</li>
        </ol>
        
        <div style={{ marginTop: '16px', padding: '12px', background: '#fffbeb', borderRadius: '6px', border: '1px solid #fcd34d' }}>
          <strong style={{ color: '#b45309' }}>💡 Tip:</strong>
          <span style={{ color: '#92400e', marginLeft: '8px' }}>
            Get SharePoint Site ID and Drive ID using Microsoft Graph Explorer: 
            <code style={{ background: '#fef3c7', padding: '2px 6px', borderRadius: '4px', marginLeft: '4px' }}>
              GET /sites/{'hostname'}:/{'site-path'}
            </code>
          </span>
        </div>
      </div>

      {/* Recent Sync History */}
      {logicAppStatus?.recentHistory && logicAppStatus.recentHistory.length > 0 && (
        <div style={{
          background: 'white',
          borderRadius: '12px',
          padding: '24px',
          boxShadow: '0 1px 3px rgba(0,0,0,0.1)'
        }}>
          <h3 style={{ marginBottom: '16px' }}>📜 Recent Sync History</h3>
          <div style={{ maxHeight: '300px', overflowY: 'auto' }}>
            {logicAppStatus.recentHistory.slice().reverse().map((entry, idx) => (
              <div 
                key={idx}
                style={{
                  padding: '12px',
                  borderBottom: '1px solid #e5e7eb',
                  fontSize: '13px'
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span style={{ 
                    background: entry.errorCount > 0 ? '#fee2e2' : '#d1fae5',
                    color: entry.errorCount > 0 ? '#dc2626' : '#059669',
                    padding: '2px 8px',
                    borderRadius: '4px',
                    fontSize: '11px'
                  }}>
                    {entry.errorCount > 0 ? `${entry.errorCount} errors` : 'Success'}
                  </span>
                  <span style={{ color: '#666' }}>
                    {new Date(entry.timestamp).toLocaleString()}
                  </span>
                </div>
                <div style={{ marginTop: '8px', color: '#666' }}>
                  Files processed: {entry.filesProcessed || 0}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Delta Link Status */}
      {logicAppStatus?.hasDeltaLink !== undefined && (
        <div style={{
          marginTop: '20px',
          padding: '16px',
          background: logicAppStatus.hasDeltaLink ? '#d1fae5' : '#fef3c7',
          borderRadius: '8px',
          border: `1px solid ${logicAppStatus.hasDeltaLink ? '#10b981' : '#f59e0b'}`
        }}>
          <strong style={{ color: logicAppStatus.hasDeltaLink ? '#059669' : '#b45309' }}>
            {logicAppStatus.hasDeltaLink ? '✓ Delta Link Active' : '⚠ No Delta Link'}
          </strong>
          <p style={{ 
            fontSize: '13px', 
            color: logicAppStatus.hasDeltaLink ? '#065f46' : '#92400e',
            marginTop: '4px' 
          }}>
            {logicAppStatus.hasDeltaLink 
              ? 'Incremental sync enabled - only changed files will be processed.'
              : 'Full sync required on next run. Delta link will be created after first successful sync.'}
          </p>
        </div>
      )}
    </div>
  )
}