import React, { useState } from 'react'
import { FileText, ChevronDown, ChevronUp } from 'lucide-react'

export default function SourcesList({ sources }) {
  const [expanded, setExpanded] = useState(false)

  if (!sources || sources.length === 0) return null

  return (
    <div style={{ marginTop: '16px', paddingTop: '16px', borderTop: '1px solid rgba(0,0,0,0.1)' }}>
      <button
        onClick={() => setExpanded(!expanded)}
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: '8px',
          background: 'none',
          border: 'none',
          cursor: 'pointer',
          fontSize: '13px',
          fontWeight: '600',
          color: '#6b7280',
          padding: 0
        }}
      >
        <FileText size={14} />
        {sources.length} Source{sources.length !== 1 ? 's' : ''}
        {expanded ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
      </button>

      {expanded && (
        <div style={{ marginTop: '12px', display: 'flex', flexDirection: 'column', gap: '8px' }}>
          {sources.map((source, index) => (
            <div 
              key={index}
              style={{
                padding: '10px 12px',
                background: 'rgba(0,0,0,0.03)',
                borderRadius: '6px',
                fontSize: '12px'
              }}
            >
              <div style={{ fontWeight: '600', color: '#374151', marginBottom: '4px' }}>
                {source.document_title}
              </div>
              <div style={{ color: '#6b7280' }}>
                Chunk {source.chunk_index} • Relevance: {(source.relevance_score * 100).toFixed(0)}%
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}