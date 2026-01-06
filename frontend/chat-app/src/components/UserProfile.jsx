import React from 'react'
import { User } from 'lucide-react'

export default function UserProfile({ user }) {
  if (!user) return null

  const initials = user.name
    ? user.name.split(' ').map(n => n[0]).join('').toUpperCase()
    : user.username?.substring(0, 2).toUpperCase() || 'U'

  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
      <div style={{ 
        width: '36px', 
        height: '36px', 
        borderRadius: '50%', 
        background: '#eff6ff',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        fontSize: '14px',
        fontWeight: '600',
        color: '#2563eb'
      }}>
        {initials}
      </div>
      <div style={{ display: 'flex', flexDirection: 'column' }}>
        <span style={{ fontSize: '14px', fontWeight: '500', color: '#1f2937' }}>
          {user.name || user.username}
        </span>
        <span style={{ fontSize: '12px', color: '#6b7280' }}>
          {user.username || user.email}
        </span>
      </div>
    </div>
  )
}