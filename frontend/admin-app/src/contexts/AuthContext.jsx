import React, { createContext, useContext, useState, useEffect } from 'react'
import { authService } from '../services/auth'

const AuthContext = createContext(null)

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const token = localStorage.getItem('admin_token')
    if (token) {
      authService.getCurrentUser()
        .then(setUser)
        .catch(() => {
          localStorage.removeItem('admin_token')
        })
        .finally(() => setLoading(false))
    } else {
      setLoading(false)
    }
  }, [])

  const login = async (username, password) => {
    const data = await authService.login(username, password)
    localStorage.setItem('admin_token', data.access_token)
    const userInfo = await authService.getCurrentUser()
    setUser(userInfo)
    return data
  }

  const logout = () => {
    localStorage.removeItem('admin_token')
    setUser(null)
  }

  return (
    <AuthContext.Provider value={{ user, login, logout, loading }}>
      {children}
    </AuthContext.Provider>
  )
}

export const useAuth = () => {
  const context = useContext(AuthContext)
  if (!context) {
    throw new Error('useAuth must be used within AuthProvider')
  }
  return context
}