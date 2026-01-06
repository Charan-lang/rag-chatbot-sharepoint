import React, { createContext, useContext, useState, useEffect } from 'react'
import { useMsal } from '@azure/msal-react'
import { loginRequest } from '../services/msalConfig'
import { setAuthToken } from '../services/api'

const AuthContext = createContext(null)

export const AuthProvider = ({ children }) => {
  const { instance, accounts } = useMsal()
  const [user, setUser] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const initializeAuth = async () => {
      if (accounts.length > 0) {
        setUser(accounts[0])
        await acquireToken()
      }
      setLoading(false)
    }
    
    initializeAuth()
  }, [accounts])

  const acquireToken = async () => {
    if (accounts.length === 0) {
      console.warn('No accounts available to acquire token')
      return null
    }

    const request = {
      ...loginRequest,
      account: accounts[0]
    }

    try {
      console.log('Acquiring token silently...')
      const response = await instance.acquireTokenSilent(request)
      console.log('Token acquired successfully')
      setAuthToken(response.accessToken)
      return response.accessToken
    } catch (error) {
      console.error('Silent token acquisition failed:', error)
      console.log('Redirecting to interactive login...')
      instance.acquireTokenRedirect(request)
      return null
    }
  }

  const login = () => {
    instance.loginRedirect(loginRequest)
  }

  const logout = () => {
    setAuthToken(null)
    instance.logoutRedirect()
  }

  return (
    <AuthContext.Provider value={{ user, login, logout, loading, acquireToken }}>
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