import axios from 'axios'

const API_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000'

const api = axios.create({
  baseURL: `${API_URL}/api`,
  headers: {
    'Content-Type': 'application/json',
  },
})

let currentToken = null

export const setAuthToken = (token) => {
  currentToken = token
  if (token) {
    console.log('Setting auth token in API headers')
    api.defaults.headers.common['Authorization'] = `Bearer ${token}`
  } else {
    console.log('Removing auth token from API headers')
    delete api.defaults.headers.common['Authorization']
  }
}

// Add request interceptor to ensure token is set
api.interceptors.request.use(
  (config) => {
    if (currentToken && !config.headers.Authorization) {
      config.headers.Authorization = `Bearer ${currentToken}`
    }
    console.log(`API Request: ${config.method.toUpperCase()} ${config.url}`)
    if (config.headers.Authorization) {
      console.log('Authorization header present:', config.headers.Authorization.substring(0, 20) + '...')
    } else {
      console.warn('No Authorization header in request!')
    }
    return config
  },
  (error) => {
    console.error('Request interceptor error:', error)
    return Promise.reject(error)
  }
)

// Handle 401 errors
api.interceptors.response.use(
  (response) => {
    console.log(`API Response: ${response.status} ${response.config.url}`)
    return response
  },
  (error) => {
    if (error.response?.status === 401) {
      console.error('401 Unauthorized - Token may be invalid or expired')
      // Don't auto-redirect, let the component handle it
      // This prevents redirect loops
    }
    console.error('API Error:', error.response?.status, error.response?.data)
    return Promise.reject(error)
  }
)

export default api