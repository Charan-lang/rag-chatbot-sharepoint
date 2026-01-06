import api from './api'

export const authService = {
  login: async (username, password) => {
    const response = await api.post('/auth/admin/login', { username, password })
    return response.data
  },

  getCurrentUser: async () => {
    const response = await api.get('/auth/me')
    return response.data
  },

  logout: async () => {
    const response = await api.post('/auth/logout')
    return response.data
  }
}