import { PublicClientApplication } from '@azure/msal-browser'

const msalConfig = {
  auth: {
    clientId: import.meta.env.VITE_CLIENT_ID,
    // Use 'common' for multi-tenant or specific tenant ID for single-tenant
    authority: import.meta.env.VITE_TENANT_ID 
      ? `https://login.microsoftonline.com/${import.meta.env.VITE_TENANT_ID}`
      : 'https://login.microsoftonline.com/common',
    redirectUri: import.meta.env.VITE_REDIRECT_URI || 'http://localhost:5174'
  },
  cache: {
    cacheLocation: 'sessionStorage',
    storeAuthStateInCookie: false
  }
}

export const loginRequest = {
  // Use the client ID as scope to get an access token for this application
  scopes: [`api://${import.meta.env.VITE_CLIENT_ID}/User.Read`]
}

// Fallback for simple ID token (if API scope not configured)
export const loginRequestBasic = {
  scopes: ['openid', 'profile', 'email']
}

export const msalInstance = new PublicClientApplication(msalConfig)

