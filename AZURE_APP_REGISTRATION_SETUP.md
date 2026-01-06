# Azure App Registration Setup - Group Claims Configuration

This guide configures your Azure AD app to include group memberships in tokens **without requiring user consent**.

## Why This Matters

- **Before**: Backend called Graph API `/me/memberOf` → required `GroupMember.Read.All` permission → admin consent required → "Request pending" screen
- **After**: Groups included in token claims → backend reads from token → no Graph calls → no user consent needed

## Your Current Setup

- **Tenant**: `42ae93a2-7d09-43aa-8b42-87222081daaa` (happiestbarik.onmicrosoft.com)
- **App Registration**: Client ID `3fd1cf85-6819-4393-8c60-01631e86f400`
- **Admin Account**: paresh@happiestbarik.onmicrosoft.com
- **Permission Model**: 
  - **Delegated** (User sign-in): User.Read, openid, profile
  - **Application** (Server-to-server): Sites.Read.All, Group.Read.All (for SharePoint/Graph backend calls)

## Configuration Steps

### 1. Navigate to App Registration

1. Sign in to [Azure Portal](https://portal.azure.com) **as paresh@happiestbarik.onmicrosoft.com**
2. Go to **Microsoft Entra ID** (formerly Azure AD)
3. Select **App registrations**
4. Find and click your app with Client ID: `3fd1cf85-6819-4393-8c60-01631e86f400`

### 2. Configure Token Claims (CRITICAL STEP)

1. In the left menu, click **Token configuration**
2. Click **+ Add groups claim**
3. Select the group types to include:
   - ✅ **Security groups** (recommended)
   - ✅ **Distribution lists** (optional)
   - ⬜ **Directory roles** (optional - only if you use admin roles)
4. Under **Customize token properties by type**, for both **ID** and **Access** tokens:
   - Select **Group ID** (this returns the object ID of groups)
   - *Alternatively, if you prefer group names instead of IDs, select "sAMAccountName" or "DNS domain name", but Group ID is recommended*
5. Click **Add**

### 3. Grant Admin Consent for Delegated Permissions

Since you're using Application permissions already, you need to also grant admin consent for the delegated permissions users need:

1. Click **API permissions** in left menu
2. You should see:
   - **Delegated permissions** (for user sign-in):
     - ✅ `User.Read` (Microsoft Graph)
     - ✅ `openid`
     - ✅ `profile`
   - **Application permissions** (for backend server calls):
     - ✅ `Sites.Read.All` or similar (for SharePoint)
     - ✅ `Group.Read.All` (if reading groups server-side)

3. Click **Grant admin consent for happiestbarik** button at the top
4. Click **Yes** to confirm
5. All permissions should show green checkmarks under "Status"

### 4. Handle Group Overage (Optional but Recommended)

If users belong to many groups (>200), Azure includes `hasgroups` claim instead of listing all groups.

**To handle this:**
1. Keep the backend code as-is (it already checks for overage)
2. Grant **optional** admin consent for `GroupMember.Read.All` delegated permission
3. The backend will automatically call Graph API only when overage occurs

**To add the permission (for overage scenarios only):**
1. Click **API permissions** in left menu
2. Click **+ Add a permission**
3. Select **Microsoft Graph** → **Delegated permissions**
4. Search and select: `GroupMember.Read.All`
5. Click **Add permissions**
6. Click **Grant admin consent for happiestbarik** (needed for overage cases with >200 groups)

### 5. Verify Redirect URIs

1. Click **Authentication** in left menu
2. Under **Platform configurations** → **Single-page application**
3. Ensure these redirect URIs are listed:
   - `http://localhost:5174`
   - `http://localhost:5174/` (with trailing slash)
   - `http://127.0.0.1:5174` (optional, for local development)
4. If deploying to production, add your production URLs
5. Click **Save**

### 6. Verify API Permissions

1. Click **API permissions**
2. You should see **at minimum**:
   
   **Delegated permissions** (for user authentication):
   - ✅ `User.Read` (Microsoft Graph, Delegated) - **Admin consent granted**
   - ✅ `openid` (implicit) - **Admin consent granted**
   - ✅ `profile` (implicit) - **Admin consent granted**
   
   **Application permissions** (for backend operations):
   - ✅ Your SharePoint/Graph permissions - **Admin consent granted**

3. **IMPORTANT**: All delegated permissions must show "Granted for happiestbarik" status

## Testing

### 1. Clear Browser Cache
```bash
# Open browser DevTools (F12)
# Application → Storage → Clear site data
# Or use incognito/private browsing
```

### 2. Restart Frontend
```bash
cd frontend/chat-app
npm run dev
```

### 3. Sign In
1. Navigate to `http://localhost:5174`
2. Click **Sign in with Microsoft**
3. You should see standard Azure AD login (no admin approval screen)
4. After successful login, you should see the chat interface

### 4. Verify Token Contains Groups

Open browser DevTools → Network tab → Filter by `/api/chat/query` → Check request headers:

```bash
# Decode the Authorization Bearer token at https://jwt.ms/
# Look for "groups" claim in the token payload:
{
  "aud": "b50fe1ba-7332-48d5-9ac9-b775e85db1d1",
  "oid": "user-object-id",
  "groups": [
    "group-id-1",
    "group-id-2",
    "group-id-3"
  ],
  ...
}
```

## Troubleshooting

### Still Seeing "Request Pending"?
- **Cause**: Admin consent not granted for delegated permissions
- **Solution**: Sign in as **paresh@happiestbarik.onmicrosoft.com** and:
  1. Go to Azure Portal → Microsoft Entra ID → App registrations → Your app
  2. Click **API permissions**
  3. Click **Grant admin consent for happiestbarik**
  4. Confirm by clicking **Yes**

**OR use admin consent URL:**
```
https://login.microsoftonline.com/42ae93a2-7d09-43aa-8b42-87222081daaa/adminconsent?client_id=3fd1cf85-6819-4393-8c60-01631e86f400&redirect_uri=http://localhost:5174
```
- Sign in as paresh@happiestbarik.onmicrosoft.com
- Click **Accept**
- All users can now sign in without approval

### Groups Not Appearing in Token?
- Wait 5-10 minutes after configuring token claims (Azure propagation delay)
- Sign out and sign in again
- Check Token configuration page - ensure groups claim was saved
- User must belong to at least one security group

### Empty Groups Array?
- User may not belong to any security groups
- Test with a user who is in at least one security group
- Verify in Azure Portal → Users → Select user → Groups

### Backend Returns "Invalid Entra ID token"?
- Backend now accepts Microsoft Graph tokens (fixed)
- Check backend logs: Look for "Token verified successfully with audience"
- Ensure backend was restarted after code changes
- Verify token is being sent in Authorization header (check browser DevTools → Network)

## Production Deployment

When deploying to production:

1. Update redirect URIs in Azure Portal:
   - Add your production domain (e.g., `https://chat.yourdomain.com`)
2. Update frontend `.env`:
   ```env
   VITE_REDIRECT_URI=https://chat.yourdomain.com
   ```
3. Consider these optional enhancements:
   - Enable **Conditional Access** policies
   - Configure **Token lifetime** policies
   - Set up **Continuous Access Evaluation** (CAE)
   - Enable **sign-in logs** for monitoring

## Summary

✅ **What changed:**
- Backend now reads groups from token claims (primary method)
- Fallback to Graph API only if groups not in token or overage occurs
- No admin consent required for basic usage

✅ **User experience:**
- Sign in → Instant access (no approval waiting)
- Groups automatically resolved
- Permission filtering works immediately

✅ **Next steps:**
1. Configure token claims in Azure Portal (steps above)
2. Test with a user account
3. Verify groups appear in decoded token
4. Test chat queries to ensure permission filtering works
