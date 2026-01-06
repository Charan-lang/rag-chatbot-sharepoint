# Azure Logic App - SharePoint Permission Sync

This guide explains how to set up Azure Logic Apps for automatic permission synchronization between SharePoint and Azure AI Search.

## Overview

The Logic App runs every **5 minutes** and:
1. Uses **Delta Query** to get only changed files (efficient)
2. Fetches permissions for each changed file
3. Calls your backend API to update Azure AI Search index
4. Saves the delta link for the next run (incremental sync)

## Architecture

```
┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐
│   SharePoint    │      │   Azure Logic   │      │   Backend API   │
│   Online        │◄────►│   App (5 min)   │─────►│   (FastAPI)     │
└─────────────────┘      └─────────────────┘      └────────┬────────┘
                                                          │
                                                          ▼
                                                 ┌─────────────────┐
                                                 │  Azure AI       │
                                                 │  Search Index   │
                                                 └─────────────────┘
```

## Prerequisites

1. **Azure Subscription** with permissions to create Logic Apps
2. **App Registration** with these Microsoft Graph permissions:
   - `Files.Read.All` (Application)
   - `Sites.Read.All` (Application)
3. **Backend API** deployed and accessible
4. **SharePoint Site ID** and **Drive ID**

## Getting SharePoint IDs

### Using Microsoft Graph Explorer

1. Go to [Microsoft Graph Explorer](https://developer.microsoft.com/en-us/graph/graph-explorer)
2. Sign in with your account
3. Run these queries:

**Get Site ID:**
```
GET https://graph.microsoft.com/v1.0/sites/{tenant}.sharepoint.com:/sites/{site-name}
```

**Get Drive ID:**
```
GET https://graph.microsoft.com/v1.0/sites/{site-id}/drives
```

### Using Azure CLI
```bash
# Get site ID
az rest --method get --url "https://graph.microsoft.com/v1.0/sites/root:/sites/YourSiteName"

# Get drives
az rest --method get --url "https://graph.microsoft.com/v1.0/sites/{site-id}/drives"
```

## Deployment Options

### Option 1: Azure Portal (Manual)

1. Go to Azure Portal → Create Resource → Logic App
2. Choose **Consumption** plan (pay-per-execution) or **Standard** (for production)
3. Import the workflow from `azure-resources/logic-apps/simple-permission-sync.json`
4. Update the `Config` variable with your values

### Option 2: Bicep/ARM Template (Recommended)

```bash
# Deploy using Azure CLI
az deployment group create \
  --resource-group YOUR_RESOURCE_GROUP \
  --template-file azure-resources/deployment/logic-app-permission-sync.bicep \
  --parameters \
    backendApiUrl="https://your-backend.azurewebsites.net" \
    backendApiKey="YOUR_SECRET_KEY" \
    tenantId="YOUR_TENANT_ID" \
    clientId="YOUR_CLIENT_ID" \
    clientSecret="YOUR_CLIENT_SECRET" \
    sharePointSiteId="YOUR_SITE_ID" \
    sharePointDriveId="YOUR_DRIVE_ID"
```

### Option 3: PowerShell
```powershell
New-AzResourceGroupDeployment `
  -ResourceGroupName "YOUR_RESOURCE_GROUP" `
  -TemplateFile "azure-resources/deployment/logic-app-permission-sync.bicep" `
  -backendApiUrl "https://your-backend.azurewebsites.net" `
  -backendApiKey "YOUR_SECRET_KEY" `
  -tenantId "YOUR_TENANT_ID" `
  -clientId "YOUR_CLIENT_ID" `
  -clientSecret "YOUR_CLIENT_SECRET" `
  -sharePointSiteId "YOUR_SITE_ID" `
  -sharePointDriveId "YOUR_DRIVE_ID"
```

## Configuration Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `backendApiUrl` | Your backend API base URL | `https://myapp.azurewebsites.net` |
| `backendApiKey` | API key (same as SECRET_KEY in .env) | `your-secret-key-here` |
| `tenantId` | Azure AD Tenant ID | `42ae93a2-7d09-43aa-8b42-87222081daaa` |
| `clientId` | App Registration Client ID | `3fd1cf85-6819-4393-8c60-01631e86f400` |
| `clientSecret` | App Registration Client Secret | `your-client-secret` |
| `sharePointSiteId` | SharePoint Site ID | `contoso.sharepoint.com,xxxxx,xxxxx` |
| `sharePointDriveId` | Document Library Drive ID | `b!xxxxxxxxxxxx` |

## Backend API Endpoints

The Logic App calls these endpoints:

### GET `/api/logic-app/delta-link`
Gets the saved delta link for incremental sync.

**Headers:**
- `X-API-Key`: Your API key

**Response:**
```json
{
  "deltaLink": "https://graph.microsoft.com/v1.0/drives/.../delta?token=...",
  "lastSyncTimestamp": "2025-01-02T10:00:00Z"
}
```

### POST `/api/logic-app/sync-permission`
Syncs permissions for a single file.

**Headers:**
- `X-API-Key`: Your API key
- `Content-Type`: application/json

**Body:**
```json
{
  "sharepoint_item_id": "01ABCDEF...",
  "drive_id": "b!xxxx",
  "file_name": "document.pdf",
  "modified_at": "2025-01-02T10:00:00Z",
  "permissions_raw": [
    {
      "id": "...",
      "roles": ["read"],
      "grantedToV2": {
        "user": {
          "id": "user-guid",
          "email": "user@company.com"
        }
      }
    }
  ]
}
```

**Response:**
```json
{
  "success": true,
  "document_id": "doc-123",
  "chunks_updated": 5,
  "message": "Successfully updated 5 chunks"
}
```

### POST `/api/logic-app/save-delta-link`
Saves the delta link for the next sync run.

**Body:**
```json
{
  "deltaLink": "https://graph.microsoft.com/v1.0/...",
  "processedCount": 10,
  "timestamp": "2025-01-02T10:05:00Z"
}
```

### GET `/api/logic-app/sync-status`
Gets the current sync status (for monitoring).

**Response:**
```json
{
  "lastSyncTimestamp": "2025-01-02T10:05:00Z",
  "deltaLink": true,
  "totalSyncs": 100,
  "totalFilesProcessed": 500,
  "totalErrors": 2,
  "recentHistory": [...]
}
```

## How Delta Query Works

1. **First Run**: Calls `/drives/{driveId}/root/delta` to get ALL files
2. **Subsequent Runs**: Uses saved `deltaLink` to get ONLY changes
3. **Benefits**: Efficient, only processes what changed

```
First Run:  GET /drives/.../delta → Returns all files + deltaLink
Second Run: GET {deltaLink} → Returns only changed files + new deltaLink
```

## Monitoring

### Azure Portal
1. Go to your Logic App
2. Click "Overview" to see run history
3. Click on individual runs to see details

### Admin Dashboard
1. Go to Admin App → Permission Sync
2. Select "Logic App (Scheduled)" tab
3. View sync statistics and history

### Alerts
Configure alerts in Azure Portal:
- Failed runs
- High error rate
- Long execution time

## Troubleshooting

### Logic App Not Running
1. Check Logic App is **Enabled** in Azure Portal
2. Verify trigger configuration (Recurrence)
3. Check for quota limits (Consumption plan)

### Authentication Errors (401)
1. Verify `clientId` and `clientSecret` are correct
2. Check App Registration has required permissions
3. Ensure admin consent is granted

### Permission Not Updated
1. Check document has `sharepoint_item_id` in search index
2. Run manual sync from Admin Dashboard
3. Re-ingest documents to populate SharePoint IDs

### API Key Invalid
1. Ensure `backendApiKey` matches `SECRET_KEY` in backend `.env`
2. Check for extra spaces in parameter values

## Cost Estimation

**Consumption Plan (Pay-per-execution):**
- ~$0.000025 per action
- Running every 5 min = 288 runs/day
- With 10 files avg = ~2,880 actions/day
- **Estimated: ~$2-5/month**

**Standard Plan:**
- Fixed monthly cost
- Better for high-volume scenarios

## Security Best Practices

1. **Use Key Vault** for secrets:
   ```bicep
   clientSecret: {
     reference: {
       keyVault: { id: keyVaultResourceId }
       secretName: 'SharePointClientSecret'
     }
   }
   ```

2. **Use Managed Identity** instead of client credentials

3. **Restrict API access** with IP filtering

4. **Enable diagnostic logging** for audit trail

## Workflow JSON Files

| File | Description |
|------|-------------|
| `simple-permission-sync.json` | HTTP-only workflow (recommended) |
| `sharepoint-sync-workflow.json` | Uses SharePoint connector |
| `permission-sync-graph-api.json` | Advanced Graph API workflow |

## Related Documentation

- [Microsoft Graph Delta Query](https://docs.microsoft.com/en-us/graph/delta-query-overview)
- [Azure Logic Apps Documentation](https://docs.microsoft.com/en-us/azure/logic-apps/)
- [SharePoint REST API](https://docs.microsoft.com/en-us/sharepoint/dev/sp-add-ins/get-to-know-the-sharepoint-rest-service)
