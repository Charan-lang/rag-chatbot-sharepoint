// Azure Logic App for SharePoint Permission Sync
// This Bicep template deploys a Logic App that syncs SharePoint permissions to Azure AI Search every 5 minutes

@description('The name of the Logic App')
param logicAppName string = 'sharepoint-permission-sync'

@description('Location for all resources')
param location string = resourceGroup().location

@description('Backend API URL')
param backendApiUrl string

@description('Backend API Key for authentication')
@secure()
param backendApiKey string

@description('Azure AD Tenant ID')
param tenantId string

@description('Azure AD Client ID')
param clientId string

@description('Azure AD Client Secret')
@secure()
param clientSecret string

@description('SharePoint Site ID')
param sharePointSiteId string

@description('SharePoint Drive ID')
param sharePointDriveId string

@description('Sync interval in minutes')
param syncIntervalMinutes int = 5

// API Connection for SharePoint (optional - if using SharePoint connector)
resource sharepointConnection 'Microsoft.Web/connections@2016-06-01' = {
  name: 'sharepoint-connection'
  location: location
  properties: {
    displayName: 'SharePoint Connection'
    api: {
      id: subscriptionResourceId('Microsoft.Web/locations/managedApis', location, 'sharepointonline')
    }
  }
}

// Logic App - Consumption Plan
resource logicApp 'Microsoft.Logic/workflows@2019-05-01' = {
  name: logicAppName
  location: location
  properties: {
    state: 'Enabled'
    definition: {
      '$schema': 'https://schema.management.azure.com/providers/Microsoft.Logic/schemas/2016-06-01/workflowdefinition.json#'
      contentVersion: '1.0.0.0'
      parameters: {
        backendApiUrl: {
          type: 'String'
          defaultValue: backendApiUrl
        }
        backendApiKey: {
          type: 'SecureString'
          defaultValue: backendApiKey
        }
        tenantId: {
          type: 'String'
          defaultValue: tenantId
        }
        clientId: {
          type: 'String'
          defaultValue: clientId
        }
        clientSecret: {
          type: 'SecureString'
          defaultValue: clientSecret
        }
        sharePointSiteId: {
          type: 'String'
          defaultValue: sharePointSiteId
        }
        driveId: {
          type: 'String'
          defaultValue: sharePointDriveId
        }
      }
      triggers: {
        Recurrence: {
          type: 'Recurrence'
          recurrence: {
            frequency: 'Minute'
            interval: syncIntervalMinutes
          }
        }
      }
      actions: {
        Get_Access_Token: {
          type: 'Http'
          inputs: {
            method: 'POST'
            uri: 'https://login.microsoftonline.com/@{parameters(\'tenantId\')}/oauth2/v2.0/token'
            headers: {
              'Content-Type': 'application/x-www-form-urlencoded'
            }
            body: 'client_id=@{parameters(\'clientId\')}&client_secret=@{parameters(\'clientSecret\')}&scope=https://graph.microsoft.com/.default&grant_type=client_credentials'
          }
          runAfter: {}
        }
        Check_Delta_Link: {
          type: 'Http'
          inputs: {
            method: 'GET'
            uri: '@{parameters(\'backendApiUrl\')}/api/logic-app/delta-link'
            headers: {
              'X-API-Key': '@{parameters(\'backendApiKey\')}'
            }
          }
          runAfter: {
            Get_Access_Token: ['Succeeded']
          }
        }
        Get_Changes_From_SharePoint: {
          type: 'Http'
          inputs: {
            method: 'GET'
            uri: '@{if(empty(body(\'Check_Delta_Link\')?[\'deltaLink\']), concat(\'https://graph.microsoft.com/v1.0/drives/\', parameters(\'driveId\'), \'/root/delta\'), body(\'Check_Delta_Link\')?[\'deltaLink\'])}'
            headers: {
              Authorization: 'Bearer @{body(\'Get_Access_Token\')?[\'access_token\']}'
            }
          }
          runAfter: {
            Check_Delta_Link: ['Succeeded']
          }
        }
        Initialize_Results: {
          type: 'InitializeVariable'
          inputs: {
            variables: [
              {
                name: 'SyncResults'
                type: 'array'
                value: []
              }
            ]
          }
          runAfter: {
            Get_Changes_From_SharePoint: ['Succeeded']
          }
        }
        Filter_Files_Only: {
          type: 'Query'
          inputs: {
            from: '@body(\'Get_Changes_From_SharePoint\')?[\'value\']'
            where: '@not(equals(item()?[\'file\'], null))'
          }
          runAfter: {
            Initialize_Results: ['Succeeded']
          }
        }
        For_Each_File: {
          type: 'Foreach'
          foreach: '@body(\'Filter_Files_Only\')'
          actions: {
            Get_File_Permissions: {
              type: 'Http'
              inputs: {
                method: 'GET'
                uri: 'https://graph.microsoft.com/v1.0/drives/@{parameters(\'driveId\')}/items/@{items(\'For_Each_File\')?[\'id\']}/permissions'
                headers: {
                  Authorization: 'Bearer @{body(\'Get_Access_Token\')?[\'access_token\']}'
                }
              }
              runAfter: {}
            }
            Call_Backend_Sync: {
              type: 'Http'
              inputs: {
                method: 'POST'
                uri: '@{parameters(\'backendApiUrl\')}/api/logic-app/sync-permission'
                headers: {
                  'Content-Type': 'application/json'
                  'X-API-Key': '@{parameters(\'backendApiKey\')}'
                }
                body: {
                  sharepoint_item_id: '@{items(\'For_Each_File\')?[\'id\']}'
                  drive_id: '@{parameters(\'driveId\')}'
                  file_name: '@{items(\'For_Each_File\')?[\'name\']}'
                  modified_at: '@{items(\'For_Each_File\')?[\'lastModifiedDateTime\']}'
                  permissions_raw: '@{body(\'Get_File_Permissions\')?[\'value\']}'
                }
              }
              runAfter: {
                Get_File_Permissions: ['Succeeded']
              }
            }
            Append_Result: {
              type: 'AppendToArrayVariable'
              inputs: {
                name: 'SyncResults'
                value: {
                  itemId: '@{items(\'For_Each_File\')?[\'id\']}'
                  fileName: '@{items(\'For_Each_File\')?[\'name\']}'
                  success: '@{equals(outputs(\'Call_Backend_Sync\')[\'statusCode\'], 200)}'
                }
              }
              runAfter: {
                Call_Backend_Sync: ['Succeeded', 'Failed']
              }
            }
          }
          runAfter: {
            Filter_Files_Only: ['Succeeded']
          }
          runtimeConfiguration: {
            concurrency: {
              repetitions: 5
            }
          }
        }
        Save_Delta_Link: {
          type: 'Http'
          inputs: {
            method: 'POST'
            uri: '@{parameters(\'backendApiUrl\')}/api/logic-app/save-delta-link'
            headers: {
              'Content-Type': 'application/json'
              'X-API-Key': '@{parameters(\'backendApiKey\')}'
            }
            body: {
              deltaLink: '@{body(\'Get_Changes_From_SharePoint\')?[\'@odata.deltaLink\']}'
              processedCount: '@{length(body(\'Filter_Files_Only\'))}'
              timestamp: '@{utcNow()}'
            }
          }
          runAfter: {
            For_Each_File: ['Succeeded', 'Failed']
          }
        }
      }
    }
  }
}

// Output the Logic App resource ID and callback URL
output logicAppId string = logicApp.id
output logicAppName string = logicApp.name
