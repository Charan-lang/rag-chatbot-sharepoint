# Permissions-Aware RAG Chatbot

Enterprise-grade RAG chatbot with document-level permissions using Azure OpenAI, Azure AI Search, and SharePoint integration.

## 🚀 Features

- ✅ **Permission-Aware Retrieval**: Only shows content users are authorized to see
- ✅ **SharePoint Integration**: Automatically syncs documents and permissions
- ✅ **Azure OpenAI**: GPT-4 powered responses with source citations
- ✅ **Admin Portal**: Manage documents, chunks, and permissions
- ✅ **SSO Authentication**: Microsoft Entra ID integration for chat users
- ✅ **Audit Logging**: Complete traceability of all actions
- ✅ **Vector Search**: Semantic search with Azure AI Search

## 📋 Prerequisites

- Python 3.9+
- Node.js 18+
- Azure subscription with:
  - Azure OpenAI Service
  - Azure AI Search
  - SharePoint Online
  - Azure Active Directory (Entra ID)

## 🏗️ Project Structure

```
permissions-aware-rag/
├── backend/              # FastAPI backend
├── frontend/
│   ├── admin-app/       # Admin portal (React)
│   └── chat-app/        # Chat interface (React + MSAL)
├── docker-compose.yml
└── README.md
```

## ⚙️ Setup

### 1. Backend Setup

```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your Azure credentials
uvicorn app.main:app --reload
```

Backend runs at: `http://localhost:8000`

### 2. Admin App Setup

```bash
cd frontend/admin-app
npm install
cp .env.example .env
npm run dev
```

Admin app runs at: `http://localhost:5173`

### 3. Chat App Setup

```bash
cd frontend/chat-app
npm install
cp .env.example .env
# Edit .env with your Azure App Registration details
npm run dev
```

Chat app runs at: `http://localhost:5174`

## 🔧 Configuration

### Azure Resources Needed

1. **Azure OpenAI**: Deploy GPT-4 and text-embedding-ada-002
2. **Azure AI Search**: Standard tier or higher
3. **App Registrations**: Create 3 apps (Backend API, Admin, Chat)
4. **SharePoint**: Configure app permissions for document access

### Environment Variables

**Backend** (`backend/.env`):
- Azure OpenAI credentials
- Azure AI Search credentials
- SharePoint app credentials
- Admin credentials

**Chat App** (`frontend/chat-app/.env`):
- Tenant ID
- Client ID (from Chat App Registration)
- Redirect URI

## 📚 Usage

### Admin Portal

1. Login with admin credentials
2. Sync documents from SharePoint
3. View chunks and permissions
4. Monitor audit logs

### Chat Application

1. Sign in with Microsoft account
2. Ask questions about your documents
3. Get AI responses with source citations
4. Only see content you have access to

## 🐳 Docker Deployment

```bash
docker-compose up -d
```

## 🧪 Testing

```bash
# Backend tests
cd backend
pytest

# Frontend tests
cd frontend/admin-app
npm test
```

## 📖 API Documentation

Once backend is running, visit:
- Swagger UI: `http://localhost:8000/api/docs`
- ReDoc: `http://localhost:8000/api/redoc`

## 🔐 Security

- Document-level permission filtering
- Chunk-level access control
- SSO authentication
- Audit logging
- Zero unauthorized data leakage

## 📄 License

MIT License

## 🤝 Support

For issues and questions, please open an issue on GitHub.

