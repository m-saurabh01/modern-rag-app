# Authentication & Authorization

## 📋 Overview

The Modern RAG Application supports flexible authentication and authorization mechanisms to secure your document processing and retrieval endpoints. The system is designed to work both with and without authentication based on your deployment requirements.

## 🔐 Authentication Methods

### **JWT Bearer Token Authentication**
The primary authentication method uses JSON Web Tokens (JWT) for stateless authentication.

**Header Format:**
```bash
Authorization: Bearer <jwt_token>
```

**Token Structure:**
```json
{
  "sub": "user_id",
  "exp": 1640995200,
  "iat": 1640908800,
  "iss": "modern-rag-app",
  "scope": ["read", "write", "admin"],
  "user_data": {
    "username": "john.doe",
    "email": "john@example.com",
    "role": "user"
  }
}
```

### **API Key Authentication** (Optional)
For service-to-service communication, API keys can be used as an alternative.

**Header Format:**
```bash
X-API-Key: <api_key>
```

**Configuration:**
```python
# Enable API key authentication
API_KEY_ENABLED = True
API_KEY_HEADER = "X-API-Key"
VALID_API_KEYS = ["key1", "key2"]  # Configure via environment
```

## ⚙️ Configuration

### **Environment Variables**
```bash
# Authentication settings
ENABLE_AUTHENTICATION=true
JWT_SECRET_KEY="your-secret-key"
JWT_ALGORITHM="HS256"
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=60
JWT_REFRESH_TOKEN_EXPIRE_DAYS=7

# API Key settings (optional)
API_KEY_ENABLED=false
API_KEY_HEADER="X-API-Key"

# CORS settings
CORS_ALLOW_ORIGINS="http://localhost:3000,https://yourdomain.com"
CORS_ALLOW_CREDENTIALS=true
```

### **FastAPI Configuration**
```python
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from config.settings import settings

app = FastAPI()

# Security scheme
security = HTTPBearer(auto_error=False)

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    if not settings.ENABLE_AUTHENTICATION:
        return None  # Authentication disabled
    
    if not credentials:
        raise HTTPException(
            status_code=401,
            detail="Authentication required"
        )
    
    # Verify JWT token
    user = await verify_jwt_token(credentials.credentials)
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication credentials"
        )
    
    return user
```

## 🚀 Usage Examples

### **Client Authentication**

#### **Python Client**
```python
import httpx
import jwt
from datetime import datetime, timedelta

# Generate JWT token (server-side)
def create_access_token(user_data: dict):
    expire = datetime.utcnow() + timedelta(minutes=60)
    token_data = {
        **user_data,
        "exp": expire,
        "iat": datetime.utcnow(),
        "iss": "modern-rag-app"
    }
    
    token = jwt.encode(
        token_data,
        "your-secret-key",
        algorithm="HS256"
    )
    return token

# Use token in requests
async def authenticated_request():
    token = create_access_token({"sub": "user123", "scope": ["read"]})
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/query",
            headers={"Authorization": f"Bearer {token}"},
            json={"query": "What is the budget?"}
        )
        
        return response.json()
```

#### **JavaScript Client**
```javascript
// Store token (after login)
const token = await login(username, password);
localStorage.setItem('auth_token', token);

// Use token in requests
async function authenticatedQuery(query) {
    const token = localStorage.getItem('auth_token');
    
    const response = await fetch('http://localhost:8000/query', {
        method: 'POST',
        headers: {
            'Authorization': `Bearer ${token}`,
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ query })
    });
    
    if (response.status === 401) {
        // Token expired, redirect to login
        window.location.href = '/login';
        return;
    }
    
    return await response.json();
}
```

#### **cURL Examples**
```bash
# Login and get token
TOKEN=$(curl -X POST "http://localhost:8000/auth/login" \
     -H "Content-Type: application/json" \
     -d '{"username": "user", "password": "pass"}' \
     | jq -r '.access_token')

# Use token for authenticated requests
curl -X POST "http://localhost:8000/query" \
     -H "Authorization: Bearer $TOKEN" \
     -H "Content-Type: application/json" \
     -d '{"query": "What is the budget?"}'
```

## 👤 User Management

### **Login Endpoint**
```http
POST /auth/login
Content-Type: application/json

{
    "username": "john.doe",
    "password": "secure_password"
}
```

**Response:**
```json
{
    "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
    "token_type": "bearer",
    "expires_in": 3600,
    "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
    "user": {
        "id": "user123",
        "username": "john.doe",
        "email": "john@example.com",
        "role": "user",
        "permissions": ["read", "write"]
    }
}
```

### **Token Refresh**
```http
POST /auth/refresh
Content-Type: application/json

{
    "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
}
```

**Response:**
```json
{
    "access_token": "new_access_token...",
    "token_type": "bearer",
    "expires_in": 3600
}
```

### **User Profile**
```http
GET /auth/me
Authorization: Bearer <token>
```

**Response:**
```json
{
    "id": "user123",
    "username": "john.doe",
    "email": "john@example.com",
    "role": "user",
    "permissions": ["read", "write"],
    "created_at": "2024-01-15T10:00:00Z",
    "last_login": "2024-08-15T09:30:00Z"
}
```

## 🛡️ Authorization Levels

### **Permission-Based Access Control**
The system supports granular permissions for different operations.

#### **Permission Types**
- `read`: View documents and perform queries
- `write`: Upload documents and modify collections
- `delete`: Remove documents and collections
- `admin`: Full system administration
- `config`: Modify system configuration

#### **Role-Based Permissions**
```json
{
    "roles": {
        "viewer": ["read"],
        "user": ["read", "write"],
        "manager": ["read", "write", "delete"],
        "admin": ["read", "write", "delete", "admin", "config"]
    }
}
```

### **Endpoint Protection**
```python
from fastapi import Depends
from typing import List

def require_permissions(required_permissions: List[str]):
    def permission_checker(current_user = Depends(get_current_user)):
        if not current_user:
            raise HTTPException(401, "Authentication required")
        
        user_permissions = current_user.get("permissions", [])
        
        for permission in required_permissions:
            if permission not in user_permissions:
                raise HTTPException(
                    403, 
                    f"Insufficient permissions. Required: {permission}"
                )
        
        return current_user
    
    return permission_checker

# Protect endpoints
@app.post("/documents/upload")
async def upload_document(
    user = Depends(require_permissions(["write"]))
):
    # Upload logic here
    pass

@app.delete("/documents/{doc_id}")
async def delete_document(
    doc_id: str,
    user = Depends(require_permissions(["delete"]))
):
    # Delete logic here
    pass
```

### **Collection-Level Access Control**
```python
# Collection-specific permissions
async def check_collection_access(
    collection_name: str,
    required_permission: str,
    current_user = Depends(get_current_user)
):
    # Check if user has access to specific collection
    user_collections = current_user.get("collections", [])
    
    if collection_name not in user_collections:
        raise HTTPException(
            403,
            f"No access to collection: {collection_name}"
        )
    
    return True

@app.post("/collections/{collection_name}/query")
async def query_collection(
    collection_name: str,
    query_request: QueryRequest,
    user = Depends(get_current_user),
    collection_access = Depends(lambda: check_collection_access(collection_name, "read", user))
):
    # Query logic here
    pass
```

## 🔒 Security Best Practices

### **JWT Token Security**
```python
import secrets
from datetime import datetime, timedelta
import jwt

class JWTHandler:
    def __init__(self):
        self.secret_key = secrets.token_urlsafe(32)  # Generate secure key
        self.algorithm = "HS256"
        self.access_token_expire_minutes = 60
    
    def create_access_token(self, data: dict):
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes)
        
        to_encode.update({
            "exp": expire,
            "iat": datetime.utcnow(),
            "iss": "modern-rag-app",
            "jti": secrets.token_urlsafe(16)  # Unique token ID
        })
        
        return jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
    
    def verify_token(self, token: str):
        try:
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm],
                options={"verify_exp": True}
            )
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(401, "Token has expired")
        except jwt.InvalidTokenError:
            raise HTTPException(401, "Invalid token")
```

### **Password Hashing**
```python
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)
```

### **Rate Limiting**
```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Apply rate limiting to auth endpoints
@app.post("/auth/login")
@limiter.limit("5/minute")  # 5 login attempts per minute
async def login(request: Request, credentials: LoginRequest):
    # Login logic
    pass
```

## 🚨 Error Handling

### **Authentication Errors**
```python
class AuthenticationError(HTTPException):
    def __init__(self, detail: str = "Authentication failed"):
        super().__init__(status_code=401, detail=detail)

class AuthorizationError(HTTPException):
    def __init__(self, detail: str = "Insufficient permissions"):
        super().__init__(status_code=403, detail=detail)

# Error responses
{
    "error": {
        "code": "AUTHENTICATION_REQUIRED",
        "message": "Valid authentication credentials required",
        "details": {
            "missing": "Authorization header",
            "expected": "Bearer token"
        }
    }
}

{
    "error": {
        "code": "INSUFFICIENT_PERMISSIONS",
        "message": "User lacks required permissions",
        "details": {
            "required": ["write"],
            "user_permissions": ["read"]
        }
    }
}
```

### **Common Error Scenarios**
- **Missing Token**: Return 401 with clear message
- **Invalid Token**: Return 401 with token validation error
- **Expired Token**: Return 401 with expiration message
- **Insufficient Permissions**: Return 403 with required permissions
- **Rate Limit Exceeded**: Return 429 with retry information

## 🧪 Testing Authentication

### **Test Token Generation**
```python
# Create test tokens for development
def create_test_token(user_role: str = "user"):
    test_data = {
        "sub": "test_user",
        "role": user_role,
        "permissions": get_role_permissions(user_role),
        "exp": datetime.utcnow() + timedelta(hours=24)
    }
    
    return jwt.encode(test_data, "test-secret", algorithm="HS256")

# Test different permission levels
admin_token = create_test_token("admin")
user_token = create_test_token("user")
viewer_token = create_test_token("viewer")
```

### **Authentication Testing**
```python
import pytest
from fastapi.testclient import TestClient

def test_protected_endpoint_requires_auth():
    response = client.post("/query", json={"query": "test"})
    assert response.status_code == 401

def test_valid_token_allows_access():
    token = create_test_token("user")
    headers = {"Authorization": f"Bearer {token}"}
    
    response = client.post(
        "/query", 
        json={"query": "test"},
        headers=headers
    )
    assert response.status_code == 200

def test_insufficient_permissions():
    viewer_token = create_test_token("viewer")
    headers = {"Authorization": f"Bearer {viewer_token}"}
    
    response = client.post(
        "/documents/upload",
        headers=headers
    )
    assert response.status_code == 403
```

## 🔧 Optional: Disable Authentication

For development or internal deployments, authentication can be disabled:

```python
# Environment configuration
ENABLE_AUTHENTICATION=false

# All endpoints become publicly accessible
async def get_current_user_optional():
    if not settings.ENABLE_AUTHENTICATION:
        return None  # No authentication required
    
    # Standard authentication logic
    return await get_current_user()
```

## 📚 Related Documentation

- **[API Endpoints](endpoints.md)** - Complete endpoint reference
- **[Rate Limiting](rate_limiting.md)** - API rate limiting and quotas
- **[Error Handling](../core/exception_handling.md)** - Error handling strategies
- **[Configuration](../config/settings.md)** - System configuration guide

---

**The Modern RAG Application provides flexible, secure authentication that can be easily enabled or disabled based on your deployment needs.**
