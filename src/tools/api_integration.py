"""
API Integration Tool for the Agentic AI system.
"""

import requests
import aiohttp
import asyncio
import json
import base64
import hmac
import hashlib
import time
from typing import Dict, Any, List, Optional, Union
from langchain.tools import BaseTool
from langchain.pydantic_v1 import BaseModel, Field
from urllib.parse import urlencode, urljoin
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class APIRequestInput(BaseModel):
    """Input for API requests."""
    url: str = Field(description="The API endpoint URL")
    method: str = Field(default="GET", description="HTTP method (GET, POST, PUT, DELETE, PATCH)")
    headers: Optional[Dict[str, str]] = Field(default=None, description="HTTP headers")
    params: Optional[Dict[str, Any]] = Field(default=None, description="URL parameters")
    data: Optional[Union[Dict[str, Any], str]] = Field(default=None, description="Request body data")
    auth_type: Optional[str] = Field(default=None, description="Authentication type (bearer, basic, api_key)")
    auth_data: Optional[Dict[str, str]] = Field(default=None, description="Authentication data")
    timeout: int = Field(default=30, description="Request timeout in seconds")


class APIIntegrationTool(BaseTool):
    """Tool for integrating with various APIs."""
    
    name = "api_integration"
    description = """
    Integrate with various APIs:
    - request: Make HTTP requests to APIs
    - oauth: Handle OAuth authentication
    - webhook: Create and manage webhooks
    - rate_limit: Handle rate limiting
    - batch: Process batch API requests
    """
    
    # Class-level configuration
    _config = {
        "api_keys": {},
        "session": None,
        "rate_limits": {}
    }
    
    def __init__(self, api_keys: Optional[Dict[str, str]] = None):
        super().__init__()
        # Store configuration in class-level dict to avoid Pydantic issues
        self._config["api_keys"] = api_keys or {}
    
    def _run(self, operation: str, **kwargs) -> Dict[str, Any]:
        """Execute API operation synchronously."""
        try:
            if operation == "request":
                return self._make_request(**kwargs)
            elif operation == "oauth":
                return self._handle_oauth(**kwargs)
            elif operation == "webhook":
                return self._handle_webhook(**kwargs)
            elif operation == "rate_limit":
                return self._handle_rate_limit(**kwargs)
            elif operation == "batch":
                return self._process_batch(**kwargs)
            else:
                raise ValueError(f"Unknown operation: {operation}")
        except Exception as e:
            logger.error(f"Error in API operation {operation}: {str(e)}")
            return {"error": str(e), "success": False}
    
    async def _arun(self, operation: str, **kwargs) -> Dict[str, Any]:
        """Execute API operation asynchronously."""
        try:
            if operation == "request":
                return await self._make_request_async(**kwargs)
            elif operation == "oauth":
                return await self._handle_oauth_async(**kwargs)
            elif operation == "webhook":
                return await self._handle_webhook_async(**kwargs)
            elif operation == "rate_limit":
                return await self._handle_rate_limit_async(**kwargs)
            elif operation == "batch":
                return await self._process_batch_async(**kwargs)
            else:
                raise ValueError(f"Unknown operation: {operation}")
        except Exception as e:
            logger.error(f"Error in async API operation {operation}: {str(e)}")
            return {"error": str(e), "success": False}
    
    def _make_request(self, url: str, method: str = "GET", headers: Optional[Dict[str, str]] = None, 
                     params: Optional[Dict[str, Any]] = None, data: Optional[Union[Dict[str, Any], str]] = None,
                     auth_type: Optional[str] = None, auth_data: Optional[Dict[str, str]] = None,
                     timeout: int = 30) -> Dict[str, Any]:
        """Make HTTP request to API."""
        try:
            # Prepare headers
            request_headers = headers or {}
            
            # Handle authentication
            if auth_type and auth_data:
                auth_headers = self._prepare_auth(auth_type, auth_data)
                request_headers.update(auth_headers)
            
            # Prepare request data
            request_data = None
            if data:
                if isinstance(data, dict):
                    request_data = json.dumps(data)
                    request_headers['Content-Type'] = 'application/json'
                else:
                    request_data = data
            
            # Make request
            response = requests.request(
                method=method.upper(),
                url=url,
                headers=request_headers,
                params=params,
                data=request_data,
                timeout=timeout
            )
            
            # Parse response
            try:
                response_data = response.json()
            except:
                response_data = response.text
            
            return {
                "success": True,
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "data": response_data,
                "url": response.url,
                "method": method.upper(),
                "timestamp": datetime.now().isoformat()
            }
        
        except requests.exceptions.Timeout:
            return {
                "error": f"Request timed out after {timeout} seconds",
                "success": False
            }
        except requests.exceptions.ConnectionError:
            return {
                "error": "Connection error",
                "success": False
            }
        except Exception as e:
            logger.error(f"Error making API request: {str(e)}")
            return {"error": str(e), "success": False}
    
    async def _make_request_async(self, url: str, method: str = "GET", headers: Optional[Dict[str, str]] = None,
                                 params: Optional[Dict[str, Any]] = None, data: Optional[Union[Dict[str, Any], str]] = None,
                                 auth_type: Optional[str] = None, auth_data: Optional[Dict[str, str]] = None,
                                 timeout: int = 30) -> Dict[str, Any]:
        """Make async HTTP request to API."""
        try:
            # Prepare headers
            request_headers = headers or {}
            
            # Handle authentication
            if auth_type and auth_data:
                auth_headers = self._prepare_auth(auth_type, auth_data)
                request_headers.update(auth_headers)
            
            # Prepare request data
            request_data = None
            if data:
                if isinstance(data, dict):
                    request_data = json.dumps(data)
                    request_headers['Content-Type'] = 'application/json'
                else:
                    request_data = data
            
            # Create session if not exists
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            # Make async request
            async with self.session.request(
                method=method.upper(),
                url=url,
                headers=request_headers,
                params=params,
                data=request_data,
                timeout=aiohttp.ClientTimeout(total=timeout)
            ) as response:
                
                # Parse response
                try:
                    response_data = await response.json()
                except:
                    response_data = await response.text()
                
                return {
                    "success": True,
                    "status_code": response.status,
                    "headers": dict(response.headers),
                    "data": response_data,
                    "url": str(response.url),
                    "method": method.upper(),
                    "timestamp": datetime.now().isoformat()
                }
        
        except asyncio.TimeoutError:
            return {
                "error": f"Request timed out after {timeout} seconds",
                "success": False
            }
        except Exception as e:
            logger.error(f"Error making async API request: {str(e)}")
            return {"error": str(e), "success": False}
    
    def _prepare_auth(self, auth_type: str, auth_data: Dict[str, str]) -> Dict[str, str]:
        """Prepare authentication headers."""
        headers = {}
        
        if auth_type == "bearer":
            token = auth_data.get("token")
            if token:
                headers["Authorization"] = f"Bearer {token}"
        
        elif auth_type == "basic":
            username = auth_data.get("username")
            password = auth_data.get("password")
            if username and password:
                credentials = base64.b64encode(f"{username}:{password}".encode()).decode()
                headers["Authorization"] = f"Basic {credentials}"
        
        elif auth_type == "api_key":
            key = auth_data.get("key")
            header_name = auth_data.get("header", "X-API-Key")
            if key:
                headers[header_name] = key
        
        return headers
    
    def _handle_oauth(self, provider: str, client_id: str, client_secret: str, 
                     redirect_uri: str, scope: str = None) -> Dict[str, Any]:
        """Handle OAuth authentication flow."""
        try:
            # This is a simplified OAuth implementation
            # In a real implementation, you'd need to handle the full OAuth flow
            
            if provider == "google":
                auth_url = "https://accounts.google.com/o/oauth2/v2/auth"
                token_url = "https://oauth2.googleapis.com/token"
            elif provider == "github":
                auth_url = "https://github.com/login/oauth/authorize"
                token_url = "https://github.com/login/oauth/access_token"
            elif provider == "twitter":
                auth_url = "https://twitter.com/i/oauth2/authorize"
                token_url = "https://api.twitter.com/2/oauth2/token"
            else:
                return {
                    "error": f"Unsupported OAuth provider: {provider}",
                    "success": False
                }
            
            # Build authorization URL
            auth_params = {
                "client_id": client_id,
                "redirect_uri": redirect_uri,
                "response_type": "code"
            }
            
            if scope:
                auth_params["scope"] = scope
            
            authorization_url = f"{auth_url}?{urlencode(auth_params)}"
            
            return {
                "success": True,
                "authorization_url": authorization_url,
                "token_url": token_url,
                "provider": provider,
                "message": "Visit the authorization URL to complete OAuth flow"
            }
        
        except Exception as e:
            logger.error(f"Error handling OAuth: {str(e)}")
            return {"error": str(e), "success": False}
    
    def _handle_webhook(self, action: str, **kwargs) -> Dict[str, Any]:
        """Handle webhook operations."""
        try:
            if action == "create":
                return self._create_webhook(**kwargs)
            elif action == "verify":
                return self._verify_webhook(**kwargs)
            elif action == "process":
                return self._process_webhook(**kwargs)
            else:
                return {
                    "error": f"Unknown webhook action: {action}",
                    "success": False
                }
        except Exception as e:
            logger.error(f"Error handling webhook: {str(e)}")
            return {"error": str(e), "success": False}
    
    def _create_webhook(self, url: str, secret: str = None, events: List[str] = None) -> Dict[str, Any]:
        """Create a webhook configuration."""
        try:
            webhook_config = {
                "url": url,
                "secret": secret,
                "events": events or ["*"],
                "created_at": datetime.now().isoformat(),
                "active": True
            }
            
            return {
                "success": True,
                "webhook": webhook_config,
                "message": "Webhook configuration created"
            }
        
        except Exception as e:
            logger.error(f"Error creating webhook: {str(e)}")
            return {"error": str(e), "success": False}
    
    def _verify_webhook(self, signature: str, payload: str, secret: str) -> Dict[str, Any]:
        """Verify webhook signature."""
        try:
            # Calculate expected signature
            expected_signature = hmac.new(
                secret.encode(),
                payload.encode(),
                hashlib.sha256
            ).hexdigest()
            
            # Compare signatures
            is_valid = hmac.compare_digest(signature, expected_signature)
            
            return {
                "success": True,
                "valid": is_valid,
                "message": "Signature verified" if is_valid else "Invalid signature"
            }
        
        except Exception as e:
            logger.error(f"Error verifying webhook: {str(e)}")
            return {"error": str(e), "success": False}
    
    def _process_webhook(self, payload: Dict[str, Any], headers: Dict[str, str] = None) -> Dict[str, Any]:
        """Process webhook payload."""
        try:
            # Extract common webhook information
            event_type = headers.get("X-Event-Type") or payload.get("event_type")
            timestamp = headers.get("X-Timestamp") or payload.get("timestamp")
            
            return {
                "success": True,
                "event_type": event_type,
                "timestamp": timestamp,
                "payload": payload,
                "processed_at": datetime.now().isoformat()
            }
        
        except Exception as e:
            logger.error(f"Error processing webhook: {str(e)}")
            return {"error": str(e), "success": False}
    
    def _handle_rate_limit(self, api_name: str, requests_per_minute: int = 60) -> Dict[str, Any]:
        """Handle rate limiting."""
        try:
            current_time = time.time()
            
            # Initialize rate limit info if not exists
            if api_name not in self.rate_limits:
                self.rate_limits[api_name] = {
                    "requests": [],
                    "limit": requests_per_minute
                }
            
            rate_info = self.rate_limits[api_name]
            
            # Remove old requests (older than 1 minute)
            rate_info["requests"] = [
                req_time for req_time in rate_info["requests"]
                if current_time - req_time < 60
            ]
            
            # Check if we can make a request
            if len(rate_info["requests"]) >= requests_per_minute:
                # Calculate wait time
                oldest_request = min(rate_info["requests"])
                wait_time = 60 - (current_time - oldest_request)
                
                return {
                    "success": True,
                    "can_proceed": False,
                    "wait_time": wait_time,
                    "current_count": len(rate_info["requests"]),
                    "limit": requests_per_minute
                }
            else:
                # Record this request
                rate_info["requests"].append(current_time)
                
                return {
                    "success": True,
                    "can_proceed": True,
                    "current_count": len(rate_info["requests"]),
                    "limit": requests_per_minute
                }
        
        except Exception as e:
            logger.error(f"Error handling rate limit: {str(e)}")
            return {"error": str(e), "success": False}
    
    def _process_batch(self, requests: List[Dict[str, Any]], max_concurrent: int = 5) -> Dict[str, Any]:
        """Process batch API requests."""
        try:
            results = []
            
            # Process requests in batches
            for i in range(0, len(requests), max_concurrent):
                batch = requests[i:i + max_concurrent]
                batch_results = []
                
                for request in batch:
                    result = self._make_request(**request)
                    batch_results.append(result)
                
                results.extend(batch_results)
            
            # Calculate summary statistics
            successful = sum(1 for result in results if result.get("success"))
            failed = len(results) - successful
            
            return {
                "success": True,
                "total_requests": len(requests),
                "successful": successful,
                "failed": failed,
                "results": results,
                "timestamp": datetime.now().isoformat()
            }
        
        except Exception as e:
            logger.error(f"Error processing batch requests: {str(e)}")
            return {"error": str(e), "success": False}
    
    async def _process_batch_async(self, requests: List[Dict[str, Any]], max_concurrent: int = 5) -> Dict[str, Any]:
        """Process batch API requests asynchronously."""
        try:
            semaphore = asyncio.Semaphore(max_concurrent)
            
            async def make_request_with_semaphore(request_data):
                async with semaphore:
                    return await self._make_request_async(**request_data)
            
            # Create tasks for all requests
            tasks = [make_request_with_semaphore(request) for request in requests]
            
            # Wait for all tasks to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            processed_results = []
            for result in results:
                if isinstance(result, Exception):
                    processed_results.append({"error": str(result), "success": False})
                else:
                    processed_results.append(result)
            
            # Calculate summary statistics
            successful = sum(1 for result in processed_results if result.get("success"))
            failed = len(processed_results) - successful
            
            return {
                "success": True,
                "total_requests": len(requests),
                "successful": successful,
                "failed": failed,
                "results": processed_results,
                "timestamp": datetime.now().isoformat()
            }
        
        except Exception as e:
            logger.error(f"Error processing async batch requests: {str(e)}")
            return {"error": str(e), "success": False}
    
    # Async versions of other methods
    async def _handle_oauth_async(self, **kwargs) -> Dict[str, Any]:
        """Async version of OAuth handling."""
        return self._handle_oauth(**kwargs)
    
    async def _handle_webhook_async(self, **kwargs) -> Dict[str, Any]:
        """Async version of webhook handling."""
        return self._handle_webhook(**kwargs)
    
    async def _handle_rate_limit_async(self, **kwargs) -> Dict[str, Any]:
        """Async version of rate limiting."""
        return self._handle_rate_limit(**kwargs)
    
    def get_available_operations(self) -> List[str]:
        """Get list of available operations."""
        return ["request", "oauth", "webhook", "rate_limit", "batch"]
    
    def get_supported_auth_types(self) -> List[str]:
        """Get list of supported authentication types."""
        return ["bearer", "basic", "api_key"]
    
    def get_supported_oauth_providers(self) -> List[str]:
        """Get list of supported OAuth providers."""
        return ["google", "github", "twitter"]
    
    def add_api_key(self, service: str, key: str) -> None:
        """Add API key for a service."""
        self.api_keys[service] = key
    
    def get_rate_limit_status(self, api_name: str) -> Dict[str, Any]:
        """Get current rate limit status."""
        if api_name not in self.rate_limits:
            return {"requests": 0, "limit": 0}
        
        current_time = time.time()
        rate_info = self.rate_limits[api_name]
        
        # Count recent requests
        recent_requests = [
            req_time for req_time in rate_info["requests"]
            if current_time - req_time < 60
        ]
        
        return {
            "requests": len(recent_requests),
            "limit": rate_info["limit"],
            "remaining": rate_info["limit"] - len(recent_requests)
        }
    
    def __del__(self):
        """Clean up resources."""
        if self.session:
            try:
                asyncio.create_task(self.session.close())
            except:
                pass
