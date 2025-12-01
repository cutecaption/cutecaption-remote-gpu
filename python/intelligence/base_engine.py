"""
Base Engine - Foundation for all Intelligence Expert Services
All Intelligence services inherit from this to ensure standardized interface
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import logging
import time
import json
from datetime import datetime

class BaseEngine(ABC):
    """
    Base class for all Intelligence Engine expert services
    
    Provides:
    - Standardized initialization
    - Health check endpoints
    - Logging infrastructure
    - Error handling patterns
    - Request/response formatting
    """
    
    def __init__(self, service_name: str, version: str = "1.0.0"):
        """
        Initialize the base engine
        
        Args:
            service_name: Unique identifier for this service
            version: Service version for tracking
        """
        self.service_name = service_name
        self.version = version
        self.is_initialized = False
        self.initialization_time = None
        self.health_status = "initializing"
        self.error_count = 0
        self.request_count = 0
        
        # Setup logging
        self.logger = self._setup_logger()
        self.logger.info(f"[{service_name}] Creating service instance v{version}")
    
    def _setup_logger(self) -> logging.Logger:
        """Setup structured logging for the service"""
        logger = logging.getLogger(self.service_name)
        logger.setLevel(logging.INFO)
        
        # Console handler with formatting
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    @abstractmethod
    async def initialize(self) -> Dict[str, Any]:
        """
        Initialize the service (load models, setup resources, etc.)
        
        Must be implemented by child classes
        
        Returns:
            Dict with initialization status and metadata
        """
        pass
    
    @abstractmethod
    async def process(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a service-specific request
        
        Must be implemented by child classes
        
        Args:
            request: Request payload with action and parameters
            
        Returns:
            Dict with results and metadata
        """
        pass
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Standard health check endpoint
        
        Returns comprehensive health information
        """
        uptime = None
        if self.initialization_time:
            uptime = time.time() - self.initialization_time
        
        return {
            "service": self.service_name,
            "version": self.version,
            "status": self.health_status,
            "initialized": self.is_initialized,
            "uptime_seconds": uptime,
            "request_count": self.request_count,
            "error_count": self.error_count,
            "error_rate": self.error_count / max(self.request_count, 1),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main request handler with error handling and logging
        
        Wraps the process() method with:
        - Request ID tracking
        - Error handling
        - Performance monitoring
        - Structured logging
        
        Args:
            request: Incoming request
            
        Returns:
            Standardized response with results or error
        """
        request_id = request.get('request_id', f"req_{int(time.time() * 1000)}")
        start_time = time.time()
        action = request.get('action', 'unknown')
        
        self.request_count += 1
        
        # Only log non-health-check requests
        is_health_check = action == 'health_check'
        if not is_health_check:
            self.logger.info(f"[{request_id}] Received request: {action}")
        
        try:
            # Ensure service is initialized
            if not self.is_initialized:
                raise RuntimeError(f"Service {self.service_name} not initialized")
            
            # Process the request
            result = await self.process(request)
            
            elapsed = time.time() - start_time
            if not is_health_check:
                self.logger.info(f"[{request_id}] Completed in {elapsed:.3f}s")
            
            return {
                "success": True,
                "request_id": request_id,
                "service": self.service_name,
                "data": result,
                "elapsed_seconds": elapsed,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.error_count += 1
            elapsed = time.time() - start_time
            
            self.logger.error(f"[{request_id}] Error: {str(e)}", exc_info=True)
            
            return {
                "success": False,
                "request_id": request_id,
                "service": self.service_name,
                "error": {
                    "type": type(e).__name__,
                    "message": str(e),
                    "recoverable": self._is_recoverable_error(e)
                },
                "elapsed_seconds": elapsed,
                "timestamp": datetime.utcnow().isoformat()
            }
    
    def _is_recoverable_error(self, error: Exception) -> bool:
        """
        Determine if an error is recoverable
        
        Helps orchestrator decide whether to retry
        """
        # Recoverable errors (temporary issues)
        recoverable_types = (
            ConnectionError,
            TimeoutError,
            IOError
        )
        
        # Non-recoverable errors (permanent issues)
        non_recoverable_types = (
            ValueError,
            TypeError,
            RuntimeError
        )
        
        if isinstance(error, recoverable_types):
            return True
        if isinstance(error, non_recoverable_types):
            return False
        
        # Default to non-recoverable for safety
        return False
    
    async def shutdown(self) -> Dict[str, Any]:
        """
        Clean shutdown of the service
        
        Returns:
            Shutdown status and final metrics
        """
        self.logger.info(f"[{self.service_name}] Shutting down...")
        
        self.is_initialized = False
        self.health_status = "shutdown"
        
        final_metrics = {
            "service": self.service_name,
            "total_requests": self.request_count,
            "total_errors": self.error_count,
            "uptime_seconds": time.time() - self.initialization_time if self.initialization_time else 0
        }
        
        self.logger.info(f"[{self.service_name}] Shutdown complete: {final_metrics}")
        
        return final_metrics
    
    def _mark_initialized(self) -> None:
        """Helper to mark service as successfully initialized"""
        self.is_initialized = True
        self.initialization_time = time.time()
        self.health_status = "healthy"
        self.logger.info(f"[{self.service_name}] Service initialized successfully")
    
    def _mark_unhealthy(self, reason: str = "unknown") -> None:
        """Helper to mark service as unhealthy"""
        self.health_status = f"unhealthy: {reason}"
        self.logger.error(f"[{self.service_name}] Service marked unhealthy: {reason}")
    
    # Convenience logging methods for child classes
    def log_info(self, message: str) -> None:
        """Log info message"""
        self.logger.info(message)
    
    def log_error(self, message: str) -> None:
        """Log error message"""
        self.logger.error(message)
    
    def log_warning(self, message: str) -> None:
        """Log warning message"""
        self.logger.warning(message)
    
    def log_debug(self, message: str) -> None:
        """Log debug message"""
        self.logger.debug(message)


# Example usage for child classes:
"""
from base_engine import BaseEngine

class SemanticEngine(BaseEngine):
    def __init__(self):
        super().__init__("semantic_engine", "1.0.0")
        self.model = None
        self.index = None
    
    async def initialize(self):
        # Load CLIP model
        self.model = load_clip_model()
        self.index = load_faiss_index()
        self._mark_initialized()
        return {"model_loaded": True, "index_size": len(self.index)}
    
    async def process(self, request):
        action = request['action']
        
        if action == 'search':
            return await self._search(request['query'])
        elif action == 'embed':
            return await self._embed(request['image_path'])
        else:
            raise ValueError(f"Unknown action: {action}")
    
    async def _search(self, query):
        # Implementation
        pass
"""

