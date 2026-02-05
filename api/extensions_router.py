"""
Extensions API Router

REST API endpoints for managing plugins, language packs, and LLM providers.
"""

import os
import sys
import json
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

from fastapi import APIRouter, HTTPException, UploadFile, File, Depends, Header
from pydantic import BaseModel, Field

# Add app to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.extensions.registry.plugin_registry import get_plugin_registry
from app.extensions.registry.language_registry import get_language_registry
from app.extensions.registry.llm_registry import get_llm_registry
from app.services.pii.enhanced_ensemble_detector import EnhancedEnsembleDetector

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/extensions", tags=["extensions"])

# Get registries
plugin_registry = get_plugin_registry()
language_registry = get_language_registry()
llm_registry = get_llm_registry()


# ============================================================================
# Request/Response Models
# ============================================================================

class PluginMetadataResponse(BaseModel):
    """Plugin metadata response"""
    name: str
    version: str
    description: str
    author: str
    supported_entity_types: List[str]
    supported_languages: List[str]
    priority: int
    enabled: bool


class LanguagePackMetadataResponse(BaseModel):
    """Language pack metadata response"""
    language_code: str
    language_name: str
    script: str
    supported_entity_types: List[str]
    description: str


class LLMProviderMetadataResponse(BaseModel):
    """LLM provider metadata response"""
    name: str
    model_name: str
    enabled: bool
    is_default: bool
    local: bool
    cost_per_1k_tokens: float
    rate_limit: int


class DiscoveryRequest(BaseModel):
    """Discovery request"""
    plugins_dir: Optional[str] = "plugins/detectors"
    languages_dir: Optional[str] = "plugins/languages"


class DiscoveryResponse(BaseModel):
    """Discovery response"""
    plugins_discovered: List[str]
    languages_discovered: List[str]
    total_discovered: int


class LLMConfigureRequest(BaseModel):
    """LLM provider configuration request"""
    provider: str = Field(..., description="Provider name (openai, anthropic, ollama)")
    api_key: Optional[str] = None
    model: Optional[str] = None
    set_as_default: bool = False


class LLMDetectRequest(BaseModel):
    """LLM-based detection request"""
    text: str
    language: str = "en"
    entity_types: Optional[List[str]] = None
    provider: Optional[str] = None


class ExtensionStatsResponse(BaseModel):
    """Extension statistics response"""
    total_plugins: int
    enabled_plugins: int
    total_language_packs: int
    total_llm_providers: int
    enabled_llm_providers: int


# ============================================================================
# Plugin Management Endpoints
# ============================================================================

@router.get("/plugins", response_model=List[PluginMetadataResponse])
async def list_plugins():
    """
    List all registered plugins.

    Returns:
        List of plugin metadata
    """
    try:
        stats = plugin_registry.get_stats()
        plugins = []

        for plugin_info in stats.get("plugins", []):
            plugins.append(PluginMetadataResponse(
                name=plugin_info["name"],
                version=plugin_info["version"],
                description="",  # Add if available
                author="",  # Add if available
                supported_entity_types=plugin_info["entity_types"],
                supported_languages=plugin_info["languages"],
                priority=plugin_info["priority"],
                enabled=plugin_info["enabled"]
            ))

        return plugins

    except Exception as e:
        logger.error(f"Failed to list plugins: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/plugins/{plugin_name}")
async def get_plugin(plugin_name: str):
    """
    Get plugin details.

    Args:
        plugin_name: Name of plugin

    Returns:
        Plugin metadata
    """
    try:
        metadata = plugin_registry.get_metadata(plugin_name)
        if not metadata:
            raise HTTPException(status_code=404, detail=f"Plugin not found: {plugin_name}")

        return {
            "name": metadata.name,
            "version": metadata.version,
            "description": metadata.description,
            "author": metadata.author,
            "plugin_type": metadata.plugin_type.value,
            "supported_entity_types": metadata.supported_entity_types,
            "supported_languages": metadata.supported_languages,
            "priority": metadata.priority,
            "dependencies": metadata.dependencies,
            "enabled": plugin_registry.is_enabled(plugin_name),
            "requires_network": metadata.requires_network,
            "timeout_seconds": metadata.timeout_seconds
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get plugin {plugin_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/plugins/{plugin_name}/enable")
async def enable_plugin(plugin_name: str):
    """
    Enable a plugin.

    Args:
        plugin_name: Name of plugin to enable

    Returns:
        Success message
    """
    try:
        plugin_registry.enable_plugin(plugin_name)
        return {"message": f"Plugin {plugin_name} enabled successfully"}

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to enable plugin {plugin_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/plugins/{plugin_name}/disable")
async def disable_plugin(plugin_name: str):
    """
    Disable a plugin.

    Args:
        plugin_name: Name of plugin to disable

    Returns:
        Success message
    """
    try:
        plugin_registry.disable_plugin(plugin_name)
        return {"message": f"Plugin {plugin_name} disabled successfully"}

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to disable plugin {plugin_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/plugins/{plugin_name}")
async def unregister_plugin(plugin_name: str):
    """
    Unregister a plugin.

    Args:
        plugin_name: Name of plugin to unregister

    Returns:
        Success message
    """
    try:
        plugin_registry.unregister(plugin_name)
        return {"message": f"Plugin {plugin_name} unregistered successfully"}

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to unregister plugin {plugin_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Language Pack Endpoints
# ============================================================================

@router.get("/languages", response_model=List[LanguagePackMetadataResponse])
async def list_languages():
    """
    List all registered language packs.

    Returns:
        List of language pack metadata
    """
    try:
        stats = language_registry.get_stats()
        packs = []

        for pack_info in stats.get("language_packs", []):
            packs.append(LanguagePackMetadataResponse(
                language_code=pack_info["code"],
                language_name=pack_info["name"],
                script=pack_info["script"],
                supported_entity_types=pack_info["entity_types"],
                description=""
            ))

        return packs

    except Exception as e:
        logger.error(f"Failed to list language packs: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/languages/{language_code}")
async def get_language_pack(language_code: str):
    """
    Get language pack details.

    Args:
        language_code: Language code (e.g., 'fr', 'de', 'ar')

    Returns:
        Language pack metadata and patterns
    """
    try:
        metadata = language_registry.get_metadata(language_code)
        if not metadata:
            raise HTTPException(
                status_code=404,
                detail=f"Language pack not found: {language_code}"
            )

        patterns = language_registry.get_regex_patterns(language_code)
        policy = language_registry.get_redaction_policy(language_code)

        return {
            "language_code": metadata.language_code,
            "language_name": metadata.language_name,
            "script": metadata.script.value,
            "supported_entity_types": metadata.supported_entity_types,
            "description": metadata.description,
            "patterns": patterns,
            "redaction_policy": {
                "full_redaction": policy.full_redaction if policy else [],
                "partial_redaction": policy.partial_redaction if policy else {},
                "preserve_format": policy.preserve_format if policy else []
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get language pack {language_code}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# LLM Provider Endpoints
# ============================================================================

@router.get("/llm/providers", response_model=List[LLMProviderMetadataResponse])
async def list_llm_providers():
    """
    List all registered LLM providers.

    Returns:
        List of LLM provider metadata
    """
    try:
        stats = llm_registry.get_stats()
        providers = []

        for provider_info in stats.get("providers", []):
            providers.append(LLMProviderMetadataResponse(
                name=provider_info["name"],
                model_name=provider_info["model"],
                enabled=provider_info["enabled"],
                is_default=provider_info["is_default"],
                local=provider_info["local"],
                cost_per_1k_tokens=provider_info["cost_per_1k_tokens"],
                rate_limit=provider_info["rate_limit"]
            ))

        return providers

    except Exception as e:
        logger.error(f"Failed to list LLM providers: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/llm/configure")
async def configure_llm_provider(config: LLMConfigureRequest):
    """
    Configure an LLM provider.

    Args:
        config: Provider configuration

    Returns:
        Success message
    """
    try:
        # Import provider
        if config.provider == "openai":
            from app.extensions.llm_providers.openai_provider import create_provider
            provider = create_provider(
                api_key=config.api_key,
                model=config.model or "gpt-4"
            )
        elif config.provider == "anthropic":
            from app.extensions.llm_providers.anthropic_provider import create_provider
            provider = create_provider(
                api_key=config.api_key,
                model=config.model or "claude-3-sonnet-20240229"
            )
        elif config.provider == "ollama":
            from app.extensions.llm_providers.ollama_provider import create_provider
            provider = create_provider(
                model=config.model or "llama2:13b"
            )
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown provider: {config.provider}"
            )

        # Register provider
        llm_registry.register(
            provider,
            auto_enable=True,
            set_as_default=config.set_as_default
        )

        return {
            "message": f"LLM provider {config.provider} configured successfully",
            "provider": config.provider,
            "model": config.model
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to configure LLM provider: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/llm/detect")
async def detect_with_llm(request: LLMDetectRequest):
    """
    Detect entities using LLM.

    Args:
        request: Detection request

    Returns:
        Detected entities
    """
    try:
        # Get provider
        provider = llm_registry.get_provider(request.provider)
        if not provider:
            raise HTTPException(
                status_code=404,
                detail="No LLM provider available"
            )

        # Check rate limit
        if not llm_registry.check_rate_limit(request.provider):
            wait_time = llm_registry.get_wait_time(request.provider)
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded. Wait {wait_time:.1f}s"
            )

        # Detect entities
        entities = provider.detect_entities(
            text=request.text,
            language=request.language,
            entity_types=request.entity_types
        )

        # Convert to response format
        return {
            "entities": [entity.to_dict() for entity in entities],
            "total_entities": len(entities),
            "provider": request.provider or "default"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"LLM detection failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Discovery and Stats
# ============================================================================

@router.post("/discover", response_model=DiscoveryResponse)
async def discover_extensions(request: DiscoveryRequest):
    """
    Discover all extensions (plugins and language packs).

    Args:
        request: Discovery request with directory paths

    Returns:
        Discovery results
    """
    try:
        # Discover plugins
        plugins = plugin_registry.discover_plugins(request.plugins_dir)

        # Discover language packs
        languages = language_registry.discover_language_packs(request.languages_dir)

        return DiscoveryResponse(
            plugins_discovered=plugins,
            languages_discovered=languages,
            total_discovered=len(plugins) + len(languages)
        )

    except Exception as e:
        logger.error(f"Discovery failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats", response_model=ExtensionStatsResponse)
async def get_extension_stats():
    """
    Get extension statistics.

    Returns:
        Extension statistics
    """
    try:
        plugin_stats = plugin_registry.get_stats()
        language_stats = language_registry.get_stats()
        llm_stats = llm_registry.get_stats()

        return ExtensionStatsResponse(
            total_plugins=plugin_stats["total_plugins"],
            enabled_plugins=plugin_stats["enabled_plugins"],
            total_language_packs=language_stats["total_packs"],
            total_llm_providers=llm_stats["total_providers"],
            enabled_llm_providers=llm_stats["total_providers"]  # Simplified
        )

    except Exception as e:
        logger.error(f"Failed to get stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    """
    Health check for extensions system.

    Returns:
        Health status
    """
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "plugins": plugin_registry.get_stats()["total_plugins"],
        "languages": language_registry.get_stats()["total_packs"],
        "llm_providers": llm_registry.get_stats()["total_providers"]
    }
