#!/bin/bash
# ============================================================================
# Local Deployment Script for RedactionTool
# Deploys UI and API services using Docker Compose
# ============================================================================

set -e  # Exit on error

echo "======================================================================"
echo "RedactionTool - Local Deployment"
echo "======================================================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker is not installed${NC}"
    echo "Please install Docker: https://docs.docker.com/get-docker/"
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo -e "${RED}Error: Docker Compose is not installed${NC}"
    echo "Please install Docker Compose: https://docs.docker.com/compose/install/"
    exit 1
fi

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo -e "${YELLOW}Warning: .env file not found${NC}"
    echo "Creating .env from .env.example..."
    cp .env.example .env
    echo -e "${GREEN}✓ Created .env file${NC}"
    echo ""
    echo -e "${YELLOW}Please edit .env file with your configuration before proceeding${NC}"
    read -p "Press Enter to continue or Ctrl+C to exit..."
fi

# Create necessary directories
echo "Creating directories..."
mkdir -p uploads output audit_logs policies config
echo -e "${GREEN}✓ Directories created${NC}"
echo ""

# Build images
echo "Building Docker images..."
docker-compose -f docker-compose.improved.yml build --no-cache
echo -e "${GREEN}✓ Images built successfully${NC}"
echo ""

# Start services
echo "Starting services..."
docker-compose -f docker-compose.improved.yml up -d
echo -e "${GREEN}✓ Services started${NC}"
echo ""

# Wait for services to be healthy
echo "Waiting for services to be ready..."
sleep 10

# Check service health
echo "Checking service health..."

# Check UI
if curl -f http://localhost:8501/_stcore/health &> /dev/null; then
    echo -e "${GREEN}✓ UI service is healthy${NC}"
else
    echo -e "${RED}✗ UI service is not responding${NC}"
fi

# Check API
if curl -f http://localhost:8000/health &> /dev/null; then
    echo -e "${GREEN}✓ API service is healthy${NC}"
else
    echo -e "${RED}✗ API service is not responding${NC}"
fi

# Check Redis
if docker-compose -f docker-compose.improved.yml exec -T redis redis-cli ping &> /dev/null; then
    echo -e "${GREEN}✓ Redis service is healthy${NC}"
else
    echo -e "${RED}✗ Redis service is not responding${NC}"
fi

echo ""
echo "======================================================================"
echo "Deployment Complete!"
echo "======================================================================"
echo ""
echo "Services are running:"
echo "  - UI:    http://localhost:8501"
echo "  - API:   http://localhost:8000"
echo "  - Docs:  http://localhost:8000/docs"
echo ""
echo "To view logs:"
echo "  docker-compose -f docker-compose.improved.yml logs -f"
echo ""
echo "To stop services:"
echo "  docker-compose -f docker-compose.improved.yml down"
echo ""
echo "To stop and remove volumes:"
echo "  docker-compose -f docker-compose.improved.yml down -v"
echo ""
