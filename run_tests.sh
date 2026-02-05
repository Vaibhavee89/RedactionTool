#!/bin/bash
#
# Test Runner Script for PII Redaction Tool
#
# Usage:
#   ./run_tests.sh              # Run all tests
#   ./run_tests.sh unit         # Run only unit tests
#   ./run_tests.sh integration  # Run only integration tests
#   ./run_tests.sh coverage     # Run tests with coverage

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Print header
echo -e "${GREEN}================================${NC}"
echo -e "${GREEN}PII Redaction Tool - Test Runner${NC}"
echo -e "${GREEN}================================${NC}"
echo ""

# Check if pytest is installed
if ! command -v pytest &> /dev/null; then
    echo -e "${RED}Error: pytest not found. Install it with: pip install pytest${NC}"
    exit 1
fi

# Set test mode
TEST_MODE="${1:-all}"

case "$TEST_MODE" in
    unit)
        echo -e "${YELLOW}Running unit tests only...${NC}"
        pytest tests/unit/ -v --tb=short
        ;;

    integration)
        echo -e "${YELLOW}Running integration tests only...${NC}"
        pytest tests/integration/ -v --tb=short
        ;;

    coverage)
        echo -e "${YELLOW}Running tests with coverage...${NC}"

        # Check if pytest-cov is installed
        if ! pip show pytest-cov &> /dev/null; then
            echo -e "${YELLOW}Installing pytest-cov...${NC}"
            pip install pytest-cov
        fi

        # Run tests with coverage
        pytest tests/ \
            --cov=app \
            --cov-report=html \
            --cov-report=term-missing \
            --cov-report=xml \
            -v

        echo ""
        echo -e "${GREEN}Coverage report generated:${NC}"
        echo "  - HTML: htmlcov/index.html"
        echo "  - XML: coverage.xml"
        ;;

    quick)
        echo -e "${YELLOW}Running quick test suite (parallel)...${NC}"

        # Check if pytest-xdist is installed
        if ! pip show pytest-xdist &> /dev/null; then
            echo -e "${YELLOW}Installing pytest-xdist...${NC}"
            pip install pytest-xdist
        fi

        pytest tests/unit/ -n auto --tb=line --maxfail=5
        ;;

    all)
        echo -e "${YELLOW}Running all tests...${NC}"

        # Run unit tests
        echo ""
        echo -e "${GREEN}=== Unit Tests ===${NC}"
        pytest tests/unit/ -v --tb=short

        # Run integration tests
        echo ""
        echo -e "${GREEN}=== Integration Tests ===${NC}"
        pytest tests/integration/ -v --tb=short

        # Run evaluation tests
        echo ""
        echo -e "${GREEN}=== Evaluation Tests ===${NC}"
        pytest test_evaluation.py -v --tb=short
        ;;

    ci)
        echo -e "${YELLOW}Running CI test suite...${NC}"

        # Install required packages
        pip install -q pytest pytest-cov pytest-xdist pytest-timeout

        # Run all tests with coverage and timeouts
        pytest tests/ \
            --cov=app \
            --cov-report=xml \
            --cov-report=term \
            -n auto \
            --tb=short \
            --timeout=300 \
            -v
        ;;

    *)
        echo -e "${RED}Error: Unknown test mode '$TEST_MODE'${NC}"
        echo ""
        echo "Usage:"
        echo "  ./run_tests.sh [unit|integration|coverage|quick|all|ci]"
        echo ""
        echo "Modes:"
        echo "  unit        - Run only unit tests"
        echo "  integration - Run only integration tests"
        echo "  coverage    - Run tests with coverage report"
        echo "  quick       - Run tests in parallel (fast)"
        echo "  all         - Run all tests (default)"
        echo "  ci          - Run tests for CI/CD (with coverage and timeouts)"
        exit 1
        ;;
esac

# Check exit code
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✅ Tests completed successfully!${NC}"
    exit 0
else
    echo ""
    echo -e "${RED}❌ Tests failed!${NC}"
    exit 1
fi
