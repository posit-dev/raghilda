#!/bin/bash
# Launches a PostgreSQL + pgvector container for interactive testing.
# Usage: bash scripts/postgres-dev.sh

CONTAINER_NAME="raghilda-postgres"
POSTGRES_PASSWORD="raghilda"
POSTGRES_USER="raghilda"
POSTGRES_DB="raghilda"
PORT=5432

# Stop and remove existing container if running
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "Removing existing container '${CONTAINER_NAME}'..."
    docker rm -f "${CONTAINER_NAME}" > /dev/null
fi

echo "Starting PostgreSQL with pgvector on port ${PORT}..."
docker run -d \
    --name "${CONTAINER_NAME}" \
    -e POSTGRES_PASSWORD="${POSTGRES_PASSWORD}" \
    -e POSTGRES_USER="${POSTGRES_USER}" \
    -e POSTGRES_DB="${POSTGRES_DB}" \
    -p "${PORT}:5432" \
    pgvector/pgvector:pg17

echo ""
echo "PostgreSQL is starting up..."
echo "Connection string: postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@localhost:${PORT}/${POSTGRES_DB}"
echo ""
echo "To connect with psycopg:"
echo "  import psycopg"
echo "  con = psycopg.connect('postgresql://raghilda:raghilda@localhost:5432/raghilda')"
echo ""
echo "To stop: docker rm -f ${CONTAINER_NAME}"
