# Start Neo4j container with persistent storage using docker compose
run:
	docker compose up -d

# Stop Neo4j container
stop:
	docker compose down

# Stop Neo4j container and remove volumes
clean:
	docker compose down -v

# View Neo4j logs
logs:
	docker compose logs -f neo4j
