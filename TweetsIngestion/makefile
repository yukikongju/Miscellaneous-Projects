test_flask:
	curl -X POST http://localhost:5000/send_event \
		-H "Content-Type: application/json" \
		-d '{"user_id": 1, "event_type": "post", "event_timestamp": "2025-06-27T12:34:56Z", "event_params": {"post_id": 14, "content": "This is a test"}}'

docker_compose_up:
	docker-compose up -d

