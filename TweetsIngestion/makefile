test_flask:
	curl -X POST http://localhost:5000/send_event \
		-H "Content-Type: application/json" \
		-d '{"user_id": 1, "event_type": "post", "event_timestamp": "Sun, 31 Dec 1899 00:00:00 GMT", "event_params": {"post_id": 14, "content": "This is a test"}}'
