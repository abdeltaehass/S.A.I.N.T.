.PHONY: install train start stop api dashboard replay test

PYTHON := .venv/bin/python
PIP    := .venv/bin/pip

# ── Setup ────────────────────────────────────────────────────────────────────
install:
	python3 -m venv .venv
	$(PIP) install -r requirements.txt
	cp -n .env.example .env || true

# ── Training ─────────────────────────────────────────────────────────────────
train:
	$(PYTHON) scripts/train.py

# ── Local dev (3 separate processes) ────────────────────────────────────────
redis-start:
	redis-server --daemonize yes --logfile /tmp/saint-redis.log

api:
	$(PYTHON) api/routes.py

dashboard:
	$(PYTHON) dashboard/app.py

start: redis-start
	@echo "Starting API and dashboard..."
	@$(PYTHON) api/routes.py &
	@sleep 1
	@$(PYTHON) dashboard/app.py

stop:
	@pkill -f "python.*routes.py"   2>/dev/null || true
	@pkill -f "python.*app.py"      2>/dev/null || true
	@pkill -f "python.*replay.py"   2>/dev/null || true
	@redis-cli shutdown              2>/dev/null || true
	@echo "All processes stopped."

# ── Replay test traffic ──────────────────────────────────────────────────────
replay:
	$(PYTHON) scripts/replay.py --n 200

replay-all:
	$(PYTHON) scripts/replay.py --n 0

# ── Docker ───────────────────────────────────────────────────────────────────
docker-up:
	docker compose up --build

docker-down:
	docker compose down

# ── Help ─────────────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "  make install      — create venv and install dependencies"
	@echo "  make train        — download NSL-KDD and train the model"
	@echo "  make start        — start Redis + API + dashboard locally"
	@echo "  make stop         — kill all running processes"
	@echo "  make replay       — send 200 test samples through the API"
	@echo "  make replay-all   — send all 22k test samples"
	@echo "  make docker-up    — start full stack via Docker Compose"
	@echo "  make docker-down  — tear down Docker stack"
	@echo ""
