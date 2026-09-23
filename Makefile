.PHONY: help install verify sweep smoke analyze

help:
	@echo "install   Install the locked Node dependencies"
	@echo "verify    Type-check and run offline tests"
	@echo "sweep     Run all seven judges"
	@echo "smoke     Run one repetition with Jev native and GPT-5 Nano"
	@echo "analyze   Analyze RUN_ID from Phoenix"

install:
	corepack enable
	pnpm install --frozen-lockfile

verify:
	pnpm run verify

sweep:
	pnpm run sweep

smoke:
	SWEEP_REPETITIONS=1 pnpm run sweep -- jev-native openai-cheap

analyze:
	@test -n "$(RUN_ID)" || (echo "RUN_ID is required" >&2; exit 1)
	pnpm run analyze "$(RUN_ID)"
