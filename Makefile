.PHONY: build up down open-webpage coverage

#######
# Run #
#######
build:
	@docker compose build

up:
	@./scripts/setup-k3d.sh
	@./scripts/deploy-k3d.sh

down:
	@k3d cluster delete bridge-rt

open-webpage:
	@if command -v xdg-open > /dev/null; then xdg-open index.html; \
	elif command -v open > /dev/null; then open index.html; \
	elif command -v start > /dev/null; then start index.html; \
	else echo "No suitable command found to open the file."; fi

########
# Test #
########
test:
	cargo test --workspace

coverage:
	cargo llvm-cov --workspace --lib --tests
