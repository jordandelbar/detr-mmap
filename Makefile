.PHONY: build up down open-webpage coverage

#######
# Run #
#######
build:
	@docker compose build

up:
	@./scripts/setup-kind.sh
	@./scripts/deploy-kind.sh

down:
	@kind delete cluster -n kind

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
