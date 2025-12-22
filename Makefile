.PHONY: build up

build:
	@./scripts/build-images.sh

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
