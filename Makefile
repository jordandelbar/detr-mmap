.PHONY: build up

build:
	@./scripts/build-images.sh

up:
	@./scripts/setup-kind.sh
	@./scripts/deploy-kind.sh

down:
	@kind delete cluster -n kind
