#######
# Dev #
#######

fmt:
    @cargo fmt --all
    @clang-format -i ./crates/inference-cpp/**/*.{cpp,hpp}

#######
# Run #
#######

build:
    @docker buildx bake

# k3d deployement
up:
    @./scripts/setup-k3d.sh
    @./scripts/deploy-k3d.sh

down:
    @k3d cluster delete detr-mmap

# Local development (no k8s, CPU-only)
local-up:
    @docker compose -f docker/compose.local.yml up --build

local-down:
    @docker compose -f docker/compose.local.yml down

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

bench:
    @docker compose -f docker/compose.benchmark.yml up --build
