.PHONY: build serve

build:
	cargo build --manifest-path rust/wasm/Cargo.toml --target wasm32-unknown-unknown --release
	cp rust/wasm/target/wasm32-unknown-unknown/release/web_inference_wasm.wasm web/inference.wasm

serve: build
	python3 -m http.server 5173 --bind 127.0.0.1 --directory web
