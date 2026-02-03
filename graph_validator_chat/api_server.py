import os

from graph_validator_chat.server import (
    GraphValidatorHandler,
    ThreadedHTTPServer,
    _load_persisted_validator,
    initialize_validator,
)


def main() -> None:
    host = (os.getenv("API_HOST") or "0.0.0.0").strip() or "0.0.0.0"
    port = int(os.getenv("API_PORT") or "5000")

    persisted = _load_persisted_validator()
    if persisted:
        initialize_validator(
            graph=persisted.get("graph"),
            triples=persisted.get("triples"),
            id_to_name=persisted.get("id_to_name"),
            sentence_split=persisted.get("sentence_split"),
        )
        print("[API] Loaded persisted validator state")

    server = ThreadedHTTPServer((host, port), GraphValidatorHandler)
    print(f"[API] Server running on http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
