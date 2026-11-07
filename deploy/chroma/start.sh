#!/bin/sh
# Run Chroma (localhost only) and Caddy (the public listener) together, and
# exit as soon as either dies so Fly restarts the machine rather than leaving
# a proxy in front of a dead database, or a database with no proxy.
set -eu

# An empty token would make Caddy's matcher accept "Authorization: Bearer ".
if [ -z "${CHROMA_TOKEN:-}" ] || [ "${#CHROMA_TOKEN}" -lt 32 ]; then
	echo "start.sh: CHROMA_TOKEN must be set and at least 32 characters" >&2
	exit 1
fi

chroma run /chroma-config.yaml &
chroma_pid=$!
caddy run --config /etc/caddy/Caddyfile --adapter caddyfile &
caddy_pid=$!

while kill -0 "$chroma_pid" 2>/dev/null && kill -0 "$caddy_pid" 2>/dev/null; do
	sleep 2
done
echo "start.sh: a process exited (chroma alive: $(kill -0 "$chroma_pid" 2>/dev/null && echo yes || echo no))" >&2
kill "$chroma_pid" "$caddy_pid" 2>/dev/null || true
exit 1
