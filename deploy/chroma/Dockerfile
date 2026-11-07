# Self-hosted Chroma for the podcast corpus, behind a bearer-token proxy.
#
# Chroma 1.x ships NO authentication (measured: docs/self-hosted-chroma-behaviour.md),
# so Chroma binds to localhost only and Caddy is the sole public listener,
# rejecting any request without `Authorization: Bearer $CHROMA_TOKEN`.
#
# Pins: the server matches the repo's client (chromadb==1.5.9). Never `latest`
# or a `.dev` tag; upgrade deliberately and re-measure.
FROM caddy:2.11.9 AS caddy

FROM chromadb/chroma:1.5.9
COPY --from=caddy /usr/bin/caddy /usr/bin/caddy
COPY config.yaml /chroma-config.yaml
COPY Caddyfile /etc/caddy/Caddyfile
COPY start.sh /start.sh
RUN chmod +x /start.sh
EXPOSE 8080
ENTRYPOINT ["dumb-init", "--", "/start.sh"]
