"""Data plan: manifests, splits, fingerprints and the Stage 0 cache.

Raw images stay in their download form under VENUS_DATA_ROOT. Manifests
(CSV + SHA-256) are committed under backend/data/manifests; the Stage 0 cache
(512x512 FOV-normalised JPEGs) lives under VENUS_CACHE_DIR and is rebuilt by
`python -m backend.data.cache_stage0`.
"""
