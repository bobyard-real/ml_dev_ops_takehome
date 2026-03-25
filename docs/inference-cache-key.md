# Inference Cache Key Trade-offs

The distributed inference cache keys results by SHA256 hash of the raw uploaded image bytes,
**before** any format normalization (EXIF transpose, RGB conversion, resize).

## Why pre-normalization

Chosen because the real traffic shape is unknown. Hashing raw bytes is:
- Cheap — one pass over the bytes, no decode/re-encode
- Simple — no dependency on Pillow internals or normalization logic
- Safe — identical raw files always produce the same key

## Trade-offs vs post-normalization

| | Pre-normalization (current) | Post-normalization |
|---|---|---|
| **Cache hit rate** | Lower — two semantically identical images (e.g. same photo, different EXIF rotation tag) get different keys | Higher — EXIF-rotated or format-converted versions of the same image share a key |
| **Hashing cost** | Negligible — raw bytes, no decode | Higher — requires full decode + normalize before hashing |
| **Complexity** | Low | Higher — normalization must happen twice (hash + inference), or be split into a separate step |
| **Correctness risk** | Low | Higher — normalization bugs could cause incorrect cache hits |

## When to revisit

If traffic analysis shows a significant share of requests are semantically duplicate images
with differing metadata (e.g. same photo re-uploaded with different EXIF), switching to
post-normalization hashing would improve hit rate. The change is isolated to the `classify()`
method — swap `image_bytes` for a hash of the normalized numpy array.
