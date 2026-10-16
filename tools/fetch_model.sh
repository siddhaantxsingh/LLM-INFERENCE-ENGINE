#!/usr/bin/env bash
# Fetch a pretrained GGUF model from Docker Hub's `ai/` namespace (OCI
# artifacts) into a shared cache, verify its pinned sha256, and record its
# licence. huggingface.co is blocked on the build machine; Docker Hub is not.
#
#   tools/fetch_model.sh smollm2-135m-instruct-f16    # SmolLM2-135M-Instruct, F16, 271 MB
#   tools/fetch_model.sh smollm2-135m-instruct-q8_0   # same model, Q8_0, 145 MB
#   tools/fetch_model.sh qwen2.5-0.5b-f16      # Qwen2.5-0.5B-Instruct, F16, 994 MB
#
# Output: $MODELS_DIR/<name>.gguf (default MODELS_DIR=/home/user/models),
#         <name>.LICENSE and <name>.json (source, digest, licence).
# The model files are never committed to git.
set -euo pipefail

name=${1:?usage: fetch_model.sh <smollm2-135m-instruct-f16|smollm2-135m-instruct-q8_0|qwen2.5-0.5b-f16>}
dir=${MODELS_DIR:-/home/user/models}

# name -> repository, tag, pinned GGUF layer digest (= sha256 of the file), licence id
case "$name" in
  # (the 135m-* tags hold unsloth/SmolLM2-135M-Instruct-GGUF files, per the manifest annotations)
  smollm2-135m-instruct-f16)  repo=ai/smollm2; tag=135m-f16;  digest=5157ca60744d21631818364854ac8e4452e1b8022d2ab4c8a2f9cda2344afb30; lic=Apache-2.0 ;;
  smollm2-135m-instruct-q8_0) repo=ai/smollm2; tag=135m-q8_0; digest=c4a3dd037301b6ecea31d6da37f5cd793ead920dd5ddfe6d589294628d6ce66a; lic=Apache-2.0 ;;
  qwen2.5-0.5b-f16)           repo=ai/qwen2.5; tag=0.5B-F16;  digest=f1ad9d1174ce6ab47b584d522634c47e411b75bffdffd9a4e106e21e882392e5; lic=Apache-2.0 ;;
  *) echo "unknown model $name" >&2; exit 2 ;;
esac

mkdir -p "$dir"
out="$dir/$name.gguf"

# one download at a time per model (other jobs may share the cache)
exec 9>"$dir/.$name.lock"
flock 9
have=0
if [ -f "$out" ] && echo "$digest  $out" | sha256sum -c --status; then
  echo "$out: already present, sha256 ok"; have=1
  [ -f "$dir/$name.json" ] && exit 0
fi

reg=https://registry-1.docker.io/v2/$repo
token=$(curl -fsS "https://auth.docker.io/token?service=registry.docker.io&scope=repository:$repo:pull" |
  python3 -c 'import json,sys; print(json.load(sys.stdin)["token"])')
auth=(-H "Authorization: Bearer $token")
manifest=$(curl -fsS "${auth[@]}" -H "Accept: application/vnd.oci.image.manifest.v1+json" "$reg/manifests/$tag")
# the manifest must still list the pinned layer (tags can move; the digest cannot)
if ! grep -q "sha256:$digest" <<<"$manifest"; then
  echo "manifest for $repo:$tag no longer lists sha256:$digest; update the pin after review" >&2
  exit 1
fi

if [ $have = 0 ]; then
  tmp=$(mktemp "$dir/.$name.XXXXXX")
  trap 'rm -f "$tmp"' EXIT
  echo "downloading $repo:$tag (sha256:$digest) ..."
  curl -fsSL --retry 4 "${auth[@]}" -o "$tmp" "$reg/blobs/sha256:$digest"
  echo "$digest  $tmp" | sha256sum -c --status || { echo "sha256 mismatch" >&2; exit 1; }
  chmod 644 "$tmp"
  mv "$tmp" "$out"
fi

# licence: the license layer if the manifest has one, otherwise the config's licence list
lic_digest=$(python3 -c '
import json,sys
m=json.loads(sys.argv[1])
print(next((l["digest"] for l in m["layers"] if l["mediaType"].endswith(".license")), ""))' "$manifest")
if [ -n "$lic_digest" ]; then
  curl -fsSL "${auth[@]}" -o "$dir/$name.LICENSE" "$reg/blobs/$lic_digest"
else
  echo "$lic (from the artifact config; no licence file in the artifact)" > "$dir/$name.LICENSE"
fi
python3 - "$dir/$name.json" "$repo:$tag" "$digest" "$lic" "$(stat -c %s "$out")" <<'PY'
import json, sys, datetime
p, src, dg, lic, size = sys.argv[1:]
json.dump({"source": "docker.io/" + src, "sha256": dg, "license": lic, "bytes": int(size),
           "fetched": datetime.datetime.now(datetime.timezone.utc).isoformat()}, open(p, "w"), indent=1)
PY
echo "$out: ok ($(stat -c %s "$out") bytes, licence $lic)"
