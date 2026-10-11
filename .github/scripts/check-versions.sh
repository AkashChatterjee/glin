#!/usr/bin/env bash
# Usage: check-versions.sh <tag>. Fails if a release file does not carry the
# version of the tag (without its leading "v").
set -euo pipefail

VERSION="${1#v}"
status=0

check() {
  local file="$1" query="$2" found
  found="$(jq -r "$query" "$file")"
  if [ "$found" != "$VERSION" ]; then
    echo "::error file=${file}::${file} ${query} is '${found}' but the tag version is '${VERSION}'. Set it to ${VERSION}, merge to main and tag again."
    status=1
  fi
}

check server.json '.version'
check server.json '.packages[0].version'
check plugins/glin/.claude-plugin/plugin.json '.version'

exit "$status"
