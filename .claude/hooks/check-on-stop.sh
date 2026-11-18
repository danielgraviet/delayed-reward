#!/bin/sh
# Stop hook: run `make check` when Python/tooling files changed; block the stop
# and feed the failure back to Claude. After one blocked retry it lets Claude
# stop and warns the user instead, so a stubborn failure can't loop forever.

cd "${CLAUDE_PROJECT_DIR:-.}" || exit 0
input=$(cat)

changed=$(git status --porcelain -- '*.py' pyproject.toml Makefile uv.lock 2>/dev/null)
[ -z "$changed" ] && exit 0

if output=$(make check 2>&1); then
  exit 0
fi
tail=$(printf '%s\n' "$output" | tail -n 60)

if [ "$(printf '%s' "$input" | jq -r '.stop_hook_active // false')" = "true" ]; then
  jq -n '{systemMessage: "make check is still failing after a retry; see `make check` output."}'
  exit 0
fi

printf 'make check failed. Fix the code (not the rule), then finish.\n\n%s\n' "$tail" >&2
exit 2
