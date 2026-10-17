#!/usr/bin/env bash
# clang-format for src/, pinned so every machine (and CI) formats the same way.
#
#   tools/format.sh [--check] [files...]
#
# Without files: every tracked C++ file under src/. Files outside src/ or not C++ are skipped, so a
# hook can pass whatever was edited. --check changes nothing and exits 1 if a file needs formatting.
# Runs clang-format through uvx (no system install needed): CLANG_FORMAT overrides the command.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VERSION=23.1.3
CHECK=0
FILES=()
for a in "$@"; do
    case "$a" in
        --check) CHECK=1 ;;
        -*) echo "usage: $0 [--check] [files...]"; exit 1 ;;
        *) FILES+=("$a") ;;
    esac
done

cd "$ROOT" || exit 1
if [[ ${#FILES[@]} -eq 0 ]]; then
    mapfile -t FILES < <(git ls-files 'src/*.cpp' 'src/*.hpp' 'src/*.h' | grep -v '^src/gen/')
fi

TARGETS=()
for f in "${FILES[@]}"; do
    [[ -f $f ]] || continue
    abs="$(realpath "$f")"
    # Relative to the file's own checkout: a hook may pass a file from another worktree.
    top="$(git -C "$(dirname "$abs")" rev-parse --show-toplevel 2>/dev/null)" || continue
    rel="${abs#"$top"/}"
    [[ $rel == src/* && $rel != src/gen/* && $rel =~ \.(cpp|hpp|h)$ ]] && TARGETS+=("$abs")
done
[[ ${#TARGETS[@]} -eq 0 ]] && exit 0

read -r -a CF <<< "${CLANG_FORMAT:-uvx --quiet clang-format==$VERSION}"
if [[ $CHECK == 0 ]]; then
    exec "${CF[@]}" -i "${TARGETS[@]}"
fi
OUT="$("${CF[@]}" --dry-run --Werror "${TARGETS[@]}" 2>&1)" && exit 0
COUNTS="$(grep -E '^[^ ]+:[0-9]+:[0-9]+: error' <<< "$OUT" | cut -d: -f1 | sort | uniq -c)"
if [[ -z $COUNTS ]]; then
    echo "$OUT"
    echo ">> clang-format failed to run"
    exit 2
fi
awk '{ print ">> " $2 ": " $1 " places to format" }' <<< "$COUNTS"
echo ">> run tools/format.sh"
exit 1
