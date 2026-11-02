#!/bin/bash
# Add dopploc-gui to the desktop app launcher for this user (no root needed):
#   ~/.local/share/applications/io.github.digitalis_p.dopploc.desktop
#   ~/.local/share/icons/hicolor/scalable/apps/io.github.digitalis_p.dopploc.svg
# The launcher runs this checkout's .venv/bin/dopploc-gui with the checkout as working directory.
#   scripts/install-desktop.sh             install (refuses to overwrite existing files)
#   scripts/install-desktop.sh --force     install, replacing earlier copies
#   scripts/install-desktop.sh --uninstall remove exactly those two files
set -euo pipefail
id=io.github.digitalis_p.dopploc
root=$(cd "$(dirname "$0")/.." && pwd)
share="${XDG_DATA_HOME:-$HOME/.local/share}"
desktop="$share/applications/$id.desktop"
icon="$share/icons/hicolor/scalable/apps/$id.svg"
force=0
case "${1:-}" in
  "") ;;
  --force) force=1 ;;
  --uninstall)
    rm -v -f -- "$desktop" "$icon"
    command -v update-desktop-database >/dev/null && update-desktop-database -q "$share/applications" || true
    exit 0 ;;
  *) echo "usage: $0 [--force | --uninstall]" >&2; exit 2 ;;
esac

exe="$root/.venv/bin/dopploc-gui"
[ -x "$exe" ] || { echo "missing $exe -- install first: uv pip install -e '.[gui]'" >&2; exit 1; }
for f in "$desktop" "$icon"; do
  if [ -e "$f" ] && [ "$force" = 0 ]; then
    echo "exists: $f (use --force to replace)" >&2
    exit 1
  fi
done

mkdir -p "$(dirname "$desktop")" "$(dirname "$icon")"
install -m 644 "$root/share/icons/$id.svg" "$icon"
tmp=$(mktemp)
cat > "$tmp" <<EOF
[Desktop Entry]
Type=Application
Name=Doppler Locate
GenericName=Audio geolocation
Comment=Find where a recording was made from aircraft Doppler curves and ADS-B tracks
Exec=$exe %f
Path=$root
Icon=$id
Terminal=false
Categories=Science;Geography;
Keywords=doppler;adsb;geolocation;spectrogram;aircraft;dopploc;
StartupNotify=true
StartupWMClass=$id
EOF
install -m 644 "$tmp" "$desktop"
rm -f "$tmp"

command -v desktop-file-validate >/dev/null && desktop-file-validate "$desktop"
command -v update-desktop-database >/dev/null && update-desktop-database -q "$share/applications" || true
# refresh the user icon cache only if one exists (without a cache GTK scans the directory)
cache="$share/icons/hicolor/icon-theme.cache"
if [ -f "$cache" ] && command -v gtk-update-icon-cache >/dev/null; then
  gtk-update-icon-cache -q -f -t "$share/icons/hicolor"
fi
# Running icon themes (GNOME Shell's included) rescan only when the theme's top directory
# changes, not when a file lands in scalable/apps: bump its mtime so the new icon is picked up.
touch "$share/icons/hicolor"
echo "installed: $desktop"
echo "installed: $icon"
