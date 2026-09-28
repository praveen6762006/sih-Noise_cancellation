#!/usr/bin/env bash
# Download the corpora used for dataset generation. Resumable (curl -C -).
cd "$(dirname "$0")/.."
set -u
dl() { curl -L --fail --retry 5 -C - -o "$2" "$1"; }

librispeech() {
  for s in dev-clean test-clean train-clean-100; do
    if [ ! -d "data/speech/$s" ]; then
      dl "https://www.openslr.org/resources/12/$s.tar.gz" "data/downloads/$s.tar.gz" \
        && tar -xzf "data/downloads/$s.tar.gz" -C data/speech --strip-components=1 \
        && echo "LIBRISPEECH $s done" || echo "LIBRISPEECH $s FAILED"
    fi
  done
}

gunshot() {
  z=data/downloads/edge-collected-gunshot-audio.zip
  dl "https://zenodo.org/records/7004819/files/edge-collected-gunshot-audio.zip?download=1" "$z" \
    && mkdir -p data/noise/gunshot && python -c "import zipfile; zipfile.ZipFile('$z').extractall('data/noise/gunshot')" \
    && echo "GUNSHOT done" || echo "GUNSHOT FAILED"
}

librispeech &
gunshot &
wait
echo "ALL FINISHED"
