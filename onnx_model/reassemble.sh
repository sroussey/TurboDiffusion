#!/bin/bash
# Reassemble wan_dit.onnx.data from split parts.
# Run from the onnx_model/ directory.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT="$SCRIPT_DIR/wan_dit.onnx.data"

if [ -f "$OUTPUT" ]; then
    echo "wan_dit.onnx.data already exists. Remove it first to reassemble."
    exit 1
fi

echo "Reassembling wan_dit.onnx.data from parts..."
cat "$SCRIPT_DIR"/wan_dit.onnx.data.part_* > "$OUTPUT"

echo "Done. File size: $(du -h "$OUTPUT" | cut -f1)"
echo ""
echo "To use the ONNX model:"
echo "  import onnxruntime as ort"
echo "  sess = ort.InferenceSession('onnx_model/wan_dit.onnx')"
