#!/usr/bin/env bash
# Regenerate Python gRPC stubs from the proto definition.
# Usage:  ./scripts/gen_proto.sh

set -euo pipefail
cd "$(dirname "$0")/.."

GRPC_DIR="src/inferpack/api/grpc"
PROTO_FILE="$GRPC_DIR/inferpack.proto"

python -m grpc_tools.protoc \
  -I "$GRPC_DIR" \
  --python_out="$GRPC_DIR" \
  --grpc_python_out="$GRPC_DIR" \
  --pyi_out="$GRPC_DIR" \
  "$PROTO_FILE"

# Fix the generated import so it works as a subpackage
sed -i 's/^import inferpack_pb2 as/from inferpack.api.grpc import inferpack_pb2 as/' \
  "$GRPC_DIR/inferpack_pb2_grpc.py"

echo "Proto stubs regenerated in $GRPC_DIR"
