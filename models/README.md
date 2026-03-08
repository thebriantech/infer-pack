# Triton Model Repository
#
# This directory is mounted into the Triton container at /models.
# Each subdirectory is a model with a config.pbtxt and versioned
# model files.
#
# Structure:
#   models/
#     face_detection_model/
#       config.pbtxt
#       1/
#         model.onnx   (or model.plan for TensorRT)
#     face_feature_extraction_model/
#       config.pbtxt
#       1/
#         model.onnx
#
# In Phase 1, model files must be placed here manually or via a
# setup script before starting the platform.
