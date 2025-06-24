#from insightface.app import FaceAnalysis

#model = FaceAnalysis(name='buffalo_s', providers=['CUDAExecutionProvider'])  # Must include this
#model.prepare(ctx_id=0)  # 0 for GPU

#print("Using providers:", model.det_model.session.get_providers())

import onnxruntime as ort
print("Available providers:", ort.get_available_providers())
