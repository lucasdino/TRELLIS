#!/usr/bin/env python3
"""
Run TRELLIS as a Flask API.
POST /generate  form‑data:  file=<image>
Returns:  trellis_result.zip  (contains .glb, .ply, preview.mp4)
"""

import os, uuid, shutil, imageio, traceback, json
from flask import Flask, request, abort, Response
from PIL import Image
from rembg import remove
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils

# ----------------------- model loads once at startup --------------------------
print(" [TRELLIS] Loading model … (first run downloads weights)")
try:
    pipe = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
    pipe.cuda()
    print(" [TRELLIS] Model ready")
except Exception as e:
    print(f" [ERROR] Failed to load model: {e}")
    traceback.print_exc()
    pipe = None

app = Flask(__name__)

BOUNDARY = "frame"

def generate_frames():
    tmp_dir = None
    try:
        # Check for file in request
        if "file" not in request.files or request.files["file"].filename == "":
            print("[ERROR] No file in request")
            # This error happens before we can stream, so abort is still okay.
            # For streaming errors, we'd yield a JSON error.
            abort(400, "No image uploaded (multipart/form‑data ‘file=…’)")

        yield f"--{BOUNDARY}\r\nContent-Type: application/json\r\n\r\n" + \
              json.dumps({"status": "progress", "step": "Preprocessing", "message": "Receiving and preparing image..."}) + \
              "\r\n"

        # Read and preprocess image
        img_file = request.files["file"].stream
        img = Image.open(img_file)
        
        if img.mode != "RGBA":
            img = remove(img)

        yield f"--{BOUNDARY}\r\nContent-Type: application/json\r\n\r\n" + \
              json.dumps({"status": "progress", "step": "Preprocessing", "message": "Image preprocessing complete."}) + \
              "\r\n"

        # Run the model
        yield f"--{BOUNDARY}\r\nContent-Type: application/json\r\n\r\n" + \
              json.dumps({"status": "progress", "step": "Generation", "message": "Starting 3D model generation..."}) + \
              "\r\n"
        
        out = pipe.run(img, seed=0)

        yield f"--{BOUNDARY}\r\nContent-Type: application/json\r\n\r\n" + \
              json.dumps({"status": "progress", "step": "Generation", "message": "3D model generation complete. Processing outputs..."}) + \
              "\r\n"

        # Create temporary directory for results
        tmp_id = uuid.uuid4().hex
        tmp_dir = f"/tmp/trellis_{tmp_id}" # Consider making this OS-agnostic, e.g. tempfile.mkdtemp()
        os.makedirs(tmp_dir, exist_ok=True)

        # Generate turntable video
        yield f"--{BOUNDARY}\r\nContent-Type: application/json\r\n\r\n" + \
              json.dumps({"status": "progress", "step": "Rendering Video", "message": "Rendering turntable video..."}) + \
              "\r\n"
        video_path = os.path.join(tmp_dir, "preview.mp4")
        try:
            frames = render_utils.render_video(out["gaussian"][0])["color"]
            imageio.mimsave(video_path, frames, fps=30)
            with open(video_path, 'rb') as f:
                video_bytes = f.read()
            yield f"--{BOUNDARY}\r\nContent-Type: video/mp4\r\nContent-Disposition: attachment; filename=\"preview.mp4\"\r\n\r\n".encode() + \
                  video_bytes + b"\r\n"
            yield f"--{BOUNDARY}\r\nContent-Type: application/json\r\n\r\n" + \
                  json.dumps({"status": "progress", "step": "Rendering Video", "message": "Turntable video complete."}) + \
                  "\r\n"
        except Exception as e:
            print(f"[ERROR] Failed to render turntable video: {e}")
            traceback.print_exc()
            yield f"--{BOUNDARY}\r\nContent-Type: application/json\r\n\r\n" + \
                  json.dumps({"status": "error", "step": "Rendering Video", "message": f"Failed to render turntable video: {str(e)}"}) + \
                  "\r\n"


        # Generate GLB mesh
        yield f"--{BOUNDARY}\r\nContent-Type: application/json\r\n\r\n" + \
              json.dumps({"status": "progress", "step": "Generating GLB", "message": "Generating GLB mesh..."}) + \
              "\r\n"
        glb_path = os.path.join(tmp_dir, "output.glb")
        try:
            glb_mesh = postprocessing_utils.to_glb(out["gaussian"][0],
                                                   out["mesh"][0],
                                                   simplify=0.95,
                                                   texture_size=1024)
            glb_mesh.export(glb_path)
            with open(glb_path, 'rb') as f:
                glb_bytes = f.read()
            yield f"--{BOUNDARY}\r\nContent-Type: application/octet-stream\r\nContent-Disposition: attachment; filename=\"output.glb\"\r\n\r\n".encode() + \
                  glb_bytes + b"\r\n"
            yield f"--{BOUNDARY}\r\nContent-Type: application/json\r\n\r\n" + \
                  json.dumps({"status": "progress", "step": "Generating GLB", "message": "GLB mesh complete."}) + \
                  "\r\n"
        except Exception as e:
            print(f"[ERROR] Failed to generate GLB mesh: {e}")
            traceback.print_exc()
            yield f"--{BOUNDARY}\r\nContent-Type: application/json\r\n\r\n" + \
                  json.dumps({"status": "error", "step": "Generating GLB", "message": f"Failed to generate GLB mesh: {str(e)}"}) + \
                  "\r\n"

        # Save point cloud
        yield f"--{BOUNDARY}\r\nContent-Type: application/json\r\n\r\n" + \
              json.dumps({"status": "progress", "step": "Generating PLY", "message": "Generating PLY file..."}) + \
              "\r\n"
        ply_path = os.path.join(tmp_dir, "output.ply")
        try:
            out["gaussian"][0].save_ply(ply_path)
            with open(ply_path, 'rb') as f:
                ply_bytes = f.read()
            yield f"--{BOUNDARY}\r\nContent-Type: application/octet-stream\r\nContent-Disposition: attachment; filename=\"output.ply\"\r\n\r\n".encode() + \
                  ply_bytes + b"\r\n"
            yield f"--{BOUNDARY}\r\nContent-Type: application/json\r\n\r\n" + \
                  json.dumps({"status": "progress", "step": "Generating PLY", "message": "PLY file complete."}) + \
                  "\r\n"
        except Exception as e:
            print(f"[ERROR] Failed to save PLY file: {e}")
            traceback.print_exc()
            yield f"--{BOUNDARY}\r\nContent-Type: application/json\r\n\r\n" + \
                  json.dumps({"status": "error", "step": "Generating PLY", "message": f"Failed to save PLY file: {str(e)}"}) + \
                  "\r\n"

        yield f"--{BOUNDARY}\r\nContent-Type: application/json\r\n\r\n" + \
              json.dumps({"status": "complete", "message": "All files generated."}) + \
              "\r\n"
        yield f"--{BOUNDARY}--\r\n" # End of multipart stream

    except Exception as e:
        print(f"[ERROR] Exception occurred during generation stream: {e}")
        traceback.print_exc()
        # Ensure the boundary is properly terminated if an error occurs mid-stream
        # Note: The client might have already disconnected or might not parse this correctly if the stream is broken.
        error_message = json.dumps({"status": "error", "step": "Streaming", "message": f"Unhandled exception during generation: {str(e)}"})
        yield f"--{BOUNDARY}\r\nContent-Type: application/json\r\n\r\n{error_message}\r\n--{BOUNDARY}--\r\n"
    finally:
        if tmp_dir and os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)


@app.route("/generate", methods=["POST"])
def generate():
    if pipe is None:
        return "Model not loaded. Check server logs.", 503
    return Response(generate_frames(), mimetype=f"multipart/x-mixed-replace; boundary={BOUNDARY}")

# ----------------------- entrypoint ------------------------------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    host = "0.0.0.0"
    print(f"\n[TRELLIS] ⇨  connect to  http://<YOUR‑HOST>:{port}/generate\n")
    app.run(host=host, port=port, threaded=True)