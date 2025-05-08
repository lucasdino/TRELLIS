#!/usr/bin/env python3
"""
Run TRELLIS as a Flask API.
POST /generate  form‑data:  file=<image>
Returns:  multipart/x-mixed-replace stream with JSON progress updates, preview video and GLB model
"""

import os, uuid, shutil, imageio, traceback, json, logging
from flask import Flask, request, abort, Response
from PIL import Image
from rembg import remove
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ----------------------- model loads once at startup --------------------------
print(" [TRELLIS] Loading model.")
try:
    pipe = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
    pipe.cuda()
    print(" [TRELLIS] Model ready.")
except Exception as e:
    print(f" [ERROR] Failed to load model: {e}")
    traceback.print_exc()
    pipe = None

app = Flask(__name__)

BOUNDARY = "frame"

def send_progress_update(step, message):
    """Helper function to send progress update to client and log it"""
    logger.info(f"   [SENDING TO CLIENT] Progress update - {step}: {message}")
    return f"--{BOUNDARY}\r\nContent-Type: application/json\r\n\r\n" + \
           json.dumps({"status": "progress", "step": step, "message": message}) + \
           "\r\n"

def send_error_update(step, message):
    """Helper function to send error update to client and log it"""
    logger.error(f"   [SENDING TO CLIENT] Error message - {step}: {message}")
    return f"--{BOUNDARY}\r\nContent-Type: application/json\r\n\r\n" + \
           json.dumps({"status": "error", "step": step, "message": message}) + \
           "\r\n"

def send_file(file_path, mime_type, filename):
    """Helper function to send a file to the client and log it"""
    try:
        with open(file_path, 'rb') as f:
            file_bytes = f.read()
        file_size_kb = len(file_bytes) / 1024
        logger.info(f"   [SENDING TO CLIENT] File {filename} ({file_size_kb:.2f}KB) with MIME type {mime_type}")
        return f"--{BOUNDARY}\r\nContent-Type: {mime_type}\r\nContent-Disposition: attachment; filename=\"{filename}\"\r\n\r\n".encode() + \
               file_bytes + b"\r\n"
    except Exception as e:
        logger.error(f"Failed to send file {filename}: {e}")
        raise

# Generator function for streaming response
def generate_frames_impl(image_data):
    tmp_dir = None
    temp_file_path = None
    try:
        # First yield: progress update
        yield send_progress_update("Preprocessing", "Receiving and preparing image...")

        # Create temporary directory for results
        tmp_id = uuid.uuid4().hex
        tmp_dir = os.path.join("/tmp", f"trellis_{tmp_id}")
        os.makedirs(tmp_dir, exist_ok=True)
        
        # Save the image data to a temporary file
        temp_file_path = os.path.join(tmp_dir, "input_image.png")
        with open(temp_file_path, "wb") as f:
            f.write(image_data)
        
        # Read and preprocess image from the saved file
        img = Image.open(temp_file_path)
        
        if img.mode != "RGBA":
            img = remove(img)

        yield send_progress_update("Preprocessing", "Image preprocessing complete.")

        # Run the model
        yield send_progress_update("Generation", "Starting 3D model generation...")
        
        out = pipe.run(img, seed=0) # pipe is globally accessible

        yield send_progress_update("Generation", "3D model generation complete. Processing outputs...")

        # Generate turntable video
        yield send_progress_update("Rendering Video", "Rendering turntable video...")
        video_path = os.path.join(tmp_dir, "preview.mp4")
        try:
            frames = render_utils.render_video(out["gaussian"][0])["color"]
            imageio.mimsave(video_path, frames, fps=30)
            
            # Send the video file
            yield send_file(video_path, "video/mp4", "preview.mp4")
            yield send_progress_update("Rendering Video", "Turntable video complete.")
        except Exception as e:
            logger.error(f"Failed to render turntable video: {e}")
            traceback.print_exc()
            yield send_error_update("Rendering Video", f"Failed to render turntable video: {str(e)}")

        # Generate GLB mesh
        yield send_progress_update("Generating GLB", "Generating GLB mesh...")
        glb_path = os.path.join(tmp_dir, "output.glb")
        try:
            glb_mesh = postprocessing_utils.to_glb(out["gaussian"][0],
                                                  out["mesh"][0],
                                                  simplify=0.95,
                                                  texture_size=1024)
            glb_mesh.export(glb_path)
            
            # Send the GLB file
            yield send_file(glb_path, "application/octet-stream", "output.glb")
            yield send_progress_update("Generating GLB", "GLB mesh complete.")
        except Exception as e:
            logger.error(f"Failed to generate GLB mesh: {e}")
            traceback.print_exc()
            yield send_error_update("Generating GLB", f"Failed to generate GLB mesh: {str(e)}")

        # Send completion message
        logger.info("⮕ SENDING TO CLIENT: Process complete message")
        yield f"--{BOUNDARY}\r\nContent-Type: application/json\r\n\r\n" + \
              json.dumps({"status": "complete", "message": "All files generated."}) + \
              "\r\n"
        yield f"--{BOUNDARY}--\r\n" # End of multipart stream

    except Exception as e:
        logger.error(f"Exception occurred during generation stream: {e}")
        traceback.print_exc()
        error_message = json.dumps({"status": "error", "step": "Streaming", "message": f"Unhandled exception during generation: {str(e)}"})
        yield f"--{BOUNDARY}\r\nContent-Type: application/json\r\n\r\n{error_message}\r\n--{BOUNDARY}--\r\n"
    finally:
        if tmp_dir and os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)


@app.route("/generate", methods=["POST"])
def generate():
    if pipe is None:
        logger.error("Model not loaded")
        return "Model not loaded. Check server logs.", 503

    if "file" not in request.files or request.files["file"].filename == "":
        logger.error("No file in request or empty filename")
        error_payload = json.dumps({
            "status": "error",
            "step": "Initialization",
            "message": "No image uploaded or filename empty. Please include a file in your request."
        })
        return Response(error_payload, status=400, mimetype="application/json")

    try:
        uploaded_file = request.files["file"]
        logger.info(f"Received file upload: {uploaded_file.filename}")
        
        # Read the file data immediately to prevent stream closure issues
        image_data = uploaded_file.read()
        logger.info(f"Starting to stream response for request")
        
        # Call the implementation function with the image data
        return Response(generate_frames_impl(image_data), mimetype=f"multipart/x-mixed-replace; boundary={BOUNDARY}")
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        traceback.print_exc()
        error_payload = json.dumps({
            "status": "error",
            "step": "Initialization",
            "message": f"Error processing request: {str(e)}"
        })
        return Response(error_payload, status=500, mimetype="application/json")

# ----------------------- entrypoint ------------------------------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    host = "0.0.0.0"
    print(f"\n[TRELLIS] ⮕ Server ready. Connect to http://<YOUR‑HOST>:{port}/generate\n")
    app.run(host=host, port=port, threaded=True)