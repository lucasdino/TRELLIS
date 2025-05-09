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
from zipfile import ZipFile

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
    part = (
        f"--{BOUNDARY}\r\n"
        "Content-Type: application/json\r\n\r\n"
        + json.dumps({"status": "progress", "step": step, "message": message})
        + "\r\n"
    )
    return part.encode("utf-8")

def send_error_update(step, message):
    """Helper function to send error update to client and log it"""
    logger.error(f"   [SENDING TO CLIENT] Error message - {step}: {message}")
    part = (
        f"--{BOUNDARY}\r\n"
        "Content-Type: application/json\r\n\r\n"
        + json.dumps({"status": "error", "step": step, "message": message})
        + "\r\n"
    )
    return part.encode("utf-8")

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
        yield send_progress_update("Preprocessing", "Preparing image...")

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

        yield send_progress_update("Preprocessing", "Image processed.")

        # Run the model
        yield send_progress_update("Rendering Video", "Starting asset generation.")
        
        out = pipe.run(img, seed=0)

        # Generate turntable video
        yield send_progress_update("Rendering Video", "Cooking up a preliminary video...")
        video_path = os.path.join(tmp_dir, "preview.mp4")
        try:
            frames = render_utils.render_video(out["gaussian"][0])["color"]
            imageio.mimsave(video_path, frames, fps=30)
            logger.info(f"   [SENDING TO CLIENT] preview.mp4 on disk is {os.path.getsize(video_path)} bytes")
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
            logger.info(f"   [SENDING TO CLIENT] output.glb on disk is {os.path.getsize(glb_path)} bytes")
            # Send the GLB file
            yield send_progress_update("Generating GLB", "GLB meshed.")
            yield send_file(glb_path, "application/octet-stream", "output.glb")
        except Exception as e:
            logger.error(f"Failed to generate GLB mesh: {e}")
            traceback.print_exc()
            yield send_error_update("Generating GLB", f"Failed to generate GLB mesh: {str(e)}")

        # Send completion message
        logger.info("⮕ SENDING TO CLIENT: Process complete message")
        yield (
            f"--{BOUNDARY}\r\n"
            "Content-Type: application/json\r\n\r\n"
            + json.dumps({"status": "complete", "message": "All files generated."})
            + "\r\n"
        ).encode("utf-8")
        yield f"--{BOUNDARY}--\r\n".encode("utf-8")  # End of multipart stream

    except Exception as e:
        logger.error(f"Exception occurred during generation stream: {e}")
        traceback.print_exc()
        error_message = json.dumps({"status": "error", "step": "Streaming", "message": f"Unhandled exception during generation: {str(e)}"})
        yield (
            f"--{BOUNDARY}\r\n"
            "Content-Type: application/json\r\n\r\n"
            + error_message
            + "\r\n--{BOUNDARY}--\r\n"
        ).encode("utf-8")
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

@app.route('/generate_zip', methods=['POST'])
def generate_zip():
    if pipe is None:
        return "Model not loaded. Check server logs.", 503
    if 'file' not in request.files or request.files['file'].filename == '':
        return Response(
            json.dumps({"status": "error", "step": "Initialization", "message": "No image uploaded or filename empty."}),
            status=400, mimetype='application/json'
        )
    # Read upload and create temp dir
    uploaded = request.files['file']
    image_data = uploaded.read()
    tmp_id = uuid.uuid4().hex
    tmp_dir = os.path.join('/tmp', f'zip_{tmp_id}')
    os.makedirs(tmp_dir, exist_ok=True)
    try:
        # Save and preprocess image
        input_path = os.path.join(tmp_dir, 'input.png')
        with open(input_path, 'wb') as f:
            f.write(image_data)
        img = Image.open(input_path)
        if img.mode != 'RGBA':
            img = remove(img)
        # Run model
        out = pipe.run(img, seed=0)
        # Render video
        video_path = os.path.join(tmp_dir, 'preview.mp4')
        frames = render_utils.render_video(out['gaussian'][0])['color']
        imageio.mimsave(video_path, frames, fps=30)
        # Export GLB
        glb_path = os.path.join(tmp_dir, 'output.glb')
        mesh = postprocessing_utils.to_glb(out['gaussian'][0], out['mesh'][0], simplify=0.95, texture_size=1024)
        mesh.export(glb_path)
        # Create zip
        zip_path = os.path.join(tmp_dir, 'generated_output.zip')
        with ZipFile(zip_path, 'w') as zipf:
            zipf.write(video_path, arcname='preview.mp4')
            zipf.write(glb_path, arcname='output.glb')
        # Send zip file
        return Response(
            open(zip_path, 'rb'),
            mimetype='application/zip',
            headers={'Content-Disposition': 'attachment; filename="generated_output.zip"'}
        )
    finally:
        # Cleanup temp directory
        shutil.rmtree(tmp_dir, ignore_errors=True)

# ----------------------- entrypoint ------------------------------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    host = "0.0.0.0"
    print(f"\n[TRELLIS] ⮕ Server ready. Connect to http://<YOUR‑HOST>:{port}/generate\n")
    app.run(host=host, port=port, threaded=True)