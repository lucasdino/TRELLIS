#!/usr/bin/env python3
"""
Run TRELLIS as a Flask API.
POST /generate  form‑data:  file=<image>
Returns:  trellis_result.zip  (contains .glb, .ply, preview.mp4)
"""

import os, uuid, zipfile, shutil, imageio
from flask import Flask, request, send_file, abort
from PIL import Image
from rembg import remove
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils

# ----------------------- model loads once at startup --------------------------
print(" [TRELLIS] loading model … (first run downloads weights)")
pipe = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large").cuda()
print(" [TRELLIS] model ready")

app = Flask(__name__)

@app.route("/generate", methods=["POST"])
def generate():
    if "file" not in request.files or request.files["file"].filename == "":
        abort(400, "No image uploaded (multipart/form‑data ‘file=…’)")
    img = Image.open(request.files["file"].stream)
    if img.mode != "RGBA":                       # ensure alpha mask
        img = remove(img)

    out = pipe.run(img, seed=0)                  # → dict
    tmp_id  = uuid.uuid4().hex
    tmp_dir = f"/tmp/trellis_{tmp_id}"; os.makedirs(tmp_dir)
    # Turntable video ----------------------------------------------------------
    frames   = render_utils.render_video(out["gaussian"][0])["color"]
    imageio.mimsave(f"{tmp_dir}/preview.mp4", frames, fps=30)
    # GLB mesh -----------------------------------------------------------------
    glb_mesh = postprocessing_utils.to_glb(out["gaussian"][0],
                                           out["mesh"][0],
                                           simplify=0.95, texture_size=1024)
    glb_mesh.export(f"{tmp_dir}/output.glb")
    # Point cloud --------------------------------------------------------------
    out["gaussian"][0].save_ply(f"{tmp_dir}/output.ply")
    # Zip everything -----------------------------------------------------------
    zip_path = shutil.make_archive(tmp_dir, "zip", tmp_dir)
    return send_file(zip_path,
                     mimetype="application/zip",
                     as_attachment=True,
                     download_name="trellis_result.zip")

# ----------------------- entrypoint ------------------------------------------
if __name__ == "__main__":
    # Vast.ai places the container’s external port number in $PORT
    port = int(os.getenv("PORT", "5000"))
    host = "0.0.0.0"
    print(f"\n[TRELLIS] ⇨  connect to  http://<YOUR‑HOST>:{port}/generate\n")
    app.run(host=host, port=port, threaded=True)
