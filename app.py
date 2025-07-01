from flask import Flask, request, render_template
from transformers import pipeline
import os

app = Flask(__name__)

detector = pipeline("image-classification", model="umm-maybe/AI-image-detector")

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST": #post -> user submission
        if "image" not in request.files:
            return "No file uploaded!", 400
        
        file = request.files["image"]
        if file.filename == "":
            return "No file selected!", 400
        
        upload_path = "static/uploaded_image.jpg"
        file.save(upload_path)
        
        #AI detection
        result = detector(upload_path)[0]
        label = result["label"]
        confidence = round(result["score"] * 100, 2)
        
        return render_template(
            "index.html",
            result=f"{label} ({confidence}%)",
            image_path=upload_path
        )
    
    return render_template("index.html", result=None)

if __name__ == "__main__":
    app.run(debug=True, port = 5001) 