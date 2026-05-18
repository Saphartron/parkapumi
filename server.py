from flask import Flask, Response, render_template_string

from vision import generate_frames

app = Flask(__name__)

HTML = """
<!DOCTYPE html>
<html>
<head>

<style>

body{
    margin:0;
    overflow:hidden;
    background:black;
}

.container{
    display:flex;
    width:100vw;
    height:100vh;
}

.left{
    width:70%;
    background:black;
}

.right{
    width:30%;
    background:#2a2a2a;
}

img{
    width:100%;
    height:100%;
    object-fit:contain;
}

</style>

</head>

<body>

<div class="container">

    <div class="left">
        <img src="/video_feed">
    </div>

    <div class="right"></div>

</div>

</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/video_feed")
def video_feed():

    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        threaded=True,
        debug=False
    )