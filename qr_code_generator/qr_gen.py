from flask import Flask, request, send_file
import qrcode
import io

app = Flask(__name__)

@app.route('/generate')
def generate():
    url = request.args.get('url')
    img = qrcode.make(url)
    buf = io.BytesIO()
    img.save(buf)
    buf.seek(0)
    return send_file(buf, mimetype='image/png')

if __name__ == '__main__':
    app.run()