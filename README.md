# ml_dev_ops_takehome

Minimal Django app with a ONNX image classifier and a synchronous inference endpoint.

## Requirements

- Python 3.12 or newer

## Install And Run

From the project root:

```bash
bash run_app.sh
```

What the script does:

- creates `.venv` if it does not exist
- installs the packages 
- downloads the SqueezeNet 1.1 ONNX model and ImageNet label file 
- starts Django on `127.0.0.1:8000`

Optional environment variables:

```bash
HOST=0.0.0.0 PORT=9000 PYTHON_BIN=python3 ./run_app.sh
```

## Use The App

Open `http://127.0.0.1:8000/`.

Upload an image in the browser UI and the app returns:

- the ONNX model name
- image dimensions
- the top three predicted ImageNet labels

## Use The API

Send a multipart `POST` request to `/api/infer/` with a file field named `image`.

Example:

```bash
curl -X POST \
  -F "image=@/absolute/path/to/example.jpg" \
  http://127.0.0.1:8000/api/infer/
```

Example response:

```json
{
  "model": "squeezenet1.1-7.onnx",
  "image": {
    "width": 1280,
    "height": 720
  },
  "tags": [
    {
      "label": "Labrador retriever",
      "score": 0.9643
    },
    {
      "label": "golden retriever",
      "score": 0.9121
    },
    {
      "label": "tennis ball",
      "score": 0.8015
    }
  ]
}
```

