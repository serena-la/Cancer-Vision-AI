# Cancer Vision AI

A Streamlit research prototype for individual histopathology image-patch classification using the user's trained ResNet-50 checkpoint and Grad-CAM.

## Project files

- `app.py` — multi-page Streamlit interface (Overview, Image analysis, Model & evaluation)
- `model.py` — ResNet-50 architecture and the notebook's Dropout(0.3) + Linear(2048, 2) head
- `inference.py` — checkpoint loading, notebook-matched preprocessing, prediction and Grad-CAM
- `evaluate.py` — computes validation metrics from the validation split
- `styles.css` — navy and teal visual theme
- `best.pt` — copy your trained checkpoint here (not included in this starter archive)

## 1. Add your trained model

Copy the `best.pt` file saved by your notebook into this project folder, beside `app.py`.

The checkpoint is expected to contain:
- `model`: model state dictionary
- `arch`: normally `"resnet50"`
- `size`: 512
- `class_to_idx`: mapping containing `"normal"` and `"tumor"`

Do not rename the class folders in a way that changes the mapping. The application reads the tumour class index from the checkpoint.

## 2. Create an environment and install dependencies

Windows PowerShell:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, you can run `.venv\Scripts\python.exe -m pip install -r requirements.txt` and `.venv\Scripts\python.exe -m streamlit run app.py`.

## 3. Run locally

```powershell
streamlit run app.py
```

Open the local URL printed in the terminal, usually `http://localhost:8501`.

## 4. Optional: generate validation metrics

Place the same held-out validation split used for your notebook evaluation in:

```text
data/
  val/
    normal/
      image_001.png
    tumor/
      image_002.png
```

Then run:

```powershell
python evaluate.py
```

This creates `evaluation_metrics.json`, which the Model & evaluation page reads. The script does not train or fine-tune the model. Do not use the training split as the validation split.

## Preprocessing reproduced from the notebook

- Resize longest image side to 512 pixels while preserving aspect ratio.
- Centre the resized image on a 512 × 512 white RGB canvas.
- Convert to tensor and normalise using ImageNet mean `[0.485, 0.456, 0.406]` and standard deviation `[0.229, 0.224, 0.225]`.
- ResNet-50 with Dropout(0.3) and a 2-output fully connected layer.
- Grad-CAM target layer: `model.layer4[-1]`.
- Tumour decision threshold: strictly greater than 0.50.

## Deployment notes

For Streamlit Community Cloud, push the code and `requirements.txt` to a GitHub repository, then deploy `app.py` from Streamlit Community Cloud. Add `best.pt` to the repository only if it is within the hosting provider's file-size limits and you are comfortable making the weights accessible to people with repository access. For larger checkpoints, use an appropriate private or external model-artifact host and adapt `CHECKPOINT_PATH`/download handling securely. Never commit patient-identifiable images, credentials, or private clinical data.

## Important limitations

This is a research and educational prototype, not a medical device. Model probabilities are not necessarily calibrated clinical risks. Grad-CAM highlights regions that influence the model score; it does not prove that a region contains tumour cells. Validation metrics only describe the supplied validation split and do not establish clinical safety, diagnostic accuracy, or generalisability.
