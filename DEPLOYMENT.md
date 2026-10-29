# Streamlit deployment checklist

1. Create a new public GitHub repository named `herbal-plant-classifier`.
2. Upload the files in this project package to the repository root, preserving `.github/workflows/` and `.streamlit/`.
3. Ensure repository **Settings → Actions → General → Workflow permissions** allows GitHub Actions to read and write repository contents. The workflow requests `contents: write`.
4. Open **Actions → Train Herbal Plant Classifier**. The initial push should start the model training workflow.
5. Wait for the workflow to succeed. Verify that the run published a release with `best_model.pt` and created files under `results/`.
6. Open Streamlit Community Cloud and deploy the new repository with main file `app.py`.
7. Upload a few test images and verify predictions before adding the live URL to the README.

The `app.py` model and source links are configured for `sumeet1402/herbal-plant-classifier`. If you choose a different repository name, update those URLs.
