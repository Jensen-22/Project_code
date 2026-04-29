# Plant Spectrum Health Web App

Mobile-first web app for the backend code and assets in `final_model_imgs`.

## Run

```bash
cd /Users/jensen/Desktop/final_model_imgs/mobile_spectrum_app
python3 server.py --port 8000
```

Then open:

```text
http://127.0.0.1:8000
```

## Backend

The app uses the same fixed crop and ROI values from the notebooks:

- first crop: `x=1600:2700`, `y=1500:2000`
- second crop: `x=380:900`, `y=260:390`
- extraction ROI: `x=20:500`, `y=45:110`

By default, the no-leaf reference comes from `../reference_spectrum_no_leaf.npz`.
You can also upload a fresh no-leaf reference photo in the optional field.

The health estimate is rule-based: it checks whether blue and red absorbance are high relative to green absorbance, which is the expected chlorophyll response for a healthy green leaf.
